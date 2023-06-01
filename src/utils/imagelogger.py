# Written by: Sam Sweere
import os

import pytorch_lightning as pl
import torch
import wandb
from matplotlib import cm
import numpy as np

from utils.filehandling import write_xmm_file_to_fits
from utils.ssim import ssim as get_ssim

from transforms.imageupsample import ImageUpsample
from utils.metriclogger import MetricLogger


class ImageLogger(pl.Callback):
    def __init__(
        self,
        real_samples,
        sim_samples,
        log_every_n_epochs,
        normalize,
        scaling_normalizers,
        data_range,
    ):
        super().__init__()

        self.real_samples = real_samples
        self.sim_samples = sim_samples
        self.normalize = normalize
        self.scaling_normalizers = scaling_normalizers
        self.data_range = data_range

        # Save upsampling operation that potentially has to be done
        lr_res = sim_samples[0]["lr"].shape[-1]
        hr_res = sim_samples[0]["hr"].shape[-1]

        if lr_res != hr_res:
            self.upsample = ImageUpsample(scale_factor=int(hr_res / lr_res))
        else:
            # "images_combined_cm": images_combined_cm,
            self.upsample = None
        self.metriclogger = MetricLogger(
            normalize=normalize,
            scaling_normalizers=scaling_normalizers,
            data_range=data_range,
        )

        self.log_every_n_epochs = log_every_n_epochs

    def _generate_images_scaled(self, x, pred, y, scale_normalizer):
        # Scale the images
        x = scale_normalizer.normalize_hr_image(
            x
        )  # Note that the x has been upscaled here
        pred = scale_normalizer.normalize_hr_image(pred)
        y = scale_normalizer.normalize_hr_image(y)

        # Calculate the difference images befre we denormalize the images
        # Calculate the difference of the image.
        # This is done by ((pred - img)+1)/2 such that if it is the same the value is 0.5 and
        # the difference is maximal -1, 1

        difference = ((pred - y) + 1.0) / 2.0

        # Calculate the ssim images from the denormalized images
        # SSIM settings
        winsize = 13
        sigma = 2.5
        ms_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        K = (0.01, 0.05)

        ssim_img = get_ssim(
            torch.unsqueeze(pred, 0),
            torch.unsqueeze(y, 0),
            win_size=winsize,
            win_sigma=sigma,
            data_range=self.data_range,
            K=K,
        )[1]

        return x, pred, y, difference, ssim_img

    def _save_fits_image(
        self, x, pred, label, lr_exp, hr_exp, name, metrics, generate_label
    ):
        lr_exp = lr_exp * 1000
        hr_exp = hr_exp * 1000

        # flip the images to be correct
        x = torch.flip(x, [-2])
        pred = torch.flip(pred, [-2])
        label = torch.flip(label, [-2])

        x = (x * lr_exp)[0].detach().cpu().numpy()
        pred = (pred * hr_exp)[0].detach().cpu().numpy()
        label = (label * hr_exp)[0].detach().cpu().numpy()

        output_dir = os.path.join(
            wandb.run.dir, "fits_out", name[:60]
        )  # Limit the length
        os.makedirs(output_dir)

        # Add the metric header
        metrics = metrics[0]
        # metrics_header = {}
        metricsfile = open(os.path.join(output_dir, "metrics.txt"), "w")
        for data_scaling in metrics.keys():

            for metric_name in metrics[data_scaling].keys():
                final_key = metric_name + "_" + data_scaling
                # metrics_header[final_key] = metrics[data_scaling][metric_name]

                # Write the metrics also to file
                metricsfile.write(
                    final_key + ", " + str(metrics[data_scaling][metric_name]) + "\n"
                )

        metricsfile.close()

        write_xmm_file_to_fits(
            img=x,
            output_dir=output_dir,
            source_file_name=name,
            res_mult=1,
            exposure=lr_exp.detach().item(),
            comment="Transformed Input Image (input to model)",
            out_file_name="input",
        )

        out_res_mult = int(pred.shape[0] / x.shape[0])

        write_xmm_file_to_fits(
            img=pred,
            output_dir=output_dir,
            source_file_name=name,
            res_mult=out_res_mult,
            exposure=hr_exp.detach().item(),
            comment="Model Prediction",
            out_file_name="prediction",
            in_header=None,
        )

        if generate_label:
            write_xmm_file_to_fits(
                img=label,
                output_dir=output_dir,
                source_file_name=name,
                res_mult=out_res_mult,
                exposure=hr_exp.detach().item(),
                comment="Reference Image",
                out_file_name="label",
            )

        # Save all files that currently exist in the output folder
        wandb.save(os.path.join(output_dir, "*"), base_path=output_dir)

    def _generate_images(
        self, trainer, pl_module, samples, generate_label, save_fits=False
    ):
        input_images_filenames = []
        metrics = []

        input_images = {}
        generated_images = {}
        label_images = {}
        difference_images = {}
        ssim_images = {}

        for scale_normalizer in self.scaling_normalizers:
            stretch_name = scale_normalizer.stretch_mode
            input_images[stretch_name] = []
            generated_images[stretch_name] = []
            label_images[stretch_name] = []
            difference_images[stretch_name] = []
            ssim_images[stretch_name] = []

        pl_module.eval()

        for sample in samples:
            imgs = sample["lr"]
            imgs = imgs.to(device=pl_module.device)

            img_filenames = sample["lr_img_file_name"]
            # labels = sample['hr']

            preds = pl_module(imgs)

            # Generate the combined images
            # for x, pred, y in zip(sample['lr_gt'], preds, sample['hr_gt']):
            ## Ivan change
            if generate_label:
                label = sample["hr"]
                hr_exps = sample["hr_exp"]
            else:
            # If we do not want the label images, make the label images the pred images, this simplifies the code
            # Since we can later not send the label images
                label = preds
                hr_exps = sample["lr_exp"]

            for x, pred, y, file_name, lr_exp, hr_exp in zip(
                imgs, preds, label, img_filenames, sample["lr_exp"], hr_exps
            ):
                x = x.to(pl_module.device)

                if len(x.shape) == 3 and x.shape[0] >= 2:
                    # Remove the exposure mask layer
                    x = x[0, :, :].reshape(1, x.shape[1], x.shape[2])

                if len(x.shape) >= 4 and x.shape[1] >= 2:
                    # Remove the exposure mask layer
                    x = x[:, 0, :, :].reshape(x.shape[0], 1, x.shape[2], x.shape[3])

                pred = pred.to(pl_module.device)

                # Remove the exposure mask if it is present from the second layer
                if x.shape[0] >= 2:
                    x = x[0:1, :, :]

                if generate_label:
                    y = y.to(pl_module.device)

                # flip the images to be correct
                x = torch.flip(x, [-2])
                pred = torch.flip(pred, [-2])
                y = torch.flip(y, [-2])

                # Denormalize the images in order to show the final output images
                x = self.normalize.denormalize_lr_image(x)

                if save_fits:
                    # Clone the original output for the savefits
                    x_org = x.clone()

                pred = self.normalize.denormalize_hr_image(pred)

                y = self.normalize.denormalize_hr_image(y)

                # Upscale the input image to be able to compare it
                if self.upsample:
                    x = self.upsample(x)

                metrics_gen = self.metriclogger.get_metrics(x, pred, y, log_inputs=True)

                if save_fits:
                    self._save_fits_image(
                        x=x_org,
                        pred=pred,
                        label=y,
                        lr_exp=lr_exp,
                        hr_exp=hr_exp,
                        name=file_name,
                        metrics=metrics_gen,
                        generate_label=generate_label,
                    )

                metrics += metrics_gen

                input_images_filenames.append(file_name)

                for scale_normalizer in self.scaling_normalizers:
                    stretch_name = scale_normalizer.stretch_mode
                    scaling_f = scale_normalizer.stretch_f

                    x_s, pred_s, y_s, diff_s, ssim_img_s = self._generate_images_scaled(
                        x=x, pred=pred, y=y, scale_normalizer=scale_normalizer
                    )
                    input_images[stretch_name].append(x_s)
                    generated_images[stretch_name].append(pred_s)
                    label_images[stretch_name].append(y_s)
                    difference_images[stretch_name].append(diff_s)
                    ssim_images[stretch_name].append(ssim_img_s)

        # Sort all the images based on the filename, this way they always show in the same order over multiple runs
        # (The dataloader seems not to be determnistic under different batch sizes)
        # Sort the lists based on the img_filenames
        img_filenames_arr = np.array(input_images_filenames)
        inds = list(img_filenames_arr.argsort())

        for scale_normalizer in self.scaling_normalizers:
            stretch_name = scale_normalizer.stretch_mode

            input_images[stretch_name] = [input_images[stretch_name][i] for i in inds]
            generated_images[stretch_name] = [
                generated_images[stretch_name][i] for i in inds
            ]
            label_images[stretch_name] = [label_images[stretch_name][i] for i in inds]
            difference_images[stretch_name] = [
                difference_images[stretch_name][i] for i in inds
            ]
            ssim_images[stretch_name] = [ssim_images[stretch_name][i] for i in inds]

        input_images_cm = {}
        generated_images_cm = {}
        label_images_cm = {}
        ssim_images_cm = {}
        difference_images_cm = {}

        for scale_normalizer in self.scaling_normalizers:
            stretch_name = scale_normalizer.stretch_mode

            # Convert the images to plasma colormap
            plasma = cm.get_cmap("plasma")
            input_images_cm[stretch_name] = self.convert_images_to_cm(
                plasma, input_images[stretch_name], normalize=False
            )

            generated_images_cm[stretch_name] = self.convert_images_to_cm(
                plasma, generated_images[stretch_name], normalize=False
            )

            label_images_cm[stretch_name] = self.convert_images_to_cm(
                plasma, label_images[stretch_name], normalize=False
            )

            # Choose a colormap for the ssim difference images
            viridis_r = cm.get_cmap("seismic")
            ssim_images_cm[stretch_name] = self.convert_images_to_cm(
                viridis_r, ssim_images[stretch_name]
            )

            # Choose a colormap diverging colormap
            seismic = cm.get_cmap("seismic")
            difference_images_cm[stretch_name] = self.convert_images_to_cm(
                seismic, difference_images[stretch_name]
            )

        # Combine all the generated images into a dict file
        images_dict = {
            "input_images_cm": input_images_cm,
            "generated_images_cm": generated_images_cm,
            "label_images_cm": label_images_cm,
            "difference_images_cm": difference_images_cm,
            "ssim_images_cm": ssim_images_cm,
        }

        return images_dict, metrics

    def apply_cm(self, colormap, image, scale=None, normalize=False):
        image = np.array(image.detach().cpu())[0]
        if scale:
            image = scale(image)

        if normalize:
            # Normalize the image to 0.0 and 1.0
            image = image / np.max(image)

        return colormap(image)

    def convert_images_to_cm(self, colormap, images, scale=None, normalize=False):
        images_cm = []
        for image in images:
            if type(image) == dict:
                images_cm_dict = {}
                for key in image:
                    images_cm_dict[key] = self.apply_cm(
                        colormap, image[key], scale=scale, normalize=normalize
                    )
                images_cm.append(images_cm_dict)
            else:
                images_cm.append(
                    self.apply_cm(colormap, image, scale=scale, normalize=normalize)
                )

        return images_cm

    def _log_images(
        self, trainer, pl_module, images_dict, metrics, stage, generate_label
    ):

        image_log_dict = {"global_step": trainer.global_step, "commit": False}

        for scale_normalizer in self.scaling_normalizers:
            stretch_name = scale_normalizer.stretch_mode

            # Colormap images
            image_log_dict[stage + "/input images " + stretch_name] = [
                wandb.Image(
                    x,
                    caption=f"input {round(metrics[stretch_name]['psnr_in'], 2)} / {round(metrics[stretch_name]['ssim_in'], 4)}",
                )
                for x, metrics in zip(
                    images_dict["input_images_cm"][stretch_name], metrics
                )
            ]

            image_log_dict[
                stage + "/generated images " + stretch_name + " (psnr/ssim)"
            ] = [
                wandb.Image(
                    x,
                    caption=f"generated {round(metrics[stretch_name]['psnr'], 2)} / {round(metrics[stretch_name]['ssim'], 4)}",
                )
                for x, metrics in zip(
                    images_dict["generated_images_cm"][stretch_name], metrics
                )
            ]

            if generate_label:
                image_log_dict[stage + "/label images " + stretch_name] = [
                    wandb.Image(x, caption="label")
                    for x in images_dict["label_images_cm"][stretch_name]
                ]

            image_log_dict[stage + "/difference images " + stretch_name] = [
                wandb.Image(x, caption="difference")
                for x in images_dict["difference_images_cm"][stretch_name]
            ]

            image_log_dict[stage + "/ssim map images " + stretch_name + " (ssim)"] = [
                wandb.Image(x, caption=f"{round(metrics[stretch_name]['ssim'], 6)}")
                for x, metrics in zip(
                    images_dict["ssim_images_cm"][stretch_name], metrics
                )
            ]

        trainer.logger.experiment.log(image_log_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only log the images every n epochs to reduce the amount of generated data
        # print("In val epoch end, epoch:", trainer.current_epoch)
        # try:

        # print("Current epoch:", trainer.current_epoch)
        # print("self.log_every_n_epochs:", self.log_every_n_epochs)
        # print("trainer.current_epoch % self.log_every_n_epochs != 0:", trainer.current_epoch % self.log_every_n_epochs != 0)

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            # print("Not logging images")
            return

        # print("Logging images")

        sim_images_dict, sim_metrics = self._generate_images(
            trainer, pl_module, samples=self.sim_samples, generate_label=True
        )
        real_images_dict, real_metrics = self._generate_images(
            trainer, pl_module, samples=self.real_samples, generate_label=True
        )
        self._log_images(
            trainer,
            pl_module,
            sim_images_dict,
            sim_metrics,
            stage="val_sim",
            generate_label=True,
        )
        self._log_images(
            trainer,
            pl_module,
            real_images_dict,
            real_metrics,
            stage="val_real",
            generate_label=True,
        )
        # except Exception as e:
        #     print("Failed to log validation images: ", e)

    def on_test_epoch_end(self, trainer, pl_module):
        sim_images_dict, sim_metrics = self._generate_images(
            trainer,
            pl_module,
            samples=self.sim_samples,
            generate_label=True,
            save_fits=True,
        )
        real_images_dict, real_metrics = self._generate_images(
            trainer,
            pl_module,
            samples=self.real_samples,
            generate_label=False,
            save_fits=True,
        )
        self._log_images(
            trainer,
            pl_module,
            sim_images_dict,
            sim_metrics,
            stage="test_sim",
            generate_label=True,
        )
        self._log_images(
            trainer,
            pl_module,
            real_images_dict,
            real_metrics,
            stage="test_real",
            generate_label=True,
        )
