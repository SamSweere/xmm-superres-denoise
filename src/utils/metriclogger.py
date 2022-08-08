import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import piq

from utils.ssim import ssim as get_ssim
from utils.ssim import ms_ssim as get_ms_ssim
from transforms.imageupsample import ImageUpsample


class MetricLogger(pl.Callback):
    def __init__(self, normalize, scaling_normalizers, data_range):
        super().__init__()

        # Save upsampling operation that potentially has to be done
        self.upsample = None

        self.normalize = normalize
        self.scaling_normalizers = scaling_normalizers

        self.data_range = data_range

    def log_metrics(
        self,
        input,
        gen,
        label,
        pl_module,
        stage,
        name_ext="",
        tng_set=None,
        log_inputs=False,
        log_extended=False,
    ):
        # Get the metrics, format is a dict
        metrics_scaled = self.get_metrics(
            input, gen, label, tng_set, log_inputs, log_extended
        )

        metrics = {}

        # Add _function after the metric to show that it is scaled

        for scale_normalizer in self.scaling_normalizers:
            stretch_name = scale_normalizer.stretch_mode

            metrics["tng_set"] = []

            for metric_sub_dict in metrics_scaled:
                metric_sub_dict = metric_sub_dict[stretch_name]

                for key in list(metric_sub_dict.keys()):
                    if key == "tng_set":
                        metrics["tng_set"].append(metric_sub_dict["tng_set"])

                        continue

                    if stretch_name == "linear":
                        new_key = key
                    else:
                        new_key = key + "_" + stretch_name

                    if new_key in metrics.keys():
                        metrics[new_key].append(metric_sub_dict[key])
                    else:
                        metrics[new_key] = [metric_sub_dict[key]]
                # metrics[new_key] = sqrt_metrics.pop(key)

        # Log the metrics
        for name in metrics:
            if name == "tng_set":
                # Skip the tng set names
                continue

            values = metrics[name]
            log_name = stage + "/" + name + name_ext
            for value in values:
                pl_module.log(log_name, value, on_step=False, on_epoch=True)

            if "tng_set" in metrics:
                # Log them one for one to log the tng_set names
                for i in range(len(metrics["tng_set"])):
                    tng_set_name = metrics["tng_set"][i]

                    value = values[i]

                    log_name = stage + "/" + tng_set_name + "/" + name + name_ext
                    pl_module.log(log_name, value, on_step=False, on_epoch=True)

    def _get_metric_score(
        self,
        input,
        gen,
        label,
        data_range,
        tng_set=None,
        log_input=False,
        log_extended=False,
    ):
        # data_range = self.data_range

        # try:
        # Metrics calculated using package https://github.com/photosynthesis-team/piq
        psnr = piq.psnr(gen, label, data_range=data_range).detach().item()
        if log_input:
            psnr_in = (
                piq.psnr(input, label, data_range=data_range).detach().item()
            )  # Relative psnr in comparison to the input

        winsize = 13
        sigma = 2.5
        ms_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        K = (0.01, 0.05)

        ssim = (
            get_ssim(
                gen,
                label,
                win_size=winsize,
                win_sigma=sigma,
                data_range=data_range,
                K=K,
            )[0]
            .detach()
            .item()
        )
        if log_input:
            ssim_in = (
                get_ssim(
                    input,
                    label,
                    win_size=winsize,
                    win_sigma=sigma,
                    data_range=data_range,
                    K=K,
                )[0]
                .detach()
                .item()
            )

        if gen.shape[-1] > 192:
            # We can only calculate ms_ssim if the size is bigger than 192 due to the different scales in ms_ssim
            ms_ssim = (
                get_ms_ssim(
                    gen,
                    label,
                    win_size=winsize,
                    win_sigma=sigma,
                    data_range=data_range,
                    weights=ms_weights,
                    K=K,
                )
                .detach()
                .item()
            )

            if log_input:
                ms_ssim_in = (
                    get_ms_ssim(
                        input,
                        label,
                        win_size=winsize,
                        win_sigma=sigma,
                        data_range=data_range,
                        weights=ms_weights,
                        K=K,
                    )
                    .detach()
                    .item()
                )

        l1 = nn.L1Loss()(gen, label).detach().item()
        if log_input:
            l1_in = nn.L1Loss()(input, label).detach().item()

        l2 = nn.MSELoss()(gen, label).detach().item()
        if log_input:
            l2_in = nn.MSELoss()(input, label).detach().item()

        poisson = nn.PoissonNLLLoss(log_input=False)(gen, label).detach().item()
        if log_input:
            poisson_in = (
                nn.PoissonNLLLoss(log_input=False)(input, label).detach().item()
            )

        metrics = {"psnr": psnr, "ssim": ssim, "l1": l1, "l2": l2, "poisson": poisson}

        metrics_in = {}
        if log_input:
            metrics_in = {
                "psnr_in": psnr_in,
                "ssim_in": ssim_in,
                "l1_in": l1_in,
                "l2_in": l2_in,
                "poisson_in": poisson_in,
            }

        if log_extended:
            vif_p = piq.vif_p(x=gen, y=label, data_range=data_range).detach().item()
            if log_input:
                vif_p_in = (
                    piq.vif_p(x=input, y=label, data_range=data_range).detach().item()
                )

            fsim = (
                piq.fsim(x=gen, y=label, data_range=data_range, chromatic=False)
                .detach()
                .item()
            )
            if log_input:
                fsim_in = (
                    piq.fsim(x=input, y=label, data_range=data_range, chromatic=False)
                    .detach()
                    .item()
                )

            gmsd = piq.gmsd(gen, label, data_range=data_range).detach().item()
            if log_input:
                gmsd_in = piq.gmsd(input, label, data_range=data_range).detach().item()

            ms_gmsd = (
                piq.multi_scale_gmsd(gen, label, data_range=data_range, chromatic=False)
                .detach()
                .item()
            )
            if log_input:
                ms_gmsd_in = (
                    piq.multi_scale_gmsd(
                        input, label, data_range=data_range, chromatic=False
                    )
                    .detach()
                    .item()
                )

            haarpsi = piq.haarpsi(x=gen, y=label, data_range=data_range).detach().item()
            if log_input:
                haarpsi_in = (
                    piq.haarpsi(x=input, y=label, data_range=data_range).detach().item()
                )

            mdsi = piq.mdsi(x=gen, y=label, data_range=data_range).detach().item()
            if log_input:
                mdsi_in = (
                    piq.mdsi(x=input, y=label, data_range=data_range).detach().item()
                )

            metrics["vif_p"] = vif_p
            metrics["fsim"] = fsim
            metrics["gmsd"] = gmsd
            metrics["ms_gmsd"] = ms_gmsd
            metrics["haarpsi"] = haarpsi
            metrics["mdsi"] = mdsi

            if log_input:
                metrics_in["vif_p_in"] = vif_p_in
                metrics_in["fsim_in"] = fsim_in
                metrics_in["gmsd_in"] = gmsd_in
                metrics_in["ms_gmsd_in"] = ms_gmsd_in
                metrics_in["haarpsi_in"] = haarpsi_in
                metrics_in["mdsi_in"] = mdsi_in

        metrics = {**metrics, **metrics_in}

        if gen.shape[-1] > 192:
            metrics["ms_ssim"] = ms_ssim
            if log_input:
                metrics["ms_ssim_in"] = ms_ssim_in

        if tng_set is not None:
            metrics["tng_set"] = tng_set

        return metrics

    def get_metrics(
        self, input, gen, label, tng_set=None, log_inputs=False, log_extended=False
    ):
        # Remove the exposure mask if it is present from the second layer
        if len(input.shape) == 4 and input.shape[1] >= 2:
            input = input[:, 0:1, :, :]

        if input.shape[-1] != gen.shape[-1]:
            if not self.upsample:
                # Create the upsample class
                self.upsample = ImageUpsample(
                    scale_factor=int(gen.shape[-1] / input.shape[-1])
                )

            input = self.upsample(input)

        # This should be the case anyway, but sometimes a small negative number gets through somehow
        # Maybe because of the upsample
        input = torch.clamp(input, min=0.0, max=self.data_range)
        gen = torch.clamp(gen, min=0.0, max=self.data_range)
        label = torch.clamp(label, min=0.0, max=self.data_range)

        if len(input.shape) == 3:
            # Extend the tensor
            input = torch.unsqueeze(input, 0)
        if len(gen.shape) == 3:
            # Extend the tensor
            gen = torch.unsqueeze(gen, 0)
        if len(label.shape) == 3:
            # Extend the tensor
            label = torch.unsqueeze(label, 0)

        metrics = []

        for i in range(input.shape[0]):
            # Only add the tng_set once (otherwise it will be a list)
            tng_select = None
            if tng_set is not None:
                tng_select = tng_set[i]

            scaled_metrics = {}

            for scale_normalizer in self.scaling_normalizers:
                stretch_name = scale_normalizer.stretch_mode
                # scaling_f = scale_normalizer.stretch_f

                in_s = scale_normalizer.normalize_hr_image(input[i])
                gen_s = scale_normalizer.normalize_hr_image(gen[i])
                label_s = scale_normalizer.normalize_hr_image(label[i])

                scaled_metrics[stretch_name] = self._get_metric_score(
                    torch.unsqueeze(
                        in_s, 0
                    ),  # Note that the input has been scaled up here
                    torch.unsqueeze(gen_s, 0),
                    torch.unsqueeze(label_s, 0),
                    data_range=1.0,
                    tng_set=tng_select,
                    log_input=log_inputs,
                    log_extended=log_extended,
                )

            metrics.append(scaled_metrics)

        return metrics

    def log_metrics_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        stage,
        log_inputs=False,
        log_extended=False,
    ):
        # input = batch['lr_gt']
        input = batch["lr"]
        gen = outputs

        # label = outputs[1]
        # label = batch['hr_gt']
        label = batch["hr"]

        # Add the tng set if it exists
        tng_set = None
        if "tng_set" in batch:
            tng_set = batch["tng_set"]

        if len(input.shape) >= 4 and input.shape[1] >= 2:
            # Remove the exposure mask layer
            input = input[:, 0, :, :].reshape(
                input.shape[0], 1, input.shape[2], input.shape[3]
            )

        input = self.normalize.denormalize_lr_image(input)
        if type(gen) == dict:
            for key in gen:
                gen[key] = self.normalize.denormalize_hr_image(gen[key])
        else:
            gen = self.normalize.denormalize_hr_image(gen)
        label = self.normalize.denormalize_hr_image(label)

        if type(gen) == dict:
            # We have multiple generated images, log the metrics for all of them
            for key in gen:
                name_ext = f"_h_{key}"
                self.log_metrics(
                    input,
                    gen[key],
                    label,
                    pl_module,
                    stage=stage,
                    name_ext=name_ext,
                    tng_set=tng_set,
                    log_inputs=log_inputs,
                    log_extended=log_extended,
                )
        else:
            self.log_metrics(
                input,
                gen,
                label,
                pl_module,
                stage=stage,
                tng_set=tng_set,
                log_inputs=log_inputs,
                log_extended=log_extended,
            )

    # def on_train_batch_start(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     self.log_metrics_batch_end(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage="val")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        # Only log the inputs the first time to save computation time since they will not change
        if trainer.current_epoch == 0:
            log_inputs = True
        else:
            log_inputs = False
        self.log_metrics_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            stage="val",
            log_inputs=log_inputs,
            log_extended=False,
        )

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_metrics_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            stage="test",
            log_inputs=True,
            log_extended=True,
        )
