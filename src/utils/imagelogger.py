# Written by: Sam Sweere
from pathlib import Path
from typing import Dict, Union, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from matplotlib import cm
from torchmetrics.functional import structural_similarity_index_measure as ssim

from datasets import XmmDisplayDataModule
from utils.filehandling import write_xmm_file_to_fits

_img_val_dict = {
    "input": {
        "cm": "plasma",
        "norm": False,
        "key": "{0}/input/{1}"
    },
    "generated": {
        "cm": "plasma",
        "norm": False,
        "key": "{0}/generated/{1}"
    },
    "label": {
        "cm": "plasma",
        "norm": False,
        "key": "{0}/label/{1}"
    },
    "difference": {
        "cm": "seismic",
        "norm": True,
        "key": "{0}/difference/{1}"
    },
    "ssim": {
        "cm": "seismic",
        "norm": True,
        "key": "{0}/ssim map/{1}"
    }
}


def _apply_cm(colormap, image, scale=None, normalize=False):
    if scale:
        image = scale(image)

    if normalize:
        # Normalize the image to 0.0 and 1.0
        image = image / torch.max(image)

    return cm.get_cmap(colormap)(image)


def _convert_images_to_cm(colormap, images, scale=None, normalize=False):
    images_cm = [_apply_cm(colormap, image.squeeze().cpu(), scale=scale, normalize=normalize) for image in images]
    return images_cm


def to_wandb(logger, img_type, scale_normalizer, images, stage):
    key = _img_val_dict[img_type]["key"]
    colormap = _img_val_dict[img_type]["cm"]
    norm = _img_val_dict[img_type]["norm"]
    key_format = key.format(stage, scale_normalizer)

    images = _convert_images_to_cm(colormap, images, normalize=norm)

    logger.log_image(key=key_format, images=images)


def _write_fits(
        parent_dir: Path,
        imgs: torch.Tensor,
        exps: torch.Tensor,
        imgs_filenames: List[str],
        res_mult: int,
        comment: str,
        out_file_name: str,
        in_header=None
) -> None:
    for img, exp, img_filename in zip(imgs, exps, imgs_filenames):
        out_path = parent_dir / f"{img_filename[:60]}"
        out_path.mkdir(parents=True, exist_ok=True)
        img = img * exp
        write_xmm_file_to_fits(
            img=img[0].detach().cpu().numpy(),
            output_dir=out_path,
            source_file_name=img_filename,
            res_mult=res_mult,
            exposure=exp.detach().item(),
            comment=comment,
            out_file_name=out_file_name,
            in_header=in_header
        )


def _save_fits_image(
        parent_dir: Path,
        lr_imgs: torch.Tensor,
        preds: torch.Tensor,
        labels: Optional[torch.Tensor],
        lr_exps: torch.Tensor,
        hr_exps: torch.Tensor,
        filenames: List[str],
        lr_in_header=None,
        hr_in_header=None
):
    lr_exps = lr_exps * 1000
    hr_exps = hr_exps * 1000

    if lr_in_header is not None:
        for key, value in lr_in_header.items():
            if isinstance(value, torch.Tensor):
                lr_in_header[key] = value.detach().cpu().numpy()[0]
            else:
                lr_in_header[key] = value[0]

    _write_fits(
        parent_dir=parent_dir,
        imgs=lr_imgs,
        exps=lr_exps,
        imgs_filenames=filenames,
        res_mult=1,
        comment="Transformed Input Image (input to model)",
        out_file_name="input",
        in_header=lr_in_header
    )

    out_res_mult = int(preds.shape[-1] / lr_imgs.shape[-1])

    _write_fits(
        parent_dir=parent_dir,
        imgs=preds,
        exps=hr_exps,
        imgs_filenames=filenames,
        res_mult=out_res_mult,
        comment="Model Prediction",
        out_file_name="prediction",
        in_header=lr_in_header
    )

    if labels is not None:
        _write_fits(
            parent_dir=parent_dir,
            imgs=labels,
            exps=hr_exps,
            imgs_filenames=filenames,
            res_mult=out_res_mult,
            comment="Reference Image",
            out_file_name="label",
            in_header=hr_in_header
        )


class ImageLogger(pl.Callback):
    def __init__(
            self,
            datamodule: XmmDisplayDataModule,
            log_every_n_epochs,
            normalize,
            scaling_normalizers,
            data_range,
    ):
        super(ImageLogger, self).__init__()

        self.datamodule = datamodule
        self.normalize = normalize
        self.scaling_normalizers = scaling_normalizers
        self.data_range = data_range

        self.log_every_n_epochs = log_every_n_epochs

    @rank_zero_only
    def _get_samples(
            self,
            pl_module: "pl.LightningModule",
            dataloaders: EVAL_DATALOADERS
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        outputs = {
            "lr": [],
            "pred": [],
            "true": [],
            "lr_exp": [],
            "hr_exp": [],
            "filenames": []
        }
        for dataloader in dataloaders:
            for batch in dataloader:
                lr = batch["lr"].to(pl_module.device)
                lr_exp = batch["lr_exp"]

                pred = pl_module(lr)

                if "hr" in batch:
                    true = batch["hr"].to(pl_module.device)
                    hr_exp = batch["hr_exp"]
                else:
                    true = pred.detach().clone()
                    hr_exp = lr_exp

                lr = torch.flip(lr, [-2]).detach()
                pred = torch.flip(pred, [-2]).detach()
                true = torch.flip(true, [-2]).detach()

                outputs["lr"].append(lr)
                outputs["pred"].append(pred)
                outputs["true"].append(true)
                outputs["lr_exp"].append(lr_exp)
                outputs["hr_exp"].append(hr_exp)
                outputs["filenames"].extend(batch["lr_img_file_name"])

        outputs["lr"] = torch.cat(outputs["lr"], dim=0)
        outputs["pred"] = torch.cat(outputs["pred"], dim=0)
        outputs["true"] = torch.cat(outputs["true"], dim=0)
        outputs["lr_exp"] = torch.cat(outputs["lr_exp"], dim=0).to(pl_module.device)
        outputs["hr_exp"] = torch.cat(outputs["hr_exp"], dim=0).to(pl_module.device)

        return outputs

    @rank_zero_only
    def _log_images(
            self,
            logger: WandbLogger,
            pl_module: "pl.LightningModule",
            dataloaders: EVAL_DATALOADERS,
            stage: str,
            first_epoch: bool = False
    ):
        outputs: Dict[str, Union[torch.Tensor, List[str]]] = self._get_samples(pl_module, dataloaders)

        lr = outputs.pop("lr")
        pred = outputs.pop("pred")
        true = outputs.pop("true")

        lr_exp = outputs.pop("lr_exp")
        hr_exp = outputs.pop("hr_exp")

        real_indices = lr_exp == hr_exp
        sim_indices = lr_exp != hr_exp

        filenames = np.asarray(outputs.pop("filenames"))

        for display_type, indices in zip(["real", "sim"], [real_indices, sim_indices]):
            lr_sub = lr[indices]
            pred_sub = pred[indices]
            true_sub = true[indices]
            lr_exp_sub = lr_exp[indices]
            hr_exp_sub = hr_exp[indices]
            filenames_sub = filenames[indices.cpu().numpy()].tolist()

            lr_sub = self.normalize.denormalize_lr_image(lr_sub)
            pred_sub = self.normalize.denormalize_hr_image(pred_sub)
            true_sub = self.normalize.denormalize_hr_image(true_sub)

            _save_fits_image(
                parent_dir=Path(logger.experiment.dir) / "fits_out" / f"{stage}" / f"{display_type}",
                lr_imgs=lr_sub,
                preds=pred_sub,
                labels=true_sub if display_type == "sim" else None,
                lr_exps=lr_exp_sub,
                hr_exps=hr_exp_sub,
                filenames=filenames_sub
            )

            if first_epoch:
                for scale_normalizer in self.scaling_normalizers:
                    stretch_name = scale_normalizer.stretch_mode
                    scaled_lr = scale_normalizer.normalize_lr_image(lr_sub)
                    to_wandb(logger, "input", stretch_name, scaled_lr, f"{display_type}")

                    if display_type == "sim":
                        scaled_labels = scale_normalizer.normalize_hr_image(true_sub)
                        to_wandb(logger, "label", stretch_name, scaled_labels, f"{display_type}")
            else:
                for scale_normalizer in self.scaling_normalizers:
                    stretch_name = scale_normalizer.stretch_mode

                    scaled_pred = scale_normalizer.normalize_hr_image(pred_sub)
                    to_wandb(logger, "generated", stretch_name, scaled_pred, f"{stage}/{display_type}")

                    if display_type == "sim":
                        scaled_true = scale_normalizer.normalize_hr_image(true_sub)
                        ssim_image = ssim(preds=scaled_pred, target=scaled_true, sigma=2.5, kernel_size=13,
                                          data_range=self.data_range, k1=0.01, k2=0.05, return_full_image=True)[1]
                        to_wandb(logger, "ssim", stretch_name, ssim_image, f"{stage}/{display_type}")
                        difference_image = ((scaled_pred - scaled_true) + 1.0) / 2.0
                        to_wandb(logger, "difference", stretch_name, difference_image, f"{stage}/{display_type}")

    @rank_zero_only
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking:
            if trainer.current_epoch == 0:
                self._log_images(trainer.logger, pl_module, self.datamodule.val_dataloader(), "val", True)
            if ((trainer.current_epoch + 1) % self.log_every_n_epochs) == 0:
                self._log_images(trainer.logger, pl_module, self.datamodule.val_dataloader(), "val")

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log_images(self.datamodule.test_dataloader(), pl_module, "test")

    def on_predict_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch: Dict[str, Union[torch.Tensor, List[str]]],
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        lr = batch["lr"]
        preds = batch["preds"]
        hr = batch.get("hr", None)
        filenames = batch["lr_img_file_name"]
        lr_in_header = batch.get("lr_header", None)
        hr_in_header = batch.get("hr_header", None)

        lr_exp = batch["lr_exp"]
        hr_exp = batch["hr_exp"] if "hr" in batch else lr_exp

        lr = self.normalize.denormalize_lr_image(lr)
        preds = self.normalize.denormalize_hr_image(preds)
        hr = self.normalize.denormalize_hr_image(hr) if "hr" in batch else None

        display_type = "sim" if "hr" in batch else "real"

        _save_fits_image(
            parent_dir=Path(trainer.logger.experiment.dir) / "fits_out" / "predict" / f"{display_type}",
            lr_imgs=lr,
            preds=preds,
            labels=hr,
            lr_exps=lr_exp,
            hr_exps=hr_exp,
            filenames=filenames,
            lr_in_header=lr_in_header,
            hr_in_header=hr_in_header
        )

        imgs = [_apply_cm(_img_val_dict["input"]["cm"], lr.cpu(), normalize=_img_val_dict["input"]["norm"]),
                _apply_cm(_img_val_dict["generated"]["cm"], preds.cpu(), normalize=_img_val_dict["generated"]["norm"])]
        captions = ["lr", "gen"]
        if "hr" in batch:
            imgs.append(_apply_cm(_img_val_dict["label"]["cm"], hr.cpu(), normalize=_img_val_dict["label"]["norm"]))
            captions.append("label")
            difference_image = ((preds - hr) + 1.0) / 2.0
            imgs.append(_apply_cm(_img_val_dict["difference"]["cm"], difference_image.cpu(),
                                  normalize=_img_val_dict["difference"]["norm"]))
            captions.append("diff")

        trainer.logger.log_image(key=f"pred/{filenames[0][:60]}", images=imgs, caption=captions)
