# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchmetrics import Metric

from xmm_superres_denoise.metrics import XMMMetricCollection
from xmm_superres_denoise.transforms import ImageUpsample


class Model(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        lr_shape: Tuple[int, int],
        hr_shape: Tuple[int, int],
        loss: Optional[Metric],
        metrics: Optional[XMMMetricCollection],
        extended_metrics: Optional[XMMMetricCollection],
        in_metrics: Optional[XMMMetricCollection],
        in_extended_metrics: Optional[XMMMetricCollection],
    ):
        super(Model, self).__init__()

        self.metrics = metrics
        self.ext_metrics = extended_metrics
        self.in_metrics = in_metrics
        self.in_ext_metrics = in_extended_metrics

        # Optimizer parameters
        self.learning_rate = config["learning_rate"]
        self.betas = (config["b1"], config["b2"])

        # Model and model parameters
        self.memory_efficient = config["memory_efficient"]
        self.model_name = config["name"]
        self.loss = loss
        self.batch_size = config["batch_size"]
        self.model: torch.nn.Module
        if self.model_name == "esr_gen":
            from xmm_superres_denoise.models import GeneratorRRDB_SR

            up_scale = hr_shape[0] / lr_shape[0]
            if up_scale % 2 != 0:
                raise ValueError(
                    f"Upscaling is not a multiple of two but {up_scale}, "
                    f"based on in_dims {lr_shape} and out_dims {hr_shape}"
                )

            up_scale = int(up_scale / 2)
            # Initialize generator and discriminator
            self.model = GeneratorRRDB_SR(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                num_filters=config["filters"],
                num_res_blocks=config["residual_blocks"],
                num_upsample=up_scale,
                memory_efficient=self.memory_efficient,
            )
        elif self.model_name == "rrdb_denoise":
            from xmm_superres_denoise.models import GeneratorRRDB_DN

            self.model = GeneratorRRDB_DN(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                num_filters=config["filters"],
                num_res_blocks=config["residual_blocks"],
                memory_efficient=self.memory_efficient,
            )
        elif self.model_name == "swinir":
            from xmm_superres_denoise.models import SwinIR

            self.model = SwinIR(
                img_size=config["img_size"],
                window_size=config["window_size"],
                embed_dim=config["embed_dim"],
                num_heads=config["num_heads"],
                depths=config["depths"],
                upsampler=config["upsampler"],
                in_chans=config["in_channels"],
                use_checkpoint=self.memory_efficient,
            )
        else:
            raise ValueError(
                f"Base model name {self.model_name} is not a valid model name!"
            )

    def forward(self, x) -> torch.Tensor:
        return torch.clamp(self.model(x), min=0.0, max=1.0)

    def training_step(self, batch, batch_idx):
        return self._on_step(batch, "train")

    def on_validation_start(self) -> None:
        self._on_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        self._on_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end("val")

    def test_step(self, batch, batch_idx):
        self._on_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end("test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch["preds"] = self(batch["lr"])

    def _on_step(self, batch, stage) -> Optional[Tensor]:
        lr = batch["lr"]
        preds = self(lr)
        target = batch.get("hr", preds)

        if stage == "train":
            loss = self.loss(preds=preds, target=target)
            self.log(
                f"{stage}/loss",
                loss,
                batch_size=self.batch_size,
                on_step=True,
                on_epoch=False,
            )
            return loss
        else:
            self.loss.update(preds=preds, target=target)

            if self.in_metrics is not None or self.ext_metrics is not None:
                scale_factor = target.shape[2] / lr.shape[2]
                if scale_factor != 1.0:
                    lr = ImageUpsample(scale_factor=scale_factor)(lr)

            if self.metrics is not None:
                self.metrics.update(preds=preds, target=target)

            if self.in_metrics is not None:
                self.in_metrics.update(preds=lr, target=target)

            if self.ext_metrics is not None:
                self.ext_metrics.update(preds=preds, target=target)

            if self.in_ext_metrics is not None:
                self.in_ext_metrics.update(preds=lr, target=target)

    def _on_epoch_end(self, stage):
        if stage == "train":
            self.loss.reset()
        else:
            loss = self.loss.compute()
            self.log(
                f"{stage}/loss",
                loss,
                batch_size=self.batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.loss.reset()

            to_log: List[dict] = []

            if self.metrics is not None:
                to_log.append(self.metrics.compute())
                self.metrics.reset()

            if self.ext_metrics is not None:
                to_log.append(self.ext_metrics.compute())
                self.ext_metrics.reset()

            if self.in_metrics is not None:
                to_log.append(self.in_metrics.compute())
                self.in_metrics.reset()
                if not self.trainer.sanity_checking:
                    self.in_metrics = None

            if self.in_ext_metrics is not None:
                to_log.append(self.in_ext_metrics.compute())
                self.in_ext_metrics.reset()
                if not self.trainer.sanity_checking:
                    self.in_ext_metrics = None

            for log in to_log:
                self.log_dict(
                    log,
                    batch_size=self.batch_size,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def configure_optimizers(self):
        # Optimizers
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, betas=self.betas
        )

        return optimizer
