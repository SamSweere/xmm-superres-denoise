# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from config.config import BaseModels, ModelCfg
from metrics import XMMMetricCollection
from torch import Tensor
from torchmetrics import Metric
from transforms import ImageUpsample


class Model(pl.LightningModule):
    def __init__(
        self,
        config: ModelCfg,
        lr_shape: Tuple[int, int],
        hr_shape: Tuple[int, int],
        loss: Optional[Metric],
        metrics: Optional[XMMMetricCollection],
        extended_metrics: Optional[XMMMetricCollection],
        in_metrics: Optional[XMMMetricCollection],
        in_extended_metrics: Optional[XMMMetricCollection],
    ):
        super().__init__()
        self.config = config

        self.metrics = metrics
        self.ext_metrics = extended_metrics
        self.in_metrics = in_metrics
        self.in_ext_metrics = in_extended_metrics

        # Model and model parameters
        self.loss = loss
        self.model: torch.nn.Module
        if self.config.name is BaseModels.ESR_GEN:
            from models import GeneratorRRDB_SR

            up_scale = hr_shape[0] / lr_shape[0]
            if up_scale % 2 != 0:
                raise ValueError(
                    f"Upscaling is not a multiple of two but {up_scale}, "
                    f"based on in_dims {lr_shape} and out_dims {hr_shape}"
                )

            up_scale = int(up_scale / 2)
            # Initialize generator and discriminator
            self.model = GeneratorRRDB_SR(
                in_channels=self.config.model.in_channels,
                out_channels=self.config.model.out_channels,
                num_filters=self.config.model.filters,
                num_res_blocks=self.config.model.residual_blocks,
                num_upsample=up_scale,
                memory_efficient=self.config.memory_efficient,
            )
        elif self.config.name is BaseModels.RRDB_DENOISE:
            from models import GeneratorRRDB_DN

            self.model = GeneratorRRDB_DN(
                in_channels=self.config.model.in_channels,
                out_channels=self.config.model.out_channels,
                num_filters=self.config.model.filters,
                num_res_blocks=self.config.model.residual_blocks,
                memory_efficient=self.config.memory_efficient,
            )
        elif self.config.name is BaseModels.SWINIR:
            from models import SwinIR

            self.model = SwinIR(
                img_size=self.config.model.img_size,
                window_size=self.config.model.window_size,
                embed_dim=self.config.model.embed_dim,
                num_heads=self.config.model.num_heads,
                depths=self.config.model.depths,
                upsampler=self.config.model.upsampler,
                in_chans=self.config.model.in_channels,
                use_checkpoint=self.config.memory_efficient,
            )
        else:
            raise ValueError(
                f"Base model name {self.config.name} is not a valid model name!"
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
        lr, hr = batch
        preds = self(lr)
        target = hr if hr is not None else preds

        if stage == "train":
            loss = self.loss(preds=preds, target=target)
            self.log(
                f"{stage}/loss",
                loss,
                batch_size=self.config.batch_size,
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
                batch_size=self.config.batch_size,
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
                    batch_size=self.config.batch_size,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def configure_optimizers(self):
        # Optimizers
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            betas=self.config.optimizer.betas,
        )

        return optimizer
