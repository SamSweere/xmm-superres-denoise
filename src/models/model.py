# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan
from typing import Tuple, Optional

import lightning.pytorch as pl
import torch
from torch import Tensor
from torchmetrics import Metric

from metrics import MetricsCalculator


class Model(pl.LightningModule):
    def __init__(
            self,
            config: dict,
            lr_shape: Tuple[int, int],
            hr_shape: Tuple[int, int],
            loss: Metric,
            metrics: MetricsCalculator
    ):
        super(Model, self).__init__()

        self.mc = metrics
        # Has to be set for the metrics to work
        self.metrics = metrics.metrics
        self.input_metrics = metrics.input_metrics
        self.input_extended_metrics = metrics.input_extended_metrics
        self.extended_metrics = metrics.extended_metrics

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
            from models import GeneratorRRDB_SR
            up_scale = hr_shape[0] / lr_shape[0]
            if up_scale % 2 != 0:
                raise ValueError(f"Upscaling is not a multiple of two but {up_scale}, "
                                 f"based on in_dims {lr_shape} and out_dims {hr_shape}")

            up_scale = int(up_scale / 2)
            # Initialize generator and discriminator
            self.model = GeneratorRRDB_SR(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                num_filters=config["filters"],
                num_res_blocks=config["residual_blocks"],
                num_upsample=up_scale,
                memory_efficient=self.memory_efficient
            )
        elif self.model_name == "rrdb_denoise":
            from models import GeneratorRRDB_DN
            self.model = GeneratorRRDB_DN(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                num_filters=config["filters"],
                num_res_blocks=config["residual_blocks"],
                memory_efficient=self.memory_efficient
            )
        else:
            raise ValueError(f"Base model name {self.model_name} is not a valid model name!")

    def forward(self, x) -> torch.Tensor:
        return torch.clamp(self.model(x), min=0.0, max=1.0)

    def training_step(self, batch, batch_idx):
        loss = self._on_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._on_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._on_step(batch, "test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch["preds"] = self(batch["lr"])

    def _on_step(self, batch, stage) -> Optional[Tensor]:
        lr = batch["lr"]
        preds = self(lr)
        target = batch.get("hr", preds)

        loss = self.loss(preds=preds, target=target)

        if stage == "train":
            self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=self.batch_size,
                     on_step=True, on_epoch=False)
            return loss
        else:
            self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=self.batch_size,
                     on_step=False, on_epoch=True, sync_dist=True)

            log_inputs = self.current_epoch == 0 or stage == "test"
            log_extended = stage == "test"
            lr = lr if log_inputs else None

            self.mc.update(preds=preds, lr=lr, target=target, stage=stage)

            self.log_dict(self.metrics, batch_size=self.batch_size, on_step=False, on_epoch=True, sync_dist=True)

            if log_inputs:
                self.log_dict(self.input_metrics, batch_size=self.batch_size, on_step=False, on_epoch=True,
                              sync_dist=True)

            if log_extended:
                self.log_dict(self.extended_metrics, batch_size=self.batch_size, on_step=False, on_epoch=True,
                              sync_dist=True)

                if log_inputs:
                    self.log_dict(self.input_extended_metrics, batch_size=self.batch_size, on_step=False, on_epoch=True,
                                  sync_dist=True)

    def configure_optimizers(self):
        # Optimizers
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas
        )

        return optimizer
