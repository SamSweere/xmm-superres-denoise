# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan
from typing import Tuple, Dict

import pytorch_lightning as pl
import torch

from utils.metriclogger import MetricsCalculator


class Model(pl.LightningModule):
    def __init__(
            self,
            config: dict,
            lr_shape: Tuple[int, int],
            hr_shape: Tuple[int, int],
            loss,
            metrics_calculator: MetricsCalculator
    ):
        super(Model, self).__init__()

        # Holds intermediate outputs
        self.test_val_step_output: Dict[str, Dict[str, list]] = {}

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

        self.metrics_calculator = metrics_calculator

    def forward(self, x) -> torch.Tensor:
        return torch.clamp(self.model(x), min=0.0, max=1.0)

    def training_step(self, batch, batch_idx):
        gen_hr = self(batch["lr"])
        loss = self.loss(gen_hr, batch["hr"])
        self.log("train/loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        self._log_metrics(batch, "val")

    def test_step(self, batch, batch_idx):
        self._log_metrics(batch, "test")

    def _log_metrics(self, batch, stage):
        lr = batch["lr"]
        gen_hr = self(lr)
        true_hr = batch["hr"]

        loss = self.loss(gen_hr, true_hr)
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        log_inputs = (stage == "val" and self.current_epoch == 0) or stage == "test"
        log_extended = stage == "test"

        metrics = self.metrics_calculator(lr=lr, pred=gen_hr, true=true_hr, prefix=f"{stage}/",
                                          log_inputs=log_inputs, log_extended=log_extended)
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

    def configure_optimizers(self):
        # Optimizers
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas
        )

        return optimizer
