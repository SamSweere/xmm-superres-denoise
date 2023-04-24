# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan
from typing import Tuple

import pytorch_lightning as pl
import torch


class Model(pl.LightningModule):
    def __init__(
            self,
            config: dict,
            lr_shape: Tuple[int, int],
            hr_shape: Tuple[int, int],
            loss
    ):
        super(Model, self).__init__()

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
        gen_hr = self(batch["lr"])
        loss = self.loss(gen_hr, batch["hr"])
        self.log("train/loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def _on_step(self, batch, stage):
        lr = batch["lr"]
        preds = self(lr)
        target = batch.get("hr", preds)

        loss = self.loss(preds, target)
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        return {
            "lr": lr.detach(),
            "preds": preds.detach(),
            "target": target.detach()
        }

    def validation_step(self, batch, batch_idx):
        return self._on_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._on_step(batch, "test")

    def configure_optimizers(self):
        # Optimizers
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas
        )

        return optimizer
