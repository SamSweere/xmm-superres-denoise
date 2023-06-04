# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan
import math

import pytorch_lightning as pl
import torch

from models.modules.generator_rrdb import GeneratorRRDB_DN


class LitRRDBDenoise(pl.LightningModule):
    # This is only the generator from the esr gan with a single loss

    def __init__(
        self,
        lr_shape,
        hr_shape,
        in_channels,
        out_channels,
        filters,
        residual_blocks,
        learning_rate,
        b1,
        b2,
        criterion,
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        # Make the model

        # Initialize generator and discriminator
        self.generator = GeneratorRRDB_DN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=filters,
            num_res_blocks=residual_blocks,
        )

        # Loss
        self.criterion = criterion

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.generator(x)
        return x

    def _generator_loss(self, batch):
        # It is independent of forward
        imgs_lr, imgs_hr = batch["lr"], batch["hr"]

        # Generate a high resolution image from low resolution input
        gen_hr = self(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss = self.criterion(gen_hr, imgs_hr)

        return loss, gen_hr, imgs_hr

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.

        # train generator

        loss, gen_hr, imgs_hr = self._generator_loss(batch)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, gen_hr, imgs_hr = self._generator_loss(batch)

        self.log("val/loss", loss, prog_bar=True)

        # Needed for extra loss calculation
        return gen_hr

    def test_step(self, batch, batch_idx):
        loss, gen_hr, imgs_hr = self._generator_loss(batch)

        self.log("test/loss", loss, prog_bar=True)

        # Needed for extra loss calculation
        return gen_hr

    def configure_optimizers(self):
        # Optimizers
        optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams["learning_rate"],
            betas=(self.hparams["b1"], self.hparams["b2"]),
        )

        return optimizer
