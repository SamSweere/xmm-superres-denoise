import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LitSRCNN(pl.LightningModule):
    def __init__(self, in_dims, out_dims, learning_rate):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        # Calculate the upscaling
        up_scale = out_dims[-1] / in_dims[-1]

        self.batch_size = in_dims[0]

        if up_scale % 2 != 0:
            raise ValueError(
                f"Upsaling is not a multple of two but {up_scale}, based on in_dims {in_dims} and out_dims{out_dims}"
            )

        # self.deconv = nn.ConvTranspose2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=int(up_scale), mode="nearest")
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=9, padding=2, padding_mode="replicate"
        )  # padding mode same as original Caffe code
        self.conv2 = nn.Conv2d(
            32, 16, kernel_size=9, padding=2, padding_mode="replicate"
        )
        self.conv3 = nn.Conv2d(
            16, 16, kernel_size=1, padding=4, padding_mode="replicate"
        )
        self.conv4 = nn.Conv2d(
            16, 1, kernel_size=5, padding=2, padding_mode="replicate"
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.upsample(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch["lr"], batch["hr"]
        pred = self(x)
        loss = F.mse_loss(pred, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["lr"], batch["hr"]
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("val/loss", loss, prog_bar=True)

        return pred, y

    def test_step(self, batch, batch_idx):
        x, y = batch["lr"], batch["hr"]
        pred = self(x)
        loss = F.mse_loss(pred, y)

        self.log("test/loss", loss, prog_bar=True)
        return pred, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams["learning_rate"]
        )
        return optimizer
