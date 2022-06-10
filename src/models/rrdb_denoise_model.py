# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan
import math

import torch
import pytorch_lightning as pl
from torch import nn
#
# from models.modules.rrdb_blocks import ResidualInResidualDenseBlock
#
#
# class GeneratorRRDB(nn.Module):
#     def __init__(self, in_channels, out_channels, filters=64, num_res_blocks=16):
#         super(GeneratorRRDB, self).__init__()
#         self.in_channels = in_channels
#
#         # First layer
#         self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)
#         # Residual blocks
#         self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
#         # Second conv layer post residual blocks
#         self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
#         # # Upsampling layers
#         # upsample_layers = []
#         # for _ in range(num_upsample):
#         #     upsample_layers += [
#         #         nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, stride=1, padding=1),
#         #         nn.LeakyReLU(),
#         #         nn.PixelShuffle(upscale_factor=2),
#         #     ]
#         # self.upsampling = nn.Sequential(*upsample_layers)
#         # Final output block
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding=1),
#         )
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out = self.res_blocks(out1)
#         out2 = self.conv2(out)
#         out = torch.add(out1, out2)
#         # out = self.upsampling(out)
#         out = self.conv3(out)
#         if self.in_channels > 1:
#             x = x[:, 0, :, :].reshape(x.shape[0], 1, x.shape[2], x.shape[3])
#         out = torch.add(x, out)
#         out = torch.clamp(out, min=0.0, max=1.0)
#         return out
from models.modules.generator_rrdb import GeneratorRRDB_DN


class LitRRDBDenoise(pl.LightningModule):
    # This is only the generator from the esr gan with a single loss

    def __init__(self, lr_shape, hr_shape, in_channels, out_channels, filters, residual_blocks, learning_rate, b1, b2, criterion):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        # Make the model

        # Initialize generator and discriminator
        self.generator = GeneratorRRDB_DN(in_channels=in_channels, out_channels=out_channels, num_filters=filters, num_res_blocks=residual_blocks)

        # Loss
        self.criterion = criterion

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.generator(x)
        return x

    def _generator_loss(self, batch):
        # It is independent of forward
        imgs_lr, imgs_hr = batch['lr'], batch['hr']

        # Generate a high resolution image from low resolution input
        gen_hr = self(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss = self.criterion(gen_hr, imgs_hr)

        return loss, gen_hr, imgs_hr

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.

        # train generator

        loss, gen_hr, imgs_hr = self._generator_loss(batch)
        self.log('train/loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, gen_hr, imgs_hr = self._generator_loss(batch)

        self.log('val/loss', loss, prog_bar=True)

        # Needed for extra loss calculation
        return gen_hr

    def test_step(self, batch, batch_idx):
        loss, gen_hr, imgs_hr = self._generator_loss(batch)

        self.log('test/loss', loss, prog_bar=True)

        # Needed for extra loss calculation
        return gen_hr

    def configure_optimizers(self):
        # Optimizers
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.hparams["learning_rate"],
                                       betas=(self.hparams["b1"], self.hparams["b2"]))

        return optimizer