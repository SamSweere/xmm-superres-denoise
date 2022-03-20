# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan

import torch
from torch import nn, Tensor
from torch.autograd import Variable
import pytorch_lightning as pl

from models.modules.generator_rrdb import GeneratorRRDB_SR
from models.modules.vgg19_feature_extractor import FeatureExtractor
# from models.modules.generator_rrdb import GeneratorRRDB


class Discriminator(nn.Module):
    def __init__(self, input_shape, filter_start_size):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        filter_start_size = int(filter_start_size)
        filter_sizes = [1*filter_start_size, 2*filter_start_size, 4*filter_start_size, 8*filter_start_size]

        for i, out_filters in enumerate(filter_sizes):
        # for i, out_filters in enumerate([16, 32, 64, 128]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

class LitESRGAN(pl.LightningModule):

    def __init__(self, lr_shape, hr_shape, in_channels, out_channels, filters, residual_blocks, discriminator_filter_start, learning_rate, b1, b2, warmup_batches, lambda_adv, lambda_pixel, criterion):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        # Calculate the upscaling
        up_scale = hr_shape[-1] / lr_shape[-1]

        if up_scale % 2 != 0:
            raise ValueError(
                f"Upsaling is not a multple of two but {up_scale}, based on in_dims {lr_shape} and out_dims{hr_shape}")

        # The upscale is defined as 1 for 2x upsaceling
        up_scale = int(up_scale/2)

        # Make the model

        # Initialize generator and discriminator
        self.generator = GeneratorRRDB_SR(in_channels=in_channels, out_channels=out_channels, num_filters=filters, num_res_blocks=residual_blocks, num_upsample=up_scale)
        self.discriminator = Discriminator(input_shape=(out_channels, *hr_shape), filter_start_size=discriminator_filter_start)
        # TODO: enable again if you want this
        self.feature_extractor = FeatureExtractor()

        # # Set feature extractor to inference mode
        self.feature_extractor.eval()

        # Losses
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()
        # self.criterion_content = torch.nn.L1Loss()
        # self.criterion_pixel = torch.nn.L1Loss()
        # Loss
        self.criterion_pixel = criterion

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.generator(x)
        return x

    def _generator_loss(self, batch):
        # It is independent of forward
        imgs_lr, imgs_hr = batch['lr'], batch['hr']

        # Adversarial ground truths
        valid = Variable(torch.ones((imgs_lr.size(0), *self.discriminator.output_shape), device=self.device), requires_grad=False)
        # fake = Variable(torch.zeros((imgs_lr.size(0), *self.discriminator.output_shape), device=self.device), requires_grad=False)

        # Generate a high resolution image from low resolution input
        gen_hr = self(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

        # TODO: renable warmup
        # batches_done = self.current_epocTrueh*self.trainer.num_training_batches + batch_idx
        # if batches_done < self.hparams["warmup_batches"]:
        #     # Warm-up (pixel-wise loss only)
        #     loss_G = loss_pixel
        # else:
        # Extract validity predictions from discriminator
        pred_real = self.discriminator(imgs_hr).detach()
        pred_fake = self.discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        # Keep it in eval mode
        self.feature_extractor.eval()
        gen_features = self.feature_extractor(gen_hr)
        real_features = self.feature_extractor(imgs_hr).detach()
        loss_content = self.criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = self.hparams["lambda_adv"] * loss_GAN + self.hparams["lambda_pixel"] * loss_pixel  + loss_content
        return loss_G, loss_GAN, loss_pixel, gen_hr, imgs_hr, loss_content

    def _discriminator_loss(self, batch):
        # It is independent of forward
        imgs_lr, imgs_hr = batch['lr'], batch['hr']

        # Adversarial ground truths
        valid = Variable(torch.ones((imgs_lr.size(0), *self.discriminator.output_shape), device=self.device), requires_grad=False)
        fake = Variable(torch.zeros((imgs_lr.size(0), *self.discriminator.output_shape), device=self.device), requires_grad=False)

        # Generate a high resolution image from low resolution input
        gen_hr = self(imgs_lr)

        pred_real = self.discriminator(imgs_hr)
        pred_fake = self.discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        return loss_D, loss_real, loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop.

        # train generator
        if optimizer_idx == 0:
            # loss_content
            loss_G, loss_GAN, loss_pixel, gen_hr, imgs_hr = self._generator_loss(batch)
            self.log('train/generator_loss', loss_G, prog_bar=True)
            # self.log('train/loss_content', loss_content)
            self.log('train/loss_GAN', loss_GAN)
            self.log('train/loss_pixel', loss_pixel)
            return loss_G

        # train discriminator
        if optimizer_idx == 1:
            loss_D, loss_D_real, loss_D_fake = self._discriminator_loss(batch)
            self.log('train/discriminator_loss', loss_D, prog_bar=True)
            self.log('train/discriminator_loss_real', loss_D_real)
            self.log('train/discriminator_loss_fake', loss_D_fake)
            return loss_D

    def validation_step(self, batch, batch_idx):
        # loss_content
        loss_G, loss_GAN, loss_pixel, gen_hr, imgs_hr, loss_content = self._generator_loss(batch)
        loss_D, loss_D_real, loss_D_fake = self._discriminator_loss(batch)

        self.log('val/generator_loss', loss_G, prog_bar=True)
        self.log('val/loss', loss_G, prog_bar=True)
        self.log('val/loss_content', loss_content)
        self.log('val/loss_GAN', loss_GAN)
        self.log('val/loss_epoch', loss_GAN)
        self.log('val/loss_pixel', loss_pixel)
        self.log('val/discriminator_loss', loss_D, prog_bar=True)
        self.log('val/discriminator_loss_real', loss_D_real)
        self.log('val/discriminator_loss_fake', loss_D_fake)

        # Needed for extra loss calculation
        return gen_hr

    def test_step(self, batch, batch_idx):
        # loss_content
        loss_G, loss_GAN, loss_pixel, gen_hr, imgs_hr, loss_content = self._generator_loss(batch)
        loss_D, loss_D_real, loss_D_fake = self._discriminator_loss(batch)

        self.log('test/generator_loss', loss_G, prog_bar=True)
        self.log('test/loss_content', loss_content)
        self.log('test/loss_GAN', loss_GAN)
        self.log('test/loss_pixel', loss_pixel)
        self.log('test/discriminator_loss', loss_D, prog_bar=True)
        self.log('test/discriminator_loss_real', loss_D_real)
        self.log('test/discriminator_loss_fake', loss_D_fake)

        # Needed for extra loss calculation
        return gen_hr

    def configure_optimizers(self):
        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams["learning_rate"],
                                       betas=(self.hparams["b1"], self.hparams["b2"]))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams["learning_rate"],
                                       betas=(self.hparams["b1"], self.hparams["b2"]))

        return [optimizer_G, optimizer_D]