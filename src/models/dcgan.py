from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_shape: Tuple,
    ):
        super().__init__()
        self.img_shape = img_shape

        self.init_size = img_shape[1] // 4  # Initial size before upsampling
        self.fc = nn.Linear(latent_dim, 128 * self.init_size**2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        output = self.fc(z)
        output = output.view(output.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(output)

        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        def block(in_filters, out_filters, batch_norm=True):
            layers = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_filters))
            return layers

        self.conv_blocks = nn.Sequential(
            *block(img_shape[0], 16, batch_norm=False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )

        output_size = img_shape[1] // 2**4
        self.fc = nn.Sequential(
            nn.Linear(128 * output_size**2, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        output = self.conv_blocks(img)
        output = output.view(output.size(0), -1)
        validity = self.fc(output)

        return validity


class DCGAN(pl.LightningModule):
    def __init__(
        self,
        channels: int,
        width: int,
        height: int,
        latent_dim: int = 100,
        g_lr: float = 0.001,
        d_lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        data_shape = (channels, width, height)
        self.generator = Generator(
            latent_dim=self.hparams.latent_dim,
            img_shape=data_shape,
        )
        self.discriminator = Discriminator(
            img_shape=data_shape,
        )

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_arrray = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            g_loss = self.adversarial_loss(
                self.discriminator(self(z)),
                valid,
            )
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(
                self.discriminator(imgs),
                valid,
            )

            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()),
                fake,
            )

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.g_lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.d_lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        return [opt_g, opt_d], []

    def on_train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.fc[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
