from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_shape: Tuple, latent_dim: int):
        super().__init__()
        self.input_shape = input_shape

        self.fc1 = nn.Linear(int(np.prod(input_shape)), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_mean = nn.Linear(256, latent_dim)
        self.fc3_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        logvar = self.fc3_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, output_shape: Tuple, latent_dim: int):
        super().__init__()
        self.output_shape = output_shape

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, int(np.prod(output_shape)))

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))
        return z


class VAE(pl.LightningModule):
    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        latent_dim: int = 128,
        lr: float = 0.001,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(
            input_shape=img_shape,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            output_shape=img_shape,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return mean, logvar, x_hat

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def loss_function(self, x_hat, x, mean, logvar):
        bce_loss = F.binary_cross_entropy(x_hat, x.view(-1, np.prod(x.shape[1:])), reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return bce_loss + kl_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # forward pass
        mean, logvar, x_hat = self(x)

        # compute loss and log
        loss = self.loss_function(x_hat, x, mean, logvar)
        self.log("loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
