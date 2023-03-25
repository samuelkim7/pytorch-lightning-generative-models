# main.py
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from datasets.celeba_datamodule import CelebADataModule
from datasets.mnist_datamodule import MNISTDataModule
from models.gan import GAN


def train(cfg: DictConfig) -> None:
    root_dir = Path(hydra.utils.get_original_cwd())
    data_dir = root_dir / cfg.data.data_dir
    callbacks = [TQDMProgressBar(refresh_rate=20)]

    if cfg.data.dataset == "MNIST":
        datamodule = MNISTDataModule(
            data_dir=data_dir,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.model.batch_size,
        )
    elif cfg.data.dataset == "CelebA":
        datamodule = CelebADataModule(
            data_dir=data_dir,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.model.batch_size,
        )
    model = GAN(
        *datamodule.dims,
        g_lr=cfg.model.g_lr,
        d_lr=cfg.model.d_lr,
        batch_size=cfg.model.batch_size,
        b1=cfg.model.b1,
        b2=cfg.model.b2,
    )
    trainer = pl.Trainer(
        # accelerator="mps",
        # gpus=1 if torch.backends.mps.is_available() else None,
        max_epochs=cfg.trainer.max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../configs", config_name="gan.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    pl.seed_everything(123)
    main()
