import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from datasets.mnist_datamodule import MNISTDataModule
from models.gan import GAN


def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    num_workers = os.cpu_count() - 2
    num_epochs = 100
    callbacks = [TQDMProgressBar(refresh_rate=20)]

    datamodule = MNISTDataModule(
        data_dir=data_dir,
        num_workers=num_workers,
    )
    model = GAN(
        *datamodule.dims,
        lr=0.0002,
        batch_size=256,
        b1=0.7,
        b2=0.999,
    )
    trainer = pl.Trainer(
        accelerator="mps",
        devices=1 if torch.backends.mps.is_available() else None,
        max_epochs=num_epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    pl.seed_everything(1234)
    main()
