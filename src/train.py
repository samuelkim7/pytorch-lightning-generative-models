from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig


def train(cfg: DictConfig) -> None:
    root_dir = Path(hydra.utils.get_original_cwd())
    data_dir = root_dir / cfg.data_dir
    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))

    datamodule: pl.LightningDataModule = instantiate(
        data_dir=data_dir,
        config=cfg.datamodule,
    )

    model: pl.LightningModule = instantiate(
        img_shape=datamodule.dims,
        config=cfg.model,
    )
    trainer: pl.Trainer = instantiate(
        config=cfg.trainer,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../configs", config_name="vae.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    pl.seed_everything(123)
    main()
