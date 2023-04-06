import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import Callback


class ImageLoggerVAE(Callback):
    """Sample and log images for every N epochs"""

    def __init__(self, num_samples: int, num_epochs: int):
        super().__init__()
        self.num_samples = num_samples
        self.num_epochs = num_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.num_epochs == 0:
            devices = "gpu" if torch.cuda.is_available() else "cpu"
            z = torch.randn(self.num_samples, pl_module.hparams.latent_dim, device=devices)
            sample_imgs = pl_module.decoder(z)
            sample_imgs = sample_imgs.view(self.num_samples, *pl_module.hparams.img_shape)
            grid = torchvision.utils.make_grid(sample_imgs)
            pl_module.logger.experiment.add_image("generated_images", grid, pl_module.current_epoch)
        return super().on_train_epoch_end(trainer, pl_module)
