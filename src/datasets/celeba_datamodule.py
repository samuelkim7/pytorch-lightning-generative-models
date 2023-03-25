import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CelebA


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        train_val_ratio: float = 0.9,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.num_workers = num_workers
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        self.transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ],
        )

        self.dims = (3, 64, 64)
        self.num_classes = None

    def prepare_data(self):
        CelebA(self.data_dir, split="train", download=True)
        CelebA(self.data_dir, split="test", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            celeba_full = CelebA(
                self.data_dir,
                split="train",
                transform=self.transform,
                target_type="attr",
                target_transform=None,
                download=False,
            )
            num_train = int(len(celeba_full) * self.train_val_ratio)
            num_val = len(celeba_full) - num_train
            self.celeba_train, self.celeba_val = random_split(
                celeba_full,
                [num_train, num_val],
            )

        if stage == "test" or stage is None:
            self.celeba_test = CelebA(
                self.data_dir,
                split="test",
                transform=self.transform,
                target_type="attr",
                target_transform=None,
                download=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.celeba_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.celeba_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.celeba_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
