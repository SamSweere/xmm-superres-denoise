from pathlib import Path

from config.config import DatasetCfg
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from transforms import Crop, Normalize


class BaseDataModule(LightningDataModule):
    def __init__(self, config: DatasetCfg):
        super().__init__()
        self.config = config

        self.num_workers = 0 if config.debug else 12
        self.pin_memory = self.persistent_workers = not config.debug

        self.transform = [
            Crop(
                crop_p=1.0,  # TODO
                mode=self.config.crop_mode,
            ),
            ToTensor(),
        ]

        self.normalize = Normalize(
            lr_max=config.lr.clamp_max,
            hr_max=config.hr.clamp_max,
            stretch_mode=config.scaling,
        )

        self.dataset_dir = Path(config.directory) / config.name

        self.dataset = None
        self.train_subset = None
        self.val_subset = None
        self.test_subset = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader(self.train_subset, True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.val_subset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.test_subset)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.test_subset)

    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
