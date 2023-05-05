from pathlib import Path

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from transforms import Crop, Normalize


class BaseDataModule(LightningDataModule):
    def __init__(
            self,
            config
    ):
        super(BaseDataModule, self).__init__()

        self.num_workers = 0 if config["debug"] else 12
        self.pin_memory = not config["debug"]
        self.persistent_workers = not config["debug"]

        self.batch_size = config["batch_size"]
        self.dataset_type = config["dataset_type"]

        self.transform = [
            Crop(
                crop_p=config["lr_res"] / config["dataset_lr_res"],
                mode=config["crop_mode"],
            ),
            ToTensor()
        ]

        self.normalize = Normalize(
            lr_max=config["lr_max"],
            hr_max=config["hr_max"],
            stretch_mode=config["data_scaling"],
        )

        self.dataset_dir = Path(config["dir"]) / config["dataset_name"]
        self.check_files = config["check_files"]

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
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
