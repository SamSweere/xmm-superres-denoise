from pathlib import Path

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from xmm_superres_denoise.transforms import Crop, Normalize


class BaseDataModule(LightningDataModule):
    def __init__(self, config):
        super(BaseDataModule, self).__init__()

        self.num_workers = 0 if config["debug"] else 12
        self.pin_memory = not config["debug"]
        self.persistent_workers = not config["debug"]

        self.batch_size = config["batch_size"]
        self.dataset_type = config["type"]

        self.transform = [
            Crop(
                crop_p=1.0,  # TODO
                mode=config["crop_mode"],
            ),
            ToTensor(),
        ]

        self.normalize = Normalize(
            lr_max=config["lr"]["max"],
            hr_max=config["hr"]["max"],
            config = config,
            stretch_mode=config["scaling"],
            clamp = config["clamp"]
        )

        self.dataset_dir = Path(config["dir"]) / config["name"]
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
