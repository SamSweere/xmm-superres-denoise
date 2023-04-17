from pathlib import Path

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from transforms import Crop, Normalize


class BaseDataModule(LightningDataModule):
    def __init__(
            self,
            config
    ):
        super(BaseDataModule, self).__init__()

        self.num_workers = 0 if config["debug"] else 4
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

        # Display datasets
        self.sim_display_dataset = None
        self.real_display_dataset = None
        if config["display"]["sim_display_name"]:
            from datasets import XmmSimDataset
            self.sim_display_dataset = XmmSimDataset(
                dataset_dir=Path(config["dir"]) / "display_datasets" / "xmm_sim_display_selection",
                lr_res=config["lr_res"],
                hr_res=config["hr_res"],
                dataset_lr_res=config["dataset_lr_res"],
                mode=config["mode"],
                lr_exps=config["display"]["exposure"],
                hr_exp=config["hr_exp"],
                agn=False,
                lr_background=False,
                hr_background=False,
                det_mask=config["det_mask"],
                check_files=config["check_files"],
                transform=self.transform,
                normalize=self.normalize
            )
        if config["display"]["real_display_name"]:
            from datasets import XmmDataset
            self.real_display_dataset = XmmDataset(
                dataset_dir=Path(config["dir"]) / "display_datasets" / "xmm_split_display_selection",
                dataset_lr_res=config["dataset_lr_res"],
                lr_exps=config["display"]["exposure"],
                hr_exp=None,
                det_mask=config["det_mask"],
                check_files=config["check_files"],
                transform=self.transform,
                normalize=self.normalize
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader(self.train_subset, True)

    def _get_display_dataloaders(self) -> EVAL_DATALOADERS:
        dataloaders = []
        if self.real_display_dataset:
            dataloaders.append(self.get_dataloader(self.real_display_dataset))
        if self.sim_display_dataset:
            dataloaders.append(self.get_dataloader(self.sim_display_dataset))
        return dataloaders

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataloaders = [self.get_dataloader(self.val_subset)]
        dataloaders.extend(self._get_display_dataloaders())
        return dataloaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataloaders = [self.get_dataloader(self.test_subset)]
        dataloaders.extend(self._get_display_dataloaders())
        return dataloaders

    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
