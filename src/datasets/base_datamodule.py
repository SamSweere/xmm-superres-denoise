import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]

        if config["debug"]:
            # For some reason the debugger does not like multiple workers
            self.num_workers = 0
            self.pin_memory = False
            self.persistent_workers = False
        else:
            # Take 1/4 of the cpu count for the dataloaders leaving some room for other processes
            self.num_workers = 3
            self.pin_memory = True
            self.persistent_workers = True

        # Needs to be set
        self.dataset = dataset

    def get_dataloader(self, shuffle=False):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
