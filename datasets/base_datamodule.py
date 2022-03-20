import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets.tng_dataset import TngDataset
import os


class BaseDataModule(LightningDataModule):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']

        if config['debug']:
            # For some reason the debugger does not like multiple workers
            self.num_workers = 0
            self.pin_memory = False
            self.persistent_workers = False
        else:
            # Take 1/4 of the cpu count for the dataloaders leaving some room for other processes
            self.num_workers = int(1 / 4 * os.cpu_count())
            self.pin_memory = True
            self.persistent_workers = True

        # Needs to be set
        self.dataset = dataset

    def get_dataloader(self, shuffle=False):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    # def prepare_data(self):
    #     # download, split, etc...
    #     # only called on 1 GPU/TPU in distributed
    #     return

    # def setup(self, stage= None):
    #     # make assignments here (val/train/test split)
    #     # called on every process in DDP
    #     train_val_test_split = [0.7, 0.15, 0.15]
    #
    #     train_len = int(len(self.dataset) * train_val_test_split[0])
    #     val_len = int(len(self.dataset) * train_val_test_split[1])
    #     test_len = len(self.dataset) - train_len - val_len
    #
    #     self.train, self.val, self.test = torch.utils.data.random_split(self.dataset,
    #                                                                                 [train_len, val_len, test_len],
    #                                                                                 generator=torch.Generator().manual_seed(
    #                                                                                     42))
    #
    #     print("Train set size:", len(self.train))
    #     print("Validation set size:", len(self.val))
    #     print("Test set size:", len(self.test))
    #
    #     # dataloaders = {x: torch.utils.data.DataLoader(tng_datasets[x], batch_size=configs.batch_size,
    #     #                                               shuffle=True, num_workers=0)
    #     #                for x in ['train', 'val', 'test']}

     #
    # def train_dataloader(self):
    #     return DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
    #                       pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
    #
    # def val_dataloader(self):
    #     return DataLoader(self.dataset['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
    #                       pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.dataset['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
    #                       pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
    #
    # def full_dataloader(self):
    #     return DataLoader(self.dataset['full'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
    #                       pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    # def teardown(self):
    #     # clean up after fit or test
    #     # called on every process in DDP
    #     return
