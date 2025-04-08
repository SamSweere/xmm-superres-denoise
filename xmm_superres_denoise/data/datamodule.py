import pickle
from pathlib import Path

import numpy as np
from data.config import DatasetCfg, DatasetType
from data.dataset import XmmDataset
from data.tools import save_splits
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from loguru import logger
from torch.utils.data import DataLoader, Subset, random_split


class XmmDataModule(LightningDataModule):
    def __init__(self, config: DatasetCfg):
        super().__init__()

        self.config = config

        self.dataset = XmmDataset(config=self.config)
        self.train_subset = None
        self.val_subset = None
        self.test_subset = None

        self.subset_str = None
        # TODO This probably needs fixing
        match self.config.type:
            case DatasetType.REAL:
                self.subset_str = f"res/splits/{self.config.name}/{{0}}/{{1}}ks.p"
            case DatasetType.SIM:
                self.subset_str = f"res/splits/{self.config.name}/{{0}}.p"

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
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def on_exception(self, exception):
        logger.exception(f"Cleaning up datamodule because of exception.", exception)
        self._cleanup()

    def teardown(self, stage):
        logger.info(f"Cleaning up datamodule after stage {stage}.")
        self._cleanup()

    def _cleanup(self):
        for tar in self.dataset.lq_img:
            tar.close()
        logger.success("Closed all lq_img tars.")

        for tar in self.dataset.hq_img:
            tar.close()
        logger.success("Closed all hq_img tars.")

        if self.dataset.lq_agn is not None:
            for tar in self.dataset.lq_agn:
                tar.close()
            logger.success("Closed all lq_agn tars.")

            for tar in self.dataset.hq_agn:
                tar.close()
            logger.success("Closed all hq_agn tars.")

        if self.dataset.bkg is not None:
            for tar in self.dataset.bkg:
                tar.close()
            logger.success("Closed all bkg tars.")

    def _prepare_sim_dataset(self):
        splits = ["train", "val", "test"]
        paths = [Path(self.subset_str.format(split_name)) for split_name in splits]
        exists = np.all([path.exists() for path in paths])
        if not exists:
            logger.info(
                f"Creating splits for {self.dataset.base_path} with {len(self.dataset)} samples..."
            )
            train, val, test = random_split(range(len(self.dataset)), [0.8, 0.1, 0.1])
            save_splits(paths, [train, val, test])

    def _prepare_real_dataset(self):
        splits = ["train", "val", "test"]
        paths = [
            Path(self.subset_str.format(split_name, self.config.lq.exp))
            for split_name in splits
        ]
        exists = np.all([path.exists() for path in paths])
        if not exists:
            logger.info(f"Creating splits for {self.dataset.base_path}...")
            train, val, test = random_split(range(len(self.dataset)), [0.7, 0.15, 0.15])
            save_splits(paths, [train, val, test])

    def prepare_data(self) -> None:
        # Check that for every used exposure time there is a train/val/test split
        # If there is none, create one
        match self.config.type:
            case DatasetType.SIM:
                self._prepare_sim_dataset()
            case DatasetType.REAL:
                self._prepare_real_dataset()

    def _load_indices(self, subset: str):
        match self.config.type:
            case DatasetType.SIM:
                with open(self.subset_str.format(subset), "rb") as f:
                    indices = pickle.load(f)

                return indices
            case DatasetType.REAL:
                # TODO Fix this
                used_lr_basenames = self.dataset.lr_img_files.index
                lr_exp = self.config.lq.exp
                with open(self.subset_str.format(subset, lr_exp), "rb") as f:
                    indices = pickle.load(f)
                used_indices = used_lr_basenames.get_indexer(
                    self.dataset.lr_img_files.index
                )
                indices = np.asarray(list(set(indices) & set(used_indices)))

                return indices

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_indices = self._load_indices("train")
            self.train_subset = Subset(self.dataset, train_indices)

            val_indices = self._load_indices("val")
            self.val_subset = Subset(self.dataset, val_indices)
            return

        if stage == "test" or "predict":
            test_indices = self._load_indices("test")
            self.test_subset = Subset(self.dataset, test_indices)
            return
