import pickle
from pathlib import Path

import numpy as np
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import random_split, Subset

from datasets import BaseDataModule
from datasets.utils import find_img_files, match_file_list, save_splits


class XmmDataModule(BaseDataModule):
    def __init__(
            self,
            config
    ):
        super(XmmDataModule, self).__init__(config)

        self.lr_exps = config["lr_exps"]
        self.hr_exp = config["hr_exp"]

        if self.dataset_type == "real":
            from datasets import XmmDataset
            self.dataset = XmmDataset(
                dataset_dir=self.dataset_dir,
                dataset_lr_res=config["dataset_lr_res"],
                lr_exps=self.lr_exps,
                hr_exp=self.hr_exp,
                det_mask=config["det_mask"],
                check_files=self.check_files,
                transform=self.transform,
                normalize=self.normalize
            )

            self.subset_str = f"res/splits/real_dataset/{{0}}/{{1}}.p"
        elif self.dataset_type == "sim":
            from datasets import XmmSimDataset
            self.dataset = XmmSimDataset(
                dataset_dir=self.dataset_dir,
                lr_res=config["lr_res"],
                hr_res=config["hr_res"],
                dataset_lr_res=config["dataset_lr_res"],
                mode=config["mode"],
                lr_exps=self.lr_exps,
                hr_exp=self.hr_exp,
                agn=config["agn"],
                lr_background=config["lr_background"],
                hr_background=config["hr_background"],
                det_mask=config["det_mask"],
                check_files=self.check_files,
                transform=self.transform,
                normalize=self.normalize
            )

            self.subset_str = f"res/splits/sim_dataset/{{0}}/{self.dataset.mode}.p"
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not known, options: 'real', 'sim'")

    def _prepare_sim_dataset(self):
        splits = ["train", "val", "test"]
        paths = [Path(self.subset_str.format(split_name)) for split_name in splits]
        exists = np.all([path.exists() for path in paths])
        if not exists:
            rank_zero_info(f"Creating splits for {self.dataset_dir}...")
            rank_zero_info(f"\tDataset has {self.dataset.base_name_count} base_names")
            train, val, test = random_split(range(self.dataset.base_name_count), [0.8, 0.1, 0.1])
            save_splits(paths, [train, val, test])

    def _prepare_real_dataset(self):
        splits = ["train", "val", "test"]
        img_files = find_img_files(self.dataset.lr_img_dirs)
        img_files = match_file_list(img_files, None, self.dataset.split_key)[0]
        for lr_exp in self.dataset.lr_exps:
            paths = [Path(self.subset_str.format(split_name, lr_exp)) for split_name in splits]
            exists = np.all([path.exists() for path in paths])
            if not exists:
                rank_zero_info(f"Creating splits for {self.dataset_dir}...")
                files = img_files[lr_exp]
                train, val, test = random_split(files, [0.7, 0.15, 0.15])
                save_splits(paths, [train, val, test])

    def prepare_data(self) -> None:
        # Check that for every used exposure time there is a train/val/test split
        # If there is none, create one
        if self.dataset_type == "sim":
            self._prepare_sim_dataset()
        elif self.dataset_type == "real":
            self._prepare_real_dataset()

    def _load_indices(self, subset: str):
        exps_size = self.dataset.lr_exps.size
        rank_zero_info(f"Loading subset '{subset}'...")
        if self.dataset_type == "sim":
            with open(self.subset_str.format(subset), 'rb') as f:
                indices = pickle.load(f)
                rank_zero_info(f"\tDataset has {self.dataset.base_name_count} base_names "
                               f"out of which {indices.size} are used in this subset")
                rank_zero_info(f"\tDataset has {exps_size} lr_exps")
                indices = np.asarray([indices * (i + 1) for i in range(exps_size)])
                if exps_size > 1:
                    indices = np.concatenate(indices)
                rank_zero_info(f"\t'{subset}' has {indices.size} images.")
                return indices
        elif self.dataset_type == "real":
            # TODO
            pass

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_indices = self._load_indices("train")
            self.train_subset = Subset(self.dataset, train_indices)

            val_indices = self._load_indices("val")
            self.val_subset = Subset(self.dataset, val_indices)
        if stage == "test":
            test_indices = self._load_indices("test")
            self.test_subset = Subset(self.dataset, test_indices)
