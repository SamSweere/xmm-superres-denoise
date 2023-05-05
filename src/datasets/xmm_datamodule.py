import pickle
from pathlib import Path

import numpy as np
from lightning.pytorch.utilities import rank_zero_info
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
            rank_zero_info(f"Creating splits for {self.dataset_dir} with {self.dataset.base_name_count} base_names...")
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
        if self.dataset_type == "sim":
            with open(self.subset_str.format(subset), 'rb') as f:
                indices = pickle.load(f)
                if exps_size > 1:
                    indices = np.asarray([indices * (i + 1) for i in range(exps_size)])
                    indices = np.concatenate(indices)
                rank_zero_info(f"\tDataset has {self.dataset.base_name_count} base_names "
                               f"out of which {indices.size // exps_size} are used in {subset}_subset. "
                               f"Due to {exps_size} exps, this subset has {indices.size} images.")
                return indices
        elif self.dataset_type == "real":
            used_lr_basenames = self.dataset.lr_img_files.index
            lr_exp = self.dataset.lr_exps[0]
            lr_img_files = find_img_files(self.dataset.lr_img_dirs)
            lr_img_files = match_file_list({lr_exp: lr_img_files[lr_exp]}, None, self.dataset.split_key)[0]
            with open(self.subset_str.format(subset, lr_exp), 'rb') as f:
                indices = pickle.load(f)
            used_indices = used_lr_basenames.get_indexer(lr_img_files.index)
            indices = np.asarray(list(set(indices) & set(used_indices)))
            if exps_size > 1:
                indices = np.asarray([indices * (i + 1) for i in range(exps_size)])
                indices = np.concatenate(indices)
            return indices

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_indices = self._load_indices("train")
            self.train_subset = Subset(self.dataset, train_indices)

            val_indices = self._load_indices("val")
            self.val_subset = Subset(self.dataset, val_indices)
        if stage == "test" or "predict":
            test_indices = self._load_indices("test")
            self.test_subset = Subset(self.dataset, test_indices)
