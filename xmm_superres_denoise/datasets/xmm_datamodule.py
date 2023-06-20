import pickle
from pathlib import Path

import numpy as np
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from torch.utils.data import Subset, random_split

from xmm_superres_denoise.datasets import BaseDataModule
from xmm_superres_denoise.datasets.utils import (
    find_img_files,
    match_file_list,
    save_splits,
)


class XmmDataModule(BaseDataModule):
    def __init__(self, config):
        super(XmmDataModule, self).__init__(config)

        self.lr_exps = config["lr"]["exps"]
        self.hr_exp = config["hr"]["exp"]

        if self.dataset_type == "real":
            from xmm_superres_denoise.datasets import XmmDataset

            self.dataset = XmmDataset(
                dataset_dir=self.dataset_dir,
                dataset_lr_res=config["lr"]["res"],
                lr_exps=self.lr_exps,
                hr_exp=self.hr_exp,
                det_mask=config["det_mask"],
                check_files=self.check_files,
                transform=self.transform,
                normalize=self.normalize,
            )

            self.subset_str = f"res/splits/real_dataset/{{0}}/{{1}}.p"
        elif self.dataset_type == "sim":
            from xmm_superres_denoise.datasets import XmmSimDataset

            self.dataset = XmmSimDataset(
                dataset_dir=self.dataset_dir,
                lr_res=config["lr"]["res"],
                hr_res=config["hr"]["res"],
                dataset_lr_res=config["lr"]["res"],
                mode=config["mode"],
                lr_exps=self.lr_exps,
                hr_exp=self.hr_exp,
                lr_agn=config["lr"]["agn"],
                hr_agn=config["hr"]["agn"],
                lr_background=config["lr"]["background"],
                hr_background=config["hr"]["background"],
                det_mask=config["det_mask"],
                check_files=self.check_files,
                transform=self.transform,
                normalize=self.normalize,
            )

            self.subset_str = f"res/splits/sim_dataset/{{0}}/{self.dataset.mode}.p"
        elif self.dataset_type == "boring":
            from xmm_superres_denoise.datasets import BoringDataset

            rank_zero_warn(
                "You are using the BoringDataset which is meant for testing purposes!"
                "Ignore this warning if this was on purpose."
            )
            self.dataset = BoringDataset(
                lr_exp=self.lr_exps,
                hr_exp=self.hr_exp,
                hr_res_mult=config["hr"]["res"] // config["lr"]["res"],
            )
        else:
            raise ValueError(
                f"Dataset type {self.dataset_type} not known, options: 'real', 'sim'"
            )

    def _prepare_sim_dataset(self):
        splits = ["train", "val", "test"]
        paths = [Path(self.subset_str.format(split_name)) for split_name in splits]
        exists = np.all([path.exists() for path in paths])
        if not exists:
            rank_zero_info(
                f"Creating splits for {self.dataset_dir} with {self.dataset.base_name_count} base_names..."
            )
            train, val, test = random_split(
                range(self.dataset.base_name_count), [0.8, 0.1, 0.1]
            )
            save_splits(paths, [train, val, test])

    def _prepare_real_dataset(self):
        splits = ["train", "val", "test"]
        img_files = find_img_files(self.dataset.lr_img_dirs)
        img_files = match_file_list(img_files, None, self.dataset.split_key)[0]
        for lr_exp in self.dataset.lr_exps:
            paths = [
                Path(self.subset_str.format(split_name, lr_exp))
                for split_name in splits
            ]
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
            with open(self.subset_str.format(subset), "rb") as f:
                indices = pickle.load(f)
            mult = 1
            if exps_size > 1:
                mult *= exps_size

            if self.dataset.lr_background > 1:
                mult *= self.dataset.lr_background

            if self.dataset.lr_agn > 1:
                mult *= self.dataset.lr_agn

            indices = np.asarray([indices * (i + 1) for i in range(mult)])
            indices = np.concatenate(indices)
            return indices
        elif self.dataset_type == "real":
            used_lr_basenames = self.dataset.lr_img_files.index
            lr_exp = self.dataset.lr_exps[0]
            lr_img_files = find_img_files(self.dataset.lr_img_dirs)
            lr_img_files = match_file_list(
                {lr_exp: lr_img_files[lr_exp]}, None, self.dataset.split_key
            )[0]
            with open(self.subset_str.format(subset, lr_exp), "rb") as f:
                indices = pickle.load(f)
            used_indices = used_lr_basenames.get_indexer(lr_img_files.index)
            indices = np.asarray(list(set(indices) & set(used_indices)))
            if exps_size > 1:
                indices = np.asarray([indices * (i + 1) for i in range(exps_size)])
                indices = np.concatenate(indices)
            return indices

    def setup(self, stage: str) -> None:
        if self.dataset_type == "boring":
            train, val, test = random_split(self.dataset, [0.8, 0.1, 0.1])
            self.train_subset = Subset(self.dataset, train)
            self.val_subset = Subset(self.dataset, val)
            self.test_subset = Subset(self.dataset, test)
        else:
            if stage == "fit":
                train_indices = self._load_indices("train")
                self.train_subset = Subset(self.dataset, train_indices)

                val_indices = self._load_indices("val")
                self.val_subset = Subset(self.dataset, val_indices)
            if stage == "test" or "predict":
                test_indices = self._load_indices("test")
                self.test_subset = Subset(self.dataset, test_indices)
