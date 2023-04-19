import pickle as p
from pathlib import Path

import numpy as np
from torch.utils.data import random_split, Subset

from datasets import BaseDataModule
from datasets.utils import find_img_files, match_file_list


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
            print(f"Creating splits for {self.dataset_dir}...")
            print(f"\tDataset has {self.dataset.base_name_count} base_names")
            print(f"\tDataset has {self.dataset.lr_exps.size} lr_exps")
            train, val, test = random_split(range(self.dataset.base_name_count), [0.8, 0.1, 0.1])
            for path, split in zip(paths, [train, val, test]):
                lr_exp_count = self.dataset.lr_exps.size
                indices = split.indices
                for i in range(1, lr_exp_count):
                    tmp = np.asarray(indices) * (i + 1)
                    indices.extend(tmp.tolist())
                print(f"\tSplit {path} contains {len(indices)} images")
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w+b') as f:
                    p.dump(indices, f)

    def _prepare_real_dataset(self):
        splits = ["train", "val", "test"]
        img_files = find_img_files(self.dataset.lr_img_dirs)
        img_files = match_file_list(img_files, None, self.dataset.split_key)[0]
        for lr_exp in self.dataset.lr_exps:
            paths = [Path(self.subset_str.format(split_name, lr_exp)) for split_name in splits]
            exists = np.all([path.exists() for path in paths])
            if not exists:
                print(f"Creating splits for {self.dataset_dir}...")
                files = img_files[lr_exp]
                train, val, test = random_split(files, [0.7, 0.15, 0.15])
                for path, split in zip(paths, [train, val, test]):
                    indices = split.indices
                    print(f"\tSplit {path} contains {len(indices)} images")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, 'w+b') as f:
                        p.dump(indices, f)

    def prepare_data(self) -> None:
        # Check that for every used exposure time there is a train/val/test split
        # If there is none, create one
        if self.dataset_type == "sim":
            self._prepare_sim_dataset()
        elif self.dataset_type == "real":
            self._prepare_real_dataset()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            with open(self.subset_str.format("train"), "rb") as f:
                train_indices = p.load(f)
                self.train_subset = Subset(self.dataset, train_indices)

            with open(self.subset_str.format("val"), "rb") as f:
                val_indices = p.load(f)
                self.val_subset = Subset(self.dataset, val_indices)
        if stage == "test":
            with open(self.subset_str.format("test"), "rb") as f:
                test_indices = p.load(f)
                self.test_subset = Subset(self.dataset, test_indices)
