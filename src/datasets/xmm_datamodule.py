import pickle as p
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import random_split, Subset

from datasets import BaseDataModule
from datasets.utils import find_img_dirs, find_img_files, match_file_list


class XmmDataModule(BaseDataModule):
    def __init__(
            self,
            config
    ):
        super(XmmDataModule, self).__init__(config)

        self.lr_exps = config["lr_exps"]
        self.hr_exp = config["hr_exp"]
        self.split_dict = defaultdict(Path)

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

            self.mode = ""
            self.res_mults = [""]
            self.split_key = "_image_split_"
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
                normalize=self.normalize,
            )

            self.mode = f"/{config['mode']}"
            self.res_mults = [f"{self.dataset.lr_res_mult}x", f"{self.dataset.hr_res_mult}x"]
            self.split_key = "_mult_"
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not known, options: 'real', 'sim'")

    def prepare_data(self) -> None:
        # Check that for every used exposure time there is a train/val/test split
        # If there is none, create one
        dataset_dir = str(self.dataset_dir.resolve())
        subset_str = f"{dataset_dir}/{{0}}/{{1}}ks{self.mode}"
        splits = ["train", "val", "test"]

        for res_mult in self.res_mults:
            for lr_exp in self.lr_exps:
                if self.mode:
                    paths = [Path(subset_str.format(split_name, lr_exp)) / f"{res_mult}.p" for split_name in splits]
                else:
                    paths = [Path(f"{subset_str.format(split_name, lr_exp)}.p") for split_name in splits]
                for split_name, path in zip(splits, paths):
                    self.split_dict[split_name] = path
                exists = np.all([path.exists() for path in paths])
                if not exists:
                    img_dirs = find_img_dirs(self.dataset_dir, np.asarray([lr_exp]), f"{self.mode}/{res_mult}")
                    img_files = find_img_files(img_dirs)
                    img_files = match_file_list(img_files, None, split_key=self.split_key)[0]
                    train, val, test = random_split(img_files, [0.8, 0.1, 0.1])
                    for path, split in zip(paths, [train, val, test]):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with open(path, 'w+b') as f:
                            p.dump(split.indices, f)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            with open(self.split_dict["train"], "rb") as f:
                train_indices = p.load(f)
                self.train_subset = Subset(self.dataset, train_indices)

            with open(self.split_dict["val"], "rb") as f:
                val_indices = p.load(f)
                self.val_subset = Subset(self.dataset, val_indices)
        if stage == "test":
            with open(self.split_dict["test"], "rb") as f:
                test_indices = p.load(f)
                self.test_subset = Subset(self.dataset, test_indices)
