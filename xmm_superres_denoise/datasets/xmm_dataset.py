from pathlib import Path
from random import sample
from typing import Callable, List, Optional

import numpy as np
from datasets.utils import (apply_transform, check_img_files, find_img_dirs,
                            find_img_files, load_det_mask, load_fits,
                            match_file_list, reshape_img_to_res)
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import Dataset
from transforms import Normalize


class XmmDataset(Dataset):
    """XMM-Newton dataset"""

    def __init__(
        self,
        dataset_dir: Path,
        dataset_lr_res: int,
        lr_exps: List[int],
        hr_exp: Optional[int],
        det_mask: bool = False,
        check_files: bool = False,
        transform: List[Callable] = None,
        normalize: Normalize = None,
        split_key: str = "_image_split_",
    ):
        """
        Args:
            dataset_dir (Path): Directory of the datasets
            dataset_lr_res (int): The resolution the input images are transformed to make them rectangular
                before any cropping
            lr_exps (list): Exposure of the low resolution images, in ks, if list it is the range of exposure times
            hr_exp (int): Exposure of the high resolution images, in ks
            det_mask (bool): Multiply the final image with the detectormask
            check_files (bool) (optional): Check the file integrity of the chosen dataset
        """
        if hr_exp:
            rank_zero_info("\thr_exps is not empty => Will use hr images")

        self.transform = transform if transform else []
        self.normalize = normalize
        self.split_key = split_key

        self.det_mask = det_mask
        self.dataset_lr_res = dataset_lr_res

        self.lr_exps: np.ndarray = np.asarray(lr_exps)
        self.hr_exp: Optional[np.ndarray] = np.asarray([hr_exp]) if hr_exp else None
        self.lr_res_mult = 1

        # Get all the image directories
        self.lr_img_dirs = find_img_dirs(dataset_dir, self.lr_exps)
        lr_img_files = find_img_files(self.lr_img_dirs)

        if self.hr_exp:
            self.hr_img_dirs = find_img_dirs(dataset_dir, self.hr_exp)
            hr_img_files = find_img_files(self.hr_img_dirs)
        else:
            hr_img_files = None

        self.lr_img_files, self.hr_img_files, self.base_name_count = match_file_list(
            lr_dict=lr_img_files, hr_dict=hr_img_files, split_key=split_key
        )

        self.dataset_size = self.base_name_count * len(self.lr_exps)
        rank_zero_info(
            f"\tOverall dataset size: img_count * lr_exps_count = dataset_size"
        )
        rank_zero_info(
            f"\t\t{self.base_name_count} * {len(self.lr_exps)} = {self.dataset_size}"
        )

        # Load the detector masks for the lr and hr resolutions
        if self.det_mask:
            # Since the det mask will be used on every image we load them into memory
            self.lr_det_mask = load_det_mask(self.lr_res_mult)
            # This is on the same resolution, thus the same detmask
            self.hr_det_mask = self.lr_det_mask if self.hr_exp else None

        if check_files:
            # Check the file integrity
            check_img_files(self.lr_img_files, (411, 403), "Checking lr_img_files...")
            if self.hr_img_files:
                check_img_files(
                    self.hr_img_files, (411, 403), "Checking hr_img_files..."
                )

            rank_zero_info("\tAll files are within specifications!")

    def __len__(self):
        return self.dataset_size

    def load_xmm_sample(self, idx):
        lr_exp = idx % len(self.lr_exps)
        base_name = idx % self.base_name_count

        lr_img_path = sample(self.lr_img_files.iloc[base_name].iloc[lr_exp], 1)[0]
        hr_img_path = (
            sample(self.hr_img_files.iloc[base_name].iloc[0], 1)[0]
            if self.hr_exp
            else None
        )

        lr_img = load_fits(lr_img_path)
        hr_img = load_fits(hr_img_path) if hr_img_path else None

        if self.lr_det_mask is not None:
            lr_img["img"] *= self.lr_det_mask  # Note the *=
        lr_img["img"] = reshape_img_to_res(
            dataset_lr_res=self.dataset_lr_res, img=lr_img["img"], res_mult=1
        )

        if hr_img:
            if self.hr_det_mask is not None:
                hr_img["img"] *= self.hr_det_mask
            hr_img["img"] = reshape_img_to_res(
                dataset_lr_res=self.dataset_lr_res, img=hr_img["img"], res_mult=1
            )

        return lr_img, hr_img

    def __getitem__(self, idx):
        # Load a sample based on the given index
        lr_img_sample, hr_img_sample = self.load_xmm_sample(idx=idx)

        lr_img = apply_transform(lr_img_sample["img"], self.transform)
        lr_img = self.normalize.normalize_lr_image(lr_img) if self.normalize else lr_img

        item = {
            "lr": lr_img,
            "lr_exp": lr_img_sample["exp"] // 1000,
            "lr_img_file_name": lr_img_sample["file_name"],
            "lr_header": lr_img_sample["header"],
        }

        if hr_img_sample:
            hr_img = apply_transform(hr_img_sample["img"], self.transform)
            hr_img = (
                self.normalize.normalize_hr_image(hr_img) if self.normalize else hr_img
            )

            item["hr"] = hr_img
            item["hr_exp"] = hr_img_sample["exp"] // 1000
            item["hr_header"] = hr_img_sample["header"]
            item["hr_img_file_name"] = hr_img_sample["file_name"]

        return item
