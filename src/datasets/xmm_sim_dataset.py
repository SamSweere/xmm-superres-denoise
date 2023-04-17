from pathlib import Path
from random import sample, randint
from typing import List, Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from datasets.utils import find_img_dirs, find_img_files, match_file_list, apply_transform
from datasets.utils import load_fits, reshape_img_to_res, load_det_mask, check_img_files
from transforms import Normalize


class XmmSimDataset(Dataset):
    """XMM-Newton simulated dataset"""

    def __init__(
            self,
            dataset_dir: Path,
            lr_res: int,
            hr_res: int,
            dataset_lr_res: int,
            mode: str,
            lr_exps: List[int],
            hr_exp: int,
            agn: bool,
            lr_background: int,
            hr_background: bool,
            det_mask: bool = False,
            check_files: bool = False,
            transform: List[Callable] = None,
            normalize: Normalize = None
    ):
        """
        Args:
            dataset_dir (Path): Directory of the datasets
            lr_res (int): The resolution of the final input images (low resolution)
            hr_res (int): The resolution of the final target images (high resolution)
            dataset_lr_res (int): The resolution the input images are transformed to make them rectangular
                before any cropping
            mode (string): The mode, img or agn. If agn the model samples from agn as if it is an image
            lr_exps (list): Exposure of the low resolution images, in ks, if list it is the range of exposure times
            hr_exp (int): Exposure of the high resolution images, in ks
            agn (bool): Add agns to the images
            lr_background (int): Add background to low resolution
            hr_background (bool): Add background to high resolution
            det_mask (bool): Multiply the final image with the detectormask
            check_files (bool) (optional): Check the file integrity of the chosen dataset
            transform (callable) (optional): Optional transform to be applied
            normalize (callable) (optional): Optional normalization to be applied
        """
        self.transform = transform if transform else []
        self.normalize = normalize

        self.lr_res = lr_res
        self.hr_res = hr_res
        self.mode = mode

        self.lr_exps: np.ndarray = np.asarray(lr_exps)
        self.hr_exp: np.ndarray = np.asarray([hr_exp])

        self.agn = agn
        self.det_mask = det_mask
        self.lr_background = lr_background
        self.hr_background = hr_background

        self.lr_res_mult = 1
        self.hr_res_mult = self.hr_res // self.lr_res

        # Set the image dimensions
        self.dataset_lr_res = dataset_lr_res
        self.dataset_hr_res = self.dataset_lr_res * self.hr_res_mult

        # Get all the image directories
        # Note that if the mode is agn we consider them as the base images
        lr_img_dirs = find_img_dirs(dataset_dir, self.lr_exps, f"/{self.mode}/{self.lr_res_mult}x")
        lr_img_files = find_img_files(lr_img_dirs)

        hr_img_dirs = find_img_dirs(dataset_dir, self.hr_exp, f"/{self.mode}/{self.hr_res_mult}x")
        hr_img_files = find_img_files(hr_img_dirs)

        self.lr_img_files, self.hr_img_files, self.base_name_count = match_file_list(lr_img_files, hr_img_files,
                                                                                     "_mult_")  # TODO move to parameters

        print(f"\tFound {self.base_name_count} image pairs (lr and hr simulation matches) in {dataset_dir}")

        if self.lr_background > 0:
            self.dataset_size = self.base_name_count * self.lr_exps.size * self.lr_background
            print(f"\tOverall dataset size: img_count * lr_exps_count * lr_background_count = dataset_size")
            print(f"\t\t{self.base_name_count} * {self.lr_exps.size} * {self.lr_background} = {self.dataset_size}")
        else:
            self.dataset_size = self.base_name_count * self.lr_exps.size
            print(f"\tOverall dataset size: img_count * lr_exps_count = dataset_size")
            print(f"\t\t{self.base_name_count} * {self.lr_exps.size} = {self.dataset_size}")

        if self.agn:
            lr_agn_dirs = find_img_dirs(dataset_dir, self.lr_exps, f"/agn/{self.lr_res_mult}x")
            lr_agn_files = find_img_files(lr_agn_dirs)

            hr_agn_dirs = find_img_dirs(dataset_dir, self.hr_exp, f"/agn/{self.hr_res_mult}x")
            hr_agn_files = find_img_files(hr_agn_dirs)

            self.lr_agn_files, self.hr_agn_files, self.base_agn_count = match_file_list(lr_agn_files,
                                                                                        hr_agn_files,
                                                                                        "_mult_")
            print(f"\tFound {self.base_agn_count} agn image pairs (lr and hr simulation matches)")

        if self.lr_background > 0:
            lr_background_dirs = find_img_dirs(dataset_dir, self.lr_exps, f"/background/{self.lr_res_mult}x")
            lr_background_files = find_img_files(lr_background_dirs)
            amt = min([len(file_list) for file_list in lr_background_files.values()])
            self.lr_background_files = {}
            for exp, files in lr_background_files.items():
                self.lr_background_files[exp] = sample(files, amt)
            self.lr_background_files = pd.DataFrame.from_dict(self.lr_background_files)

        if self.hr_background:
            hr_background_dirs = find_img_dirs(dataset_dir, self.hr_exp, f"/background/{self.hr_res_mult}x")
            hr_img_files = find_img_files(hr_background_dirs)
            self.hr_background_files = pd.DataFrame.from_dict(hr_img_files)

        # Load the detector masks for the lr and hr resolutions
        if self.det_mask:
            self.lr_det_mask = load_det_mask(self.lr_res_mult)
            self.hr_det_mask = load_det_mask(self.hr_res_mult)

        if check_files:
            # Check the file integrity
            check_img_files(self.lr_img_files, (411, 403), "Checking lr_img_files...")
            check_img_files(self.hr_img_files, (411 * self.hr_res_mult, 403 * self.hr_res_mult),
                            "Checking hr_img_files...")

            if self.agn:
                check_img_files(self.lr_agn_files, (411, 403), "Checking lr_agn_files...")
                check_img_files(self.hr_agn_files, (411 * self.hr_res_mult, 403 * self.hr_res_mult),
                                "Checking hr_agn_files...")

            if self.lr_background > 0:
                check_img_files(self.lr_background_files, (411, 403), "Checking lr_background_files...")

            if self.hr_background:
                check_img_files(self.hr_background_files, (411 * self.hr_res_mult, 403 * self.hr_res_mult),
                                "Checking hr_background_files...")

            print("\tAll files are within specifications!")

    def load_and_combine_simulations(
            self,
            res_mult,
            img_path,
            agn_path=None,
            background_path=None,
            det_mask=None
    ):
        # Load the image data
        img = load_fits(img_path)

        if agn_path:
            img["img"] += load_fits(agn_path)["img"]

        if background_path:
            img["img"] += load_fits(background_path)["img"]

        if det_mask:
            img["img"] *= det_mask  # Note the *=

        # The image has the shape (411, 403), we pad/crop this to (dataset_lr_res, dataset_lr_res)
        img["img"] = reshape_img_to_res(dataset_lr_res=self.dataset_lr_res, img=img["img"], res_mult=res_mult)

        return img

    def __len__(self):
        return self.dataset_size

    def load_lr_hr_xmm_sample(self, idx):
        lr_exp = idx % self.lr_exps.size
        base_name = idx % self.base_name_count

        lr_img_path = sample(self.lr_img_files.iloc[base_name].iloc[lr_exp], 1)[0]
        hr_img_path = sample(self.hr_img_files.iloc[base_name].iloc[0], 1)[0]

        lr_agn_path = None  # Define them as None in the case that self.agn is False
        hr_agn_path = None
        if self.agn:
            agn_idx = randint(0, self.base_agn_count - 1)
            lr_agn_path = sample(self.lr_agn_files.iloc[agn_idx].iloc[lr_exp], 1)[0]
            hr_agn_path = sample(self.hr_agn_files.iloc[agn_idx].iloc[0], 1)[0]

        lr_background_path = None
        if self.lr_background > 0:
            lr_background_path = self.lr_background_files[self.lr_exps[lr_exp]].sample(1).item()

        hr_background_path = None
        if self.hr_background:
            hr_background_path = self.hr_background_files[self.hr_exp].sample(1).item()

        # Load and combine the selected files
        lr_img = self.load_and_combine_simulations(
            res_mult=self.lr_res_mult,
            img_path=lr_img_path,
            agn_path=lr_agn_path,
            background_path=lr_background_path,
            det_mask=self.lr_det_mask
        )
        hr_img = self.load_and_combine_simulations(
            res_mult=self.hr_res_mult,
            img_path=hr_img_path,
            agn_path=hr_agn_path,
            background_path=hr_background_path,
            det_mask=self.hr_det_mask
        )

        return lr_img, hr_img

    def __getitem__(self, idx):
        # Load a sample based on the given index
        lr_img_sample, hr_img_sample = self.load_lr_hr_xmm_sample(idx=idx)

        lr_img = apply_transform(lr_img_sample["img"], self.transform) if self.transform else lr_img_sample["img"]
        hr_img = apply_transform(hr_img_sample["img"], self.transform) if self.transform else hr_img_sample["img"]

        lr_img = self.normalize.normalize_lr_image(lr_img) if self.normalize else lr_img
        hr_img = self.normalize.normalize_hr_image(hr_img) if self.normalize else hr_img

        item = {
            "lr": lr_img,
            "hr": hr_img,
            "lr_exp": lr_img_sample["exp"] // 1000,
            "hr_exp": hr_img_sample["exp"] // 1000,
            "lr_img_file_name": lr_img_sample["file_name"],
            "tng_set": lr_img_sample["file_name"].split("_")[0],
        }

        return item
