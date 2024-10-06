from pathlib import Path
from random import randint, sample
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset
import torch

from xmm_superres_denoise.datasets.utils import (
    apply_transform,
    check_img_files,
    find_img_dirs,
    find_img_files,
    load_det_mask,
    load_fits,
    match_file_list,
    reshape_img_to_res,
)
from xmm_superres_denoise.transforms import Normalize


class XmmSimDataset(Dataset):
    """XMM-Newton simulated dataset"""

    def __init__(
        self,
        dataset_dir: Path,
        deblend_agn_dir: Path,
        lr_res: int,
        hr_res: int,
        dataset_lr_res: int,
        mode: str,
        lr_exps: List[int],
        hr_exp: int,
        lr_agn: int,
        hr_agn: bool,
        lr_background: int,
        hr_background: bool,
        det_mask: bool = False,
        constant_img_combs: bool = False,
        check_files: bool = False,
        transform: List[Callable] = None,
        normalize: Optional[Normalize] = None,
        split_key: str = "_mult_",
    ):
        """
        Args:
            dataset_dir (Path): Directory of the xmm_superres_denoise.datasets
            lr_res (int): The resolution of the final input images (low resolution)
            hr_res (int): The resolution of the final target images (high resolution)
            dataset_lr_res (int): The resolution the input images are transformed to make them rectangular
                before any cropping
            mode (string): The mode, img or agn. If agn the model samples from agn as if it is an image
            lr_exps (list): Exposure of the low resolution images, in ks, if list it is the range of exposure times
            hr_exp (int): Exposure of the high resolution images, in ks
            lr_agn (int): Amount of agns to be used with one low resolution image. Set to 0 to use no agns.
            lr_background (int): Add background to low resolution
            hr_agn (bool): Include agns in the high resolution images
            hr_background (bool): Add background to high resolution
            det_mask (bool): Multiply the final image with the detectormask
            constant_img_combs (bool): Assign the same AGN and background component per img idx everytime
            check_files (bool) (optional): Check the file integrity of the chosen dataset
            transform (callable) (optional): Optional transform to be applied
            normalize (callable) (optional): Optional normalization to be applied
        """
        self.transform = transform if transform else []
        self.normalize = normalize
        self.split_key = split_key

        self.lr_res = lr_res
        self.hr_res = hr_res
        self.mode = mode

        self.lr_exps: np.ndarray = np.asarray(lr_exps)
        self.hr_exp: np.ndarray = np.asarray([hr_exp])

        self.lr_agn = lr_agn
        self.hr_agn = hr_agn
        self.det_mask = det_mask
        self.constant_img_combs = constant_img_combs
        self.lr_background = lr_background
        self.hr_background = hr_background

        self.lr_res_mult = 1
        self.hr_res_mult = self.hr_res // self.lr_res

        # Set the image dimensions
        self.dataset_lr_res = dataset_lr_res
        self.dataset_hr_res = self.dataset_lr_res * self.hr_res_mult
        
        self.deblend_agn_dir = deblend_agn_dir

        # Get all the image directories
        # Note that if the mode is agn we consider them as the base images
        self.lr_img_dirs = find_img_dirs(
            dataset_dir, self.lr_exps, f"{self.mode}/{self.lr_res_mult}x"
        )
        lr_img_files = find_img_files(self.lr_img_dirs)

        self.hr_img_dirs = find_img_dirs(
            dataset_dir, self.hr_exp, f"{self.mode}/{self.hr_res_mult}x"
        )
        hr_img_files = find_img_files(self.hr_img_dirs)

        self.lr_img_files, self.hr_img_files, self.base_name_count = match_file_list(
            lr_img_files, hr_img_files, split_key
        )

        self.dataset_size = self.base_name_count * self.lr_exps.size
        msg1 = f"Overall dataset size: img_count * lr_exps_count"
        msg2 = f"{self.base_name_count} * {self.lr_exps.size}"

        if self.lr_agn > 0:
            msg1 = msg1 + " * lr_agn_count"
            msg2 = msg2 + f" * {self.lr_agn}"
            self.dataset_size = self.dataset_size * self.lr_agn
           
            agn_folder_name = deblend_agn_dir if deblend_agn_dir else 'agn'
 
            lr_agn_dirs = find_img_dirs(
            dataset_dir, self.lr_exps, f"{agn_folder_name}/{self.lr_res_mult}x"
            )
            
            lr_agn_files = find_img_files(lr_agn_dirs)
            hr_agn_dirs = find_img_dirs(
                dataset_dir, self.hr_exp, f"{agn_folder_name}/{self.hr_res_mult}x"
            )
            
            hr_agn_files = find_img_files(hr_agn_dirs)
            self.lr_agn_files, self.hr_agn_files, self.base_agn_count = match_file_list(
                lr_agn_files, hr_agn_files, split_key
            )
            rank_zero_info(
                f"\tFound {self.base_agn_count} agn image pairs (lr and hr simulation matches)"
            )

        if self.lr_background > 0:
            msg1 = msg1 + " * lr_background_count"
            msg2 = msg2 + f" * {self.lr_background}"
            self.dataset_size = self.dataset_size * self.lr_background
            lr_background_dirs = find_img_dirs(
                dataset_dir, self.lr_exps, f"background/{self.lr_res_mult}x"
            )
            lr_background_files = find_img_files(lr_background_dirs)
            amt = min([len(file_list) for file_list in lr_background_files.values()])
            self.lr_background_files = {}
            for exp, files in lr_background_files.items():
                self.lr_background_files[exp] = sample(files, amt)
            self.lr_background_files = pd.DataFrame.from_dict(self.lr_background_files)

        if self.hr_background:
            hr_background_dirs = find_img_dirs(
                dataset_dir, self.hr_exp, f"background/{self.hr_res_mult}x"
            )
            hr_img_files = find_img_files(hr_background_dirs)
            self.hr_background_files = pd.DataFrame.from_dict(hr_img_files)

        # Load the detector masks for the lr and hr resolutions
        if self.det_mask:
            self.lr_det_mask = load_det_mask(self.lr_res_mult)
            self.hr_det_mask = load_det_mask(self.hr_res_mult)

        if check_files:
            # Check the file integrity
            check_img_files(self.lr_img_files, (411, 403), "Checking lr_img_files...")
            check_img_files(
                self.hr_img_files,
                (411 * self.hr_res_mult, 403 * self.hr_res_mult),
                "Checking hr_img_files...",
            )

            if self.lr_agn > 0:
                check_img_files(
                    self.lr_agn_files, (411, 403), "Checking lr_agn_files..."
                )

            if self.hr_agn:
                check_img_files(
                    self.hr_agn_files,
                    (411 * self.hr_res_mult, 403 * self.hr_res_mult),
                    "Checking hr_agn_files...",
                )

            if self.lr_background > 0:
                check_img_files(
                    self.lr_background_files,
                    (411, 403),
                    "Checking lr_background_files...",
                )

            if self.hr_background:
                check_img_files(
                    self.hr_background_files,
                    (411 * self.hr_res_mult, 403 * self.hr_res_mult),
                    "Checking hr_background_files...",
                )

            rank_zero_info("\tAll files are within specifications!")

        rank_zero_info(f"\t{msg1} = dataset_size")
        rank_zero_info(f"\t\t{msg2} = {self.dataset_size}")

    def load_and_combine_simulations(
        self, res_mult, img_path, agn_path=None, background_path=None, det_mask=None
    ):
        # Load the image data
        img = load_fits(img_path)

        if agn_path:
            agn_img = load_fits(agn_path)["img"]
            
            #TODO: Find a more elegant way to solve this issue 
            # The blended hr agn images have dimensions of 822 x 805 instead of 822 x 806, so I am adding an extra column here 
            if self.deblend_agn_dir and res_mult == 2:
                zeros_column = np.zeros((agn_img.shape[0], 1))  # Create a column of zeros with the same number of rows as 'array'
                agn_img = np.hstack((agn_img, zeros_column)) 
            img["img"] += agn_img
            
        if background_path:
            img["img"] += load_fits(background_path)["img"]

        if det_mask is not None:
            # The 4x resolution images were made with the new simulator and have slightly different dimensions
            if res_mult == 4:
                img["img"] = np.pad(img["img"], ((0, 1), (0, 3)), mode='constant', constant_values=0)   
            img["img"] *= det_mask  # Note the *=

        # The image has the shape (411, 403), we pad/crop this to (dataset_lr_res, dataset_lr_res)
        img["img"] = reshape_img_to_res(
            dataset_lr_res=self.dataset_lr_res, img=img["img"], res_mult=res_mult
        )

        return img
    
    def __len__(self):
        return self.dataset_size
    
    def define_paths_random_combinations(self, lr_exp, base_name):
        
        lr_img_path = sample(self.lr_img_files.iloc[base_name].iloc[lr_exp], 1)[0]
        hr_img_path = sample(self.hr_img_files.iloc[base_name].iloc[0], 1)[0]

        lr_agn_path = None  # Define them as None in the case that self.agn is False
        hr_agn_path = None
        if self.lr_agn > 0 or self.hr_agn:
            agn_idx = randint(0, self.base_agn_count - 1)

            if self.lr_agn > 0:
                lr_agn_path = sample(self.lr_agn_files.iloc[agn_idx].iloc[lr_exp], 1)[0]

            if self.hr_agn:
                hr_agn_path = sample(self.hr_agn_files.iloc[agn_idx].iloc[0], 1)[0]

        lr_background_path = None
        if self.lr_background > 0:
            lr_background_path = (
                self.lr_background_files[self.lr_exps[lr_exp]].sample(1).item()
            )

        hr_background_path = None
        if self.hr_background:
            hr_background_path = self.hr_background_files[self.hr_exp].sample(1).item()
            
            
        return [lr_img_path, lr_agn_path, lr_background_path, hr_img_path, hr_agn_path, hr_background_path]
            
                   
    def define_paths_constant_combinations(self, idx, lr_exp, base_name):
        
        # Assign unique index based on current index
        lr_idx = idx % len(self.lr_img_files.iloc[base_name].iloc[lr_exp])
        hr_idx = idx % len(self.hr_img_files.iloc[base_name].iloc[0])

        lr_img_path = self.lr_img_files.iloc[base_name].iloc[lr_exp][lr_idx]
        hr_img_path = self.hr_img_files.iloc[base_name].iloc[0][hr_idx]

        lr_agn_path = None  # Define them as None in the case that self.agn is False
        hr_agn_path = None

        if self.lr_agn > 0 or self.hr_agn:
            agn_idx = idx % self.base_agn_count

            if self.lr_agn > 0:
                lr_idx = idx % len(self.lr_agn_files.iloc[agn_idx].iloc[lr_exp])
                lr_agn_path = self.lr_agn_files.iloc[agn_idx].iloc[lr_exp][lr_idx]

            if self.hr_agn:
                hr_idx = idx % len(self.hr_agn_files.iloc[agn_idx].iloc[0])
                hr_agn_path = self.hr_agn_files.iloc[agn_idx].iloc[0][hr_idx]

        lr_background_path = None
        if self.lr_background > 0:
            lr_background_path = self.lr_background_files[self.lr_exps[lr_exp]].iloc[base_name]
            
        hr_background_path = None
        if self.hr_background:
            hr_background_path= self.hr_background_files[self.hr_exp].iloc[base_name].item()
            
            
        return [lr_img_path, lr_agn_path, lr_background_path, hr_img_path, hr_agn_path, hr_background_path]
    
   
    def load_lr_hr_xmm_sample(self, idx):
        lr_exp = idx % self.lr_exps.size
        base_name = idx % self.base_name_count
        
        if self.constant_img_combs:
            paths = self.define_paths_constant_combinations(idx, lr_exp, base_name)
        else:
            paths = self.define_paths_random_combinations(lr_exp, base_name)
        
        # Unpack for better legibility
        lr_img_path = paths[0]
        lr_agn_path = paths[1]
        lr_background_path = paths[2]
        hr_img_path = paths[3]
        hr_agn_path = paths[4]
        hr_background_path = paths[5]
        
        # Load and combine the selected files
        lr_img = self.load_and_combine_simulations(
            res_mult=self.lr_res_mult,
            img_path=lr_img_path,
            agn_path=lr_agn_path,
            background_path=lr_background_path,
            det_mask=self.lr_det_mask,
        )
        hr_img = self.load_and_combine_simulations(
            res_mult=self.hr_res_mult,
            img_path=hr_img_path,
            agn_path=hr_agn_path,
            background_path=hr_background_path,
            det_mask=self.hr_det_mask,
        )

        return lr_img, hr_img

    def __getitem__(self, idx):
        # Load a sample based on the given index
        lr_img_sample, hr_img_sample = self.load_lr_hr_xmm_sample(idx=idx)

        lr_img = (
            apply_transform(lr_img_sample["img"], self.transform)
            if self.transform
            else lr_img_sample["img"]
        )
        hr_img = (
            apply_transform(hr_img_sample["img"], self.transform)
            if self.transform
            else hr_img_sample["img"]
        )
        lr_img = self.normalize.normalize_lr_image(lr_img, idx = idx) if self.normalize else lr_img
        hr_img = self.normalize.normalize_hr_image(hr_img, idx = idx) if self.normalize else hr_img

        item = {
            "lr": lr_img,
            "hr": hr_img,
            "lr_exp": lr_img_sample["exp"] // 1000,
            "hr_exp": hr_img_sample["exp"] // 1000,
            "lr_img_file_name": lr_img_sample["file_name"],
            "idx": idx,
        }

        return item
