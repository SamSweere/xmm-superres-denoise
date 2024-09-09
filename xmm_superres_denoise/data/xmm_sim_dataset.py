from pathlib import Path
from random import randint, sample
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split
from config.config import DatasetCfg, DatasetType
from data.utils import (
    apply_transform,
    check_img_files,
    find_img_dirs,
    find_img_files,
    load_fits,
    match_file_list,
    reshape_img_to_res,
    save_splits,
)
from loguru import logger
from transforms import Normalize


def _load_and_combine_simulations(
        res: int,
        img_path: Path,
        agn_path: Path = None,
        background_path: Path = None,
        det_mask: Path = None
):
    # Load the image data
    img = load_fits(img_path)

    if agn_path:
        img += load_fits(agn_path)

    if background_path:
        img += load_fits(background_path)

    if det_mask is not None:
        img *= load_fits(det_mask)  # Note the *=

    img = reshape_img_to_res(res=res, img=img)

    return img


class XmmSimDataset(Dataset):
    """XMM-Newton simulated dataset"""

    def __init__(
        self,
        config: DatasetCfg,
        comb_hr_img: bool,
        transform: List[Callable] = None,
        normalize: Optional[Normalize] = None,
    ):
        """
        Args:
            transform (callable) (optional): Optional transform to be applied
            normalize (callable) (optional): Optional normalization to be applied
        """
        self.config = config
        self.transform = transform if transform else []
        self.normalize = normalize

        # TODO What about type DISPLAY?
        split_key = "_mult_" if self.config.type == DatasetType.SIM else "_image_split_"

        # Get all the image directories
        # Note that if the mode is agn we consider them as the base images
        # --- LR images --- #
        pattern = ""
        if self.config.type is DatasetType.SIM:
            pattern = f"*/*/1x"

        lr_img_dirs = find_img_dirs(self.config.img_dir, self.config.lr.exps, pattern)
        lr_img_files = find_img_files(lr_img_dirs)
        del pattern

        # --- HR images --- #
        pattern = ""
        if self.config.type is DatasetType.REAL and self.config.hr.exp:
            pattern = ""
        elif self.config.type is DatasetType.SIM and comb_hr_img:
            pattern = f"*/*/{self.config.res_mult}x_comb"
        elif self.config.type is DatasetType.SIM:
            pattern = f"*/*/{self.config.res_mult}x"

        if self.config.type is DatasetType.REAL and not self.config.hr.exp:
            hr_img_files = None
        else:
            hr_img_dirs = find_img_dirs(self.config.img_dir, self.config.hr.exp, pattern)
            hr_img_files = find_img_files(hr_img_dirs)
        del pattern

        self.lr_img_files, self.hr_img_files, self.base_name_count = match_file_list(
            lr_img_files, hr_img_files, split_key
        )

        if self.config.check_files:
            check_img_files(self.lr_img_files, (411, 403), "Checking lr_img_files...")
            check_img_files(
                self.hr_img_files,
                (411 * self.config.res_mult, 403 * self.config.res_mult),
                "Checking hr_img_files...",
            )

        self.dataset_size = self.base_name_count * len(self.config.lr.exps)
        msg1 = f"Overall dataset size: img_count * lr_exps_count"
        msg2 = f"{self.base_name_count} * {len(self.config.lr.exps)}"

        # --- AGN images --- #
        self.lr_agn_files = self.hr_agn_files = self.base_agn_count = None
        if self.config.agn > 0 and not self.config.type is DatasetType.REAL:
            msg1 = f"{msg1} * lr_agn_count"
            msg2 = f"{msg2} * {self.config.agn}"
            self.dataset_size = self.dataset_size * self.config.agn
            lr_agn_dirs = find_img_dirs(self.config.agn_dir, self.config.lr.exps, f"1x")
            lr_agn_files = find_img_files(lr_agn_dirs)

            if comb_hr_img:
                pattern = f"{self.config.res_mult}x_comb"
            else:
                pattern = f"{self.config.res_mult}x"

            hr_agn_dirs = find_img_dirs(self.config.agn_dir, self.config.hr.exp, pattern)
            hr_agn_files = find_img_files(hr_agn_dirs)

            self.lr_agn_files, self.hr_agn_files, self.base_agn_count = match_file_list(
                lr_agn_files, hr_agn_files, split_key
            )
            logger.info(f"\tFound {self.base_agn_count} agn image pairs (lr and hr simulation matches)")

            if self.config.check_files:
                check_img_files(self.lr_agn_files, (411, 403), "Checking lr_agn_files...")
                check_img_files(self.hr_agn_files, (411 * self.config.res_mult, 403 * self.config.res_mult),
                                "Checking hr_agn_files...", )

        # --- BKG images --- #
        self.lr_bkg_files = None
        if self.config.lr.bkg > 0 and not self.config.type is DatasetType.REAL:
            msg1 = f"{msg1} * lr_background_count"
            msg2 = f"{msg2} * {self.config.lr.bkg}"
            self.dataset_size = self.dataset_size * self.config.lr.bkg
            lr_background_dirs = find_img_dirs(
                self.config.bkg_dir, self.config.lr.exps, f"1x"
            )
            lr_background_files = find_img_files(lr_background_dirs)
            amt = min([len(file_list) for file_list in lr_background_files.values()])
            self.lr_bkg_files = {}
            for exp, files in lr_background_files.items():
                self.lr_bkg_files[exp] = sample(files, amt)
            self.lr_bkg_files = pd.DataFrame.from_dict(self.lr_bkg_files)

            if self.config.check_files:
                check_img_files(self.lr_bkg_files, (411, 403), "Checking lr_background_files...", )

        logger.info(f"\t{msg1} = dataset_size")
        logger.info(f"\t\t{msg2} = {self.dataset_size}")

    def __len__(self):
        return self.dataset_size

    def load_sample(self, idx):
        lr_exp = idx % len(self.config.lr.exps)
        base_name = idx % self.base_name_count

        lr_img_path = sample(self.lr_img_files.iloc[base_name].iloc[lr_exp], 1)[0]

        hr_img_path = None
        if self.hr_img_files:
            hr_img_path = sample(self.hr_img_files.iloc[base_name].iloc[0], 1)[0]

        lr_agn_path = hr_agn_path = None
        if self.lr_agn_files:
            agn_idx = randint(0, self.base_agn_count - 1)

            lr_agn_path = sample(self.lr_agn_files.iloc[agn_idx].iloc[lr_exp], 1)[0]
            hr_agn_path = sample(self.hr_agn_files.iloc[agn_idx].iloc[0], 1)[0]

        lr_background_path = None
        if self.lr_bkg_files:
            lr_background_path = (
                self.lr_bkg_files[self.config.lr.exps[lr_exp]].sample(1).item()
            )

        # Load and combine the selected files
        lr_img = _load_and_combine_simulations(
            res=self.config.lr.res,
            img_path=lr_img_path,
            agn_path=lr_agn_path,
            background_path=lr_background_path,
            det_mask=None,
        )

        hr_img = None
        if hr_img_path is not None:
            hr_img = _load_and_combine_simulations(
                res=self.config.hr.res,
                img_path=hr_img_path,
                agn_path=hr_agn_path,
                background_path=None,
                det_mask=None,
            )

        return lr_img, hr_img

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        # Load a sample based on the given index
        lr_img, hr_img = self.load_sample(idx=idx)

        lr_img = (
            apply_transform(lr_img, self.transform)
            if self.transform
            else lr_img
        )
        hr_img = (
            apply_transform(hr_img, self.transform)
            if self.transform
            else hr_img
        )

        lr_img = self.normalize.normalize_lr_image(lr_img) if self.normalize else lr_img
        hr_img = self.normalize.normalize_hr_image(hr_img) if self.normalize else hr_img

        return lr_img, hr_img

    def prepare(self, subset_str: str):
        splits = ["train", "val", "test"]
        paths = [Path(subset_str.format(split_name)) for split_name in splits]
        exists = np.all([path.exists() for path in paths])
        if not exists:
            logger.info(f"Creating splits for {self.config.directory} with {self.base_name_count} base_names...")
            train, val, test = random_split(
                range(self.base_name_count), [0.8, 0.1, 0.1]
            )
            save_splits(paths, [train, val, test])