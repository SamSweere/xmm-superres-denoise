import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from loguru import logger
from torch.utils.data import Subset
from tqdm import tqdm


def save_splits(paths: List[Path], splits: List[Subset]):
    for path, split in zip(paths, splits):
        indices = np.asarray(split.indices)
        logger.info(f"\tSplit {path} contains {len(indices)} images")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w+b") as f:
            pickle.dump(indices, f)


def find_dir(parent: Path, pattern: str) -> Path:
    glob_res = parent.glob(pattern)
    dir_path = None
    zip_file = None
    for res in glob_res:
        if res.is_dir() and res.name.endswith(pattern.replace("*", "")[-1]):
            dir_path = res
            break
        if res.name.endswith(".zip"):
            zip_file = res
    if dir_path is None and zip_file is not None:
        logger.info(f"Extracting {zip_file} to {parent}...")
        with ZipFile(zip_file, "r") as zip_f:
            zip_f.extractall(parent)
        dir_path = parent / pattern.replace(
            "*", ""
        )  # Remove the asterisk used to find the corresponding .zip file.
        # zip_file.unlink()  # Use the deletion with care!

    if dir_path is None:
        raise NotADirectoryError(
            f"Could not find any directory in {parent} matching {pattern}"
        )

    return dir_path


def find_img_dirs(
    parent: Path, exps: list[int] | int, res_mult_dir: str
) -> Dict[int, list[Path]]:
    if isinstance(exps, int):
        exps = [exps]

    res: Dict[int, list[Path]] = {}
    for exp in exps:
        glob_pattern = f"{exp}ks/**/{res_mult_dir}" if res_mult_dir else f"{exp}ks/"
        exp_dirs = list(parent.glob(glob_pattern))
        assert len(exp_dirs) > 0
        res[exp] = exp_dirs
    return res


def find_img_files(exp_dirs_dict: Dict[int, list[Path]]) -> Dict[int, List[Path]]:
    res: Dict[int, List[Path]] = {}
    for exp, img_dirs in exp_dirs_dict.items():
        files = []
        for img_dir in img_dirs:
            files.extend(get_fits_files(dataset_dir=img_dir))
        res[exp] = files
    return res


def check_img_files(
    img_files: pd.DataFrame, shape: Tuple[int, int, int], msg: str = None
):
    for base_name, files in tqdm(img_files.iterrows(), desc=msg):
        for exp, path_list in tqdm(files.items(), leave=False):
            for path in path_list:
                check_img_corr(path, shape=shape)


def check_img_corr(img_path, shape):
    img = load_fits(img_path)

    max_val = 100000
    min_val = 0

    if img.shape != shape:
        raise ValueError(
            f"ERROR {img_path} wrong shape ({img.shape}, while desired shape is {shape}"
        )

    if torch.any(torch.isnan(img)):
        raise ValueError(f"ERROR {img_path} contains a NAN")

    if torch.any(img > max_val):
        raise ValueError(f"ERROR {img_path} contains a value bigger then {max_val}")

    if torch.any(img < min_val):
        raise ValueError(f"ERROR {img_path} contains a value smaller then {min_val}")


def load_fits(fits_path: Path) -> torch.Tensor:
    # Extract the image data from the fits file and convert to float
    # (these images will be in int but since we will work with floats in pytorch we convert them to float)
    img = fits.getdata(fits_path, "PRIMARY")

    img = torch.from_numpy(img.astype(np.float32)).unsqueeze(dim=0)

    return img


def apply_transform(
    img: Union[torch.Tensor, List[torch.Tensor]], transforms: List[Callable]
):
    if type(img) == list:
        for i in range(len(img)):
            for t in transforms:
                img[i] = t(img[i])
    else:
        for t in transforms:
            img = t(img)

    return img


def load_det_mask(res_mult: int):
    data = fits.getdata(
        Path("res") / "detector_mask" / f"pn_mask_500_2000_detxy_{res_mult}x.ds", 0
    )

    return data.astype(np.float32)


def reshape_img_to_res(res: int, img: torch.Tensor) -> torch.Tensor:
    """
    Reshape the given image into (res, res)

    :param res: Resolution to be achieved
    :param img: Image to pad/crop
    :return: Padded/cropped image
    """
    y_diff = res - img.shape[1]
    y_top_pad = int(np.floor(y_diff / 2.0))
    y_bottom_pad = y_diff - y_top_pad

    x_diff = res - img.shape[2]
    x_left_pad = int(np.floor(x_diff / 2.0))
    x_right_pad = x_diff - x_left_pad

    img = torch.nn.functional.pad(
        img,
        (x_left_pad, x_right_pad, y_top_pad, y_bottom_pad, 0, 0),
        mode="constant",
        value=0,
    )

    return img


def get_fits_files(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist!")

    res: List[Path] = list(dataset_dir.glob("*.fits"))
    res.extend(list(dataset_dir.glob("*.fits.gz")))
    logger.info(f"\tDetected {len(res)} fits files in {dataset_dir}")

    return sorted(res)


def get_base_names(
    img_dict: Union[Dict[int, List[Path]], List[Path]], split_key: str
) -> Set[str]:
    if isinstance(img_dict, dict):
        base_names = []
        for exp, file_names in img_dict.items():
            base_names.append(
                set([file_name.name.split(split_key)[0] for file_name in file_names])
            )
        # Since we can't be sure that every base_name is represented for every exposure, we have to make sure that no
        # exposure has an empty list of files
        base_names = set.intersection(*base_names)
    else:
        base_names = set()
        for file_name in img_dict:
            base_name = file_name.name.split(split_key)[0]
            base_names.add(base_name)

    return base_names


def filter_img_dict(
    img_dict: Dict[int, List[Path]], base_names: set, split_key: str
) -> Dict[int, Dict[str, List[str]]]:
    filtered_img_dict = {
        exp: {base_name: [] for base_name in base_names} for exp in img_dict.keys()
    }

    for exp, file_names in img_dict.items():
        for file_name in file_names:
            base_name = file_name.name.split(split_key)[0]
            if base_name in base_names:
                filtered_img_dict[exp][base_name].append(file_name)

    return filtered_img_dict


def match_file_list(
    lr_dict: Dict[int, List[Path]],
    hr_dict: Optional[Dict[int, List[Path]]],
    split_key: str,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], int]:
    lr_base_names = get_base_names(lr_dict, split_key)
    hr_base_names = (
        get_base_names(hr_dict, split_key) if hr_dict is not None else lr_base_names
    )
    base_names = lr_base_names & hr_base_names

    if not base_names:
        raise ValueError(
            f'No base_names could be found in both given dictionaries with split_key "{split_key}"!'
        )

    lr_dict = filter_img_dict(lr_dict, base_names, split_key)
    hr_dict = (
        filter_img_dict(hr_dict, base_names, split_key) if hr_dict is not None else None
    )

    lr_df = pd.DataFrame.from_dict(lr_dict).sort_index()
    hr_df = pd.DataFrame.from_dict(hr_dict).sort_index() if hr_dict else None

    return lr_df, hr_df, len(base_names)
