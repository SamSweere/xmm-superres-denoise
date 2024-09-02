import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Subset


def save_splits(paths: List[Path], splits: List[Subset]):
    for path, split in zip(paths, splits):
        indices = np.asarray(split.indices)
        rank_zero_info(f"\tSplit {path} contains {len(indices)} images")
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
        rank_zero_info(f"Extracting {zip_file} to {parent}...")
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


def find_img_dirs(parent: Path, exps: np.ndarray, pattern: str = "") -> Dict[int, Path]:
    res: Dict[int, Path] = {exp: parent / f"{exp}ks" if (parent / f"{exp}ks").is_dir() else None for exp in exps}
    for exp in exps:
        tmp = parent / f"{exp}ks"
        assert tmp.is_dir()
        res[exp] = tmp
    return res


def find_img_files(exp_dirs_dict: Dict[int, Path]) -> Dict[int, List[Path]]:
    res: Dict[int, List[Path]] = {}
    for exp, img_dir in exp_dirs_dict.items():
        res[exp] = get_fits_files(dataset_dir=img_dir)
    return res


def check_img_files(img_files: pd.DataFrame, shape: Tuple[int, int], msg: str = ""):
    if msg:
        rank_zero_info(f"\t{msg}")
    for base_name, files in img_files.iterrows():
        for exp, path_list in files.items():
            for path in path_list:
                check_img_corr(path, shape=shape)


def check_img_corr(img_path, shape):
    img = load_fits(img_path)["img"]

    max_val = 100000
    min_val = 0

    if img.shape != shape:
        raise ValueError(
            f"ERROR {img_path} wrong shape ({img.shape}, while desired shape is {shape}"
        )

    if np.any(np.isnan(img)):
        raise ValueError(f"ERROR {img_path} contains a NAN")

    if np.any(img > max_val):
        raise ValueError(f"ERROR {img_path} contains a value bigger then {max_val}")

    if np.any(img < min_val):
        raise ValueError(f"ERROR {img_path} contains a value smaller then {min_val}")


def load_fits(fits_path: Path) -> Dict:
    try:
        with fits.open(fits_path) as hdu:
            # Extract the image data from the fits file and convert to float
            # (these images will be in int but since we will work with floats in pytorch we convert them to float)
            img: np.ndarray = hdu["PRIMARY"].data.astype(np.float32)
            exposure = hdu["PRIMARY"].header["EXPOSURE"]
            header = dict(hdu["PRIMARY"].header)

        # The `HISTORY`, `COMMENT` and 'DPSCORRF' key are causing problems
        header.pop("HISTORY", None)
        header.pop("COMMENT", None)
        header.pop("DPSCORRF", None)
        header.pop("ODSCHAIN", None)
        header.pop("SRCPOS", None)

        # Devide the image by the exposure time to get a counts/sec image
        img = img / exposure
        img = img.astype(np.float32)

        return {
            "img": img,
            "exp": exposure,
            "file_name": fits_path.name,
            "header": header,
        }
    except Exception as e:
        raise IOError(f"Failed to load FITS file {fits_path} with error:", e)


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
    with fits.open(
        Path("res") / "detector_mask" / f"pn_mask_500_2000_detxy_{res_mult}x.ds"
    ) as hdu:
        return hdu[0].data.astype(np.float32)


def reshape_img_to_res(dataset_lr_res, img, res_mult):
    # The image has the shape (411, 403), we pad/crop this to (dataset_lr_res, dataset_lr_res)
    y_diff = dataset_lr_res * res_mult - img.shape[0]
    y_top_pad = int(np.floor(y_diff / 2.0))
    y_bottom_pad = y_diff - y_top_pad

    x_diff = dataset_lr_res * res_mult - img.shape[1]
    x_left_pad = int(np.floor(x_diff / 2.0))
    x_right_pad = x_diff - x_left_pad

    if y_diff >= 0:
        # Pad the image in the y direction
        img = np.pad(
            img, ((y_top_pad, y_bottom_pad), (0, 0)), "constant", constant_values=0.0
        )
    else:
        # Crop the image in the y direction
        img = img[abs(y_top_pad) : img.shape[0] - abs(y_bottom_pad)]

    if x_diff >= 0:
        # Pad the image in the x direction
        img = np.pad(
            img, ((0, 0), (x_left_pad, x_right_pad)), "constant", constant_values=0.0
        )
    else:
        # Crop the image in the x direction
        img = img[:, abs(x_left_pad) : img.shape[1] - abs(x_right_pad)]

    return img


def get_fits_files(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist!")

    res: List[Path] = list(dataset_dir.glob("*.fits"))
    res.extend(list(dataset_dir.glob("*.fits.gz")))
    rank_zero_info(f"\tDetected {len(res)} fits files in {dataset_dir}")

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
