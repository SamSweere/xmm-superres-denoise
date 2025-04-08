import pickle
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
import torch
from astropy.io import fits
from loguru import logger
from torch.utils.data import Subset


def save_splits(paths: List[Path], splits: List[Subset]):
    for path, split in zip(paths, splits):
        indices = np.asarray(split.indices)
        logger.info(f"\tSplit {path} contains {len(indices)} images")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w+b") as f:
            pickle.dump(indices, f)


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
