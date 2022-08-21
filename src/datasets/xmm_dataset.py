import os
import random

import torch
from astropy.io import fits
from torch.utils.data import Dataset

from src.datasets.utils import (
    get_fits_files,
    reshape_img_to_res,
    load_fits,
    group_same_sources,
)


class XmmDataset(Dataset):
    """XMM-Newton dataset"""

    def __init__(
        self,
        dataset_name,
        datasets_dir,
        split,
        dataset_lr_res,
        lr_exp,
        hr_exp,
        det_mask,
        exp_channel,
        include_hr,
        check_files=False,
        transform=None,
        normalize=None,
    ):
        """
        Args:
            dataset_name (string): Name of the dataset
            datasets_dir (string): Directory of the datasets
            split (string): The split directory of the dataset, if 'full' take everything
            dataset_lr_res (int): The resolution the input images are transformed to to make them rectangular before any cropping

            lr_exp (int/list): Exposure of the low resolution images, in ks, if list it is the range of exposure times
            hr_exp (int): Exposure of the high resolution images, in ks
            det_mask (bool): Multiply the final image with the detectormask
            exp_channel (bool): Add the exposure channel
            include_hr (bool): Only include samples where both lr and hr exists
            check_files (bool) (optional): Check the file integrity of the chosen dataset
            transform (callable) (optional): Optional transform to be applied
            normalize (callable) (optional): Optional normalization to be applied
        """

        self.transform = transform
        self.normalize = normalize

        self.dataset_name = dataset_name

        self.det_mask = det_mask
        self.exp_channel = exp_channel
        self.dataset_lr_res = dataset_lr_res

        # Determine the low resolution exposure times, if it only one number that will be the exposure time
        if type(lr_exp) != list:
            self.lr_exps = [lr_exp]
        else:
            self.lr_exps = []
            stepsize = 10
            for i in range(lr_exp[0], lr_exp[1] + stepsize, stepsize):
                self.lr_exps.append(i)
        self.hr_exp = hr_exp
        self.lr_res_mult = 1
        self.include_hr = include_hr

        self.dataset_dir = os.path.join(datasets_dir, dataset_name)

        if split != "full":
            self.dataset_dir = os.path.join(self.dataset_dir, split)

        # Get all the image directories
        # Note that if the mode is agn we consider them as the base images
        self.lr_img_dirs = []
        for lr_exp in self.lr_exps:
            self.lr_img_dirs.append(os.path.join(self.dataset_dir, str(lr_exp) + "ks"))

        # Get the fits files and file names
        self.lr_img_files = []
        for lr_img_dir in self.lr_img_dirs:
            lr_img_files, _ = get_fits_files(dataset_dir=lr_img_dir)
            self.lr_img_files.append(lr_img_files)

        # Group the same sources into the same folder and get base_img files
        # Since if we use multiple exposures the selection will be limited to the highest input exposure
        # Therefore we match the full file list with the highest input exposure

        if self.include_hr:
            self.hr_img_dir = os.path.join(self.dataset_dir, str(self.hr_exp) + "ks")
            self.hr_img_files, _ = get_fits_files(dataset_dir=self.hr_img_dir)
            (
                self.lr_img_files,
                self.hr_img_files,
                self.base_img_files,
            ) = group_same_sources(
                self.lr_img_files, self.hr_img_files, split_key="_image_split_"
            )
        else:
            self.lr_img_files, _, self.base_img_files = group_same_sources(
                self.lr_img_files, self.lr_img_files[-1], split_key="_image_split_"
            )

        print(f"Found {len(self.base_img_files)} fits images in {self.dataset_dir}")

        # Sort the base files such that the picking order is always the same
        self.base_img_files.sort()

        # Load the detector masks for the lr and hr resolutions
        if self.det_mask or self.exp_channel:
            # Since the det mask will be used on every image we load them into memory
            lr_det_mask_base_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "detector_mask",
                str(self.lr_res_mult) + "x",
            )
            lr_det_mask_path = os.path.join(
                lr_det_mask_base_path, os.listdir(lr_det_mask_base_path)[0]
            )
            lr_det_mask_hdu = fits.open(lr_det_mask_path)
            self.lr_det_mask = lr_det_mask_hdu[0].data.copy()
            lr_det_mask_hdu.close()

            if self.include_hr:
                self.hr_det_mask = (
                    self.lr_det_mask
                )  # This is on the same resolution, thus the same detmask

            if check_files:
                # Check the file integrity
                self.check_dataset_correctness()

    def check_img_corr(self, img_path, shape):
        import numpy as np

        try:
            img = load_fits(img_path)["img"]

            max_val = 100000
            min_val = 0

            if img.shape != shape:
                raise ValueError(
                    f"ERROR {img_path} wrong shape ({img.shape}, while desired shape is {shape}"
                )

            if np.isnan(np.sum(img)):
                raise ValueError(f"ERROR {img_path} contains a NAN")

            if np.max(img) > max_val:
                raise ValueError(
                    f"ERROR {img_path} contains a value bigger then {max_val} ({np.max(img)})"
                )

            if np.min(img) < min_val:
                raise ValueError(
                    f"ERROR {img_path} contains a value smaller then {min_val} ({np.min(img)})"
                )
        except Exception as e:
            print("ERROR: ", e)
            print(img_path)

    def check_dataset_correctness(self):
        from tqdm import tqdm

        print("Checking lr img files:")
        for exp_index in range(len(self.lr_exps)):
            print(f"Checking exp {self.lr_exps[exp_index]}")
            for big_img_path in tqdm(self.lr_img_files[exp_index]):
                for img_path in big_img_path:
                    self.check_img_corr(
                        os.path.join(self.lr_img_dirs[exp_index], img_path),
                        shape=(411, 403),
                    )

        if self.include_hr:
            print("Checking hr img files:")
            for big_img_path in tqdm(self.hr_img_files):
                for img_path in big_img_path:
                    self.check_img_corr(
                        os.path.join(self.hr_img_dir, img_path), shape=(411, 403)
                    )

        print("All files are within specifications!")

    def __len__(self):
        return len(self.base_img_files)

    def load_xmm_sample(self, idx):
        # Get the the image index and random sample from the possibilities
        # First chose a random exposure index
        exposure_index = random.randint(0, len(self.lr_exps) - 1)
        lr_exp = self.lr_exps[exposure_index]

        lr_img_file = random.sample(self.lr_img_files[exposure_index][idx], 1)[0]
        lr_img_path = os.path.join(self.lr_img_dirs[exposure_index], lr_img_file)

        if self.include_hr:
            hr_img_file = random.sample(self.hr_img_files[idx], 1)[0]
            hr_img_path = os.path.join(self.hr_img_dir, hr_img_file)

        exp_channel = None
        if self.exp_channel:
            # Add the exposure channel from the detmask
            # Since 100ks is max we put this to be 1, and 0ks 0. Thus the exposures in ks divided by 100
            exp_channel = self.lr_det_mask * lr_exp / 100

        # Load the image data
        lr_img = load_fits(lr_img_path)

        hr_img = None
        if self.include_hr:
            hr_img = load_fits(hr_img_path)

        if self.lr_det_mask is not None:
            lr_img["img"] *= self.lr_det_mask  # Note the *=

            if self.include_hr:
                hr_img["img"] *= self.hr_det_mask

        # Load and combine the selected files
        # The image has the shape (411, 403), we pad/crop this to (dataset_lr_res, dataset_lr_res)
        lr_img["img"] = reshape_img_to_res(
            dataset_lr_res=self.dataset_lr_res, img=lr_img["img"], res_mult=1
        )

        if self.include_hr:
            hr_img["img"] = reshape_img_to_res(
                dataset_lr_res=self.dataset_lr_res, img=hr_img["img"], res_mult=1
            )

        if exp_channel is not None:
            # Add the exposure channel
            lr_img["exp_channel"] = reshape_img_to_res(
                dataset_lr_res=self.dataset_lr_res, img=exp_channel, res_mult=1
            )

        return lr_img, lr_exp, hr_img

    def __getitem__(self, idx):
        # Load a sample based on the given index
        lr_img_sample, lr_exp, hr_img_sample = self.load_xmm_sample(idx=idx)

        lr_img = lr_img_sample["img"]

        hr_img = None
        hr_exp = None
        if self.include_hr:
            hr_img = hr_img_sample["img"]

            hr_exp = hr_img_sample["exp"] / 1000  # In ks

        # Make a list to save all the images is, this removes the need for a lot of if statements
        images = [lr_img]

        if self.exp_channel:
            exp_channel = lr_img_sample["exp_channel"]
            images.append(exp_channel)

        if self.include_hr:
            images.append(hr_img)

        # Apply the transformations
        if self.transform:
            for t in self.transform:
                images = t(images)

        lr_img = images[0]
        if self.exp_channel:
            exp_channel = images[1]

        if self.include_hr:
            hr_img = images[2]

        # Apply the normalization
        if self.normalize:
            lr_img = self.normalize.normalize_lr_image(lr_img)

            if self.include_hr:
                hr_img = self.normalize.normalize_hr_image(hr_img)

        # Torch needs the data to have dimensions [1, x, x]
        lr_img = torch.unsqueeze(lr_img, axis=0)

        if self.include_hr:
            hr_img = torch.unsqueeze(hr_img, axis=0)

        if self.exp_channel:
            # Add the exp channel to the lr_img
            exp_channel = torch.unsqueeze(exp_channel, axis=0)
            lr_img = torch.cat((lr_img, exp_channel), 0)

        sample = {
            "lr": lr_img,
            "lr_exp": lr_img_sample["exp"],
            "lr_img_file_name": lr_img_sample["file_name"],
            "lr_header": lr_img_sample["header"],
        }

        if self.include_hr:
            sample["hr"] = hr_img
            sample["hr_exp"] = hr_img_sample["exp"]
            sample["hr_header"] = hr_img_sample["header"]
            sample["hr_img_file_name"] = hr_img_sample["file_name"]

        return sample
