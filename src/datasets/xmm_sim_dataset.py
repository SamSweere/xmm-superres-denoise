import os
import random

import torch
from astropy.io import fits
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.utils import (
    get_fits_files,
    match_file_list,
    load_fits,
    reshape_img_to_res,
    group_same_sources,
)


class XmmSimDataset(Dataset):
    """XMM-Newton simulated dataset"""

    def __init__(
        self,
        dataset_name,
        datasets_dir,
        split,
        lr_res,
        hr_res,
        dataset_lr_res,
        mode,
        lr_exp,
        hr_exp,
        lr_agn,
        hr_agn,
        lr_background,
        hr_background,
        exp_channel,
        det_mask,
        check_files=False,
        transform=None,
        normalize=None,
        match_files=False,
    ):
        """
        Args:
            dataset_name (string): Name of the dataset
            datasets_dir (string): Directory of the datasets
            split (string): The split directory of the dataset, if 'full' take everything
            lr_res (int): The resolution of the final input images (low resolution)
            hr_res (int): The resolution of the final target images (high resolution)
            dataset_lr_res (int): The resolution the input images are transformed to to make them rectangular before any cropping
            mode (string): The mode, img or agn. If agn the model samples from agn as if it is an image
            lr_exp (int/list): Exposure of the low resolution images, in ks, if list it is the range of exposure times
            hr_exp (int): Exposure of the high resolution images, in ks
            lr_agn (bool): Add agns to the low resolution images
            hr_agn (bool): Add agns to the high resolution images
            lr_background (bool): Add background to low resolution
            hr_background (bool): Add background to high resolution
            det_mask (bool): Multiply the final image with the detectormask
            exp_channel (bool): Add the exposure channel
            check_files (bool) (optional): Check the file integrity of the chosen dataset
            transform (callable) (optional): Optional transform to be applied
            normalize (callable) (optional): Optional normalization to be applied
            match_files (boolean) (optional): Find the same files in both the lr and hr image files. Only needed when these do not completely contain the same sources. Can take a long time
        """

        self.transform = transform
        self.normalize = normalize
        self.dataset_name = dataset_name

        self.lr_res = lr_res
        self.hr_res = hr_res
        self.mode = mode
        self.exp_channel = exp_channel

        # Determine the low resolution exposure times, if it only one number that will be the exposure time
        if type(lr_exp) != list:
            self.lr_exps = [lr_exp]
        else:
            self.lr_exps = []
            stepsize = 10
            for i in range(lr_exp[0], lr_exp[1] + stepsize, stepsize):
                self.lr_exps.append(i)

        self.hr_exp = hr_exp
        self.lr_agn = lr_agn
        self.hr_agn = hr_agn
        self.det_mask = det_mask
        self.lr_background = lr_background
        self.hr_background = hr_background

        self.upscale = round(float(self.hr_res) / float(self.lr_res))
        self.lr_res_mult = 1
        self.hr_res_mult = self.upscale

        # Set the image dimensions
        self.dataset_lr_res = dataset_lr_res
        self.dataset_hr_res = self.dataset_lr_res * self.upscale

        self.dataset_dir = os.path.join(datasets_dir, dataset_name)

        if split != "full":
            self.dataset_dir = os.path.join(self.dataset_dir, split)

        # Get all the image directories
        # Note that if the mode is agn we consider them as the base images
        self.lr_img_dirs = []
        for lr_exp in self.lr_exps:
            self.lr_img_dirs.append(
                os.path.join(
                    self.dataset_dir,
                    str(lr_exp) + "ks",
                    self.mode,
                    str(self.lr_res_mult) + "x",
                )
            )
        self.hr_img_dir = os.path.join(
            self.dataset_dir,
            str(self.hr_exp) + "ks",
            self.mode,
            str(self.hr_res_mult) + "x",
        )
        # Get the fits files and file names
        self.lr_img_files = []
        for lr_img_dir in self.lr_img_dirs:
            lr_img_files, _ = get_fits_files(dataset_dir=lr_img_dir)
            self.lr_img_files.append(lr_img_files)
        self.hr_img_files, _ = get_fits_files(dataset_dir=self.hr_img_dir)

        if match_files:
            # Filter the files such that only files that have a match in both the resolution are present
            # The lr_img_files and hr_img_files will be a list of list containing all the files with the same base_img name
            self.lr_img_files, self.hr_img_files, self.base_img_files = match_file_list(
                self.lr_img_files, self.hr_img_files, split_key="_mult_"
            )
        else:
            # Group the same sources into the same folder and get base_img files
            (
                self.lr_img_files,
                self.hr_img_files,
                self.base_img_files,
            ) = group_same_sources(
                self.lr_img_files, self.hr_img_files, split_key="_mult_"
            )

        # Sort the base files such that the picking order is always the same
        self.base_img_files.sort()

        print(
            f"Found {len(self.base_img_files)} image pairs (lr and hr simulation matches)"
        )

        if self.lr_agn or self.hr_agn:
            self.lr_agn_dirs = []

            base_agn_dir_name = "agn"

            for lr_exp in self.lr_exps:
                self.lr_agn_dirs.append(
                    os.path.join(
                        self.dataset_dir,
                        str(lr_exp) + "ks",
                        base_agn_dir_name,
                        str(self.lr_res_mult) + "x",
                    )
                )
            self.hr_agn_dir = os.path.join(
                self.dataset_dir,
                str(self.hr_exp) + "ks",
                base_agn_dir_name,
                str(self.hr_res_mult) + "x",
            )

            self.lr_agn_files = []
            for lr_agn_dir in self.lr_agn_dirs:
                lr_agn_files, _ = get_fits_files(dataset_dir=lr_agn_dir)
                self.lr_agn_files.append(lr_agn_files)

            self.hr_agn_files, _ = get_fits_files(dataset_dir=self.hr_agn_dir)

            if match_files:
                # Filter the files such that only files that have a match in both the resolution are present
                (
                    self.lr_agn_files,
                    self.hr_agn_files,
                    self.base_agn_files,
                ) = match_file_list(
                    self.lr_agn_files, self.hr_agn_files, split_key="_mult_"
                )
            else:
                # Group the same agn sources into the same folder and get base_agn files
                (
                    self.lr_agn_files,
                    self.hr_agn_files,
                    self.base_agn_files,
                ) = group_same_sources(
                    self.lr_agn_files, self.hr_agn_files, split_key="_mult_"
                )

            # Sort the base files such that the picking order is always the same
            self.base_agn_files.sort()

            print(
                f"Found {len(self.base_agn_files)} agn image pairs (lr and hr simulation matches)"
            )

        if self.lr_background:
            self.lr_background_dirs = []
            for lr_exp in self.lr_exps:
                self.lr_background_dirs.append(
                    os.path.join(
                        self.dataset_dir,
                        str(lr_exp) + "ks",
                        "background",
                        str(self.lr_res_mult) + "x",
                    )
                )

            self.lr_background_files = []
            for lr_background_dir in self.lr_background_dirs:
                lr_background_files, _ = get_fits_files(dataset_dir=lr_background_dir)
                self.lr_background_files.append(lr_background_files)

            # Sort the base files such that the picking order is always the same
            self.lr_background_files.sort()

        if self.hr_background:
            self.hr_background_dir = os.path.join(
                self.dataset_dir,
                str(self.hr_exp) + "ks",
                "background",
                str(self.hr_res_mult) + "x",
            )

            self.hr_background_files, _ = get_fits_files(
                dataset_dir=self.hr_background_dir
            )

            # Sort the base files such that the picking order is always the same
            self.hr_background_files.sort()

        # Load the detector masks for the lr and hr resolutions
        if self.det_mask or self.exp_channel:
            # Since the det mask will be used on every image we load them into memory
            lr_det_mask_base_path = os.path.join(
                self.dataset_dir, "detector_mask", str(self.lr_res_mult) + "x"
            )
            lr_det_mask_path = os.path.join(
                lr_det_mask_base_path, os.listdir(lr_det_mask_base_path)[0]
            )
            lr_det_mask_hdu = fits.open(lr_det_mask_path)
            self.lr_det_mask = lr_det_mask_hdu[0].data.copy()
            lr_det_mask_hdu.close()

            hr_det_mask_base_path = os.path.join(
                self.dataset_dir, "detector_mask", str(self.hr_res_mult) + "x"
            )
            hr_det_mask_path = os.path.join(
                hr_det_mask_base_path, os.listdir(hr_det_mask_base_path)[0]
            )
            hr_det_mask_hdu = fits.open(hr_det_mask_path)
            self.hr_det_mask = hr_det_mask_hdu[0].data.copy()
            hr_det_mask_hdu.close()

        if check_files:
            # Check the file integrity
            self.check_dataset_correctness()

    def load_and_combine_simulations(
        self,
        res_mult,
        img_path,
        agn_path=None,
        background_path=None,
        det_mask=None,
        exp_channel=None,
    ):
        # print("Combining images with paths:")
        # print(img_path)
        # print(agn_path)
        # print(background_path)

        # Load the image data
        img = load_fits(img_path)

        if agn_path is not None:
            img["img"] += load_fits(agn_path)["img"]

        if background_path is not None:
            img["img"] += load_fits(background_path)["img"]

        if det_mask is not None:
            img["img"] *= det_mask  # Note the *=

        # The image has the shape (411, 403), we pad/crop this to (dataset_lr_res, dataset_lr_res)
        img["img"] = reshape_img_to_res(
            dataset_lr_res=self.dataset_lr_res, img=img["img"], res_mult=res_mult
        )

        if exp_channel is not None:
            # Add the exposure channel
            img["exp_channel"] = reshape_img_to_res(
                dataset_lr_res=self.dataset_lr_res, img=exp_channel, res_mult=res_mult
            )

        return img

    def __len__(self):
        return len(self.base_img_files)

    def load_lr_hr_xmm_sample(self, idx):
        # Get the the image index and random sample from the possibilities
        # First chose a random exposure index
        exposure_index = random.randint(0, len(self.lr_exps) - 1)
        lr_exp = self.lr_exps[exposure_index]

        lr_img_file = random.sample(self.lr_img_files[exposure_index][idx], 1)[0]
        hr_img_file = random.sample(self.hr_img_files[idx], 1)[0]
        lr_img_path = os.path.join(self.lr_img_dirs[exposure_index], lr_img_file)
        hr_img_path = os.path.join(self.hr_img_dir, hr_img_file)

        lr_agn_path = None  # Define them as None in the case that self.agn is False
        hr_agn_path = None
        if self.lr_agn or self.hr_agn:
            # Sample an agn
            # Take a random agn index
            agn_idx = random.randint(0, len(self.base_agn_files) - 1)

            if self.lr_agn:
                # Randomly take one from the different exposure times
                lr_agn_file = random.sample(
                    self.lr_agn_files[exposure_index][agn_idx], 1
                )[0]
                lr_agn_path = os.path.join(
                    self.lr_agn_dirs[exposure_index], lr_agn_file
                )

            if self.hr_agn:
                hr_agn_file = random.sample(self.hr_agn_files[agn_idx], 1)[0]
                hr_agn_path = os.path.join(self.hr_agn_dir, hr_agn_file)

        lr_background_path = None
        if self.lr_background:
            # Sample an low res background
            lr_background_file = random.sample(
                self.lr_background_files[exposure_index], 1
            )[0]
            lr_background_path = os.path.join(
                self.lr_background_dirs[exposure_index], lr_background_file
            )

        hr_background_path = None
        if self.hr_background:
            # Sample an high res background
            hr_background_file = random.sample(self.hr_background_files, 1)[0]
            hr_background_path = os.path.join(
                self.hr_background_dir, hr_background_file
            )

        exp_channel = None
        if self.exp_channel:
            # Add the exposure channel from the detmask
            # Since 100ks is max we put this to be 1, and 0ks 0. Thus the exposures in ks divided by 100
            exp_channel = self.lr_det_mask * lr_exp / 100

        # Load and combine the selected files
        lr_img = self.load_and_combine_simulations(
            res_mult=self.lr_res_mult,
            img_path=lr_img_path,
            agn_path=lr_agn_path,
            background_path=lr_background_path,
            det_mask=self.lr_det_mask,
            exp_channel=exp_channel,
        )
        hr_img = self.load_and_combine_simulations(
            res_mult=self.hr_res_mult,
            img_path=hr_img_path,
            agn_path=hr_agn_path,
            background_path=hr_background_path,
            det_mask=self.hr_det_mask,
        )

        return lr_img, lr_exp, hr_img

    def __getitem__(self, idx):
        # Load a sample based on the given index
        try:
            lr_img_sample, lr_exp, hr_img_sample = self.load_lr_hr_xmm_sample(idx=idx)
        except Exception as e:
            print("ERROR, failed to load lr hr xmm sample with error: ", e)
            print("Trying again...")
            lr_img_sample, lr_exp, hr_img_sample = self.load_lr_hr_xmm_sample(idx=idx)

        hr_exp = hr_img_sample["exp"] / 1000  # In ks

        lr_img = lr_img_sample["img"]
        hr_img = hr_img_sample["img"]

        if self.exp_channel:
            exp_channel = lr_img_sample["exp_channel"]

        # Apply the transformations
        if self.transform:
            for t in self.transform:
                if self.exp_channel:
                    lr_img, exp_channel, hr_img = t([lr_img, exp_channel, hr_img])
                else:
                    lr_img, hr_img = t([lr_img, hr_img])

        # lr_gt = lr_img
        # hr_gt = hr_img

        # Apply the normalization
        if self.normalize:
            lr_img, hr_img = self.normalize([lr_img, hr_img])

        # Torch needs the data to have dimensions [1, x, x]
        lr_img = torch.unsqueeze(lr_img, axis=0)
        hr_img = torch.unsqueeze(hr_img, axis=0)

        if self.exp_channel:
            # Add the exp channel to the lr_img
            exp_channel = torch.unsqueeze(exp_channel, axis=0)
            lr_img = torch.cat((lr_img, exp_channel), 0)

        sample = {
            "lr": lr_img,
            "hr": hr_img,
            "lr_exp": lr_exp,
            "hr_exp": hr_exp,
            "lr_img_file_name": lr_img_sample["file_name"],
            "tng_set": lr_img_sample["file_name"].split("_")[0],
        }  # , 'lr_gt': lr_gt, 'hr_gt': hr_gt}

        return sample

    def check_img_corr(self, img_path, shape):
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
        print("Checking lr img files:")
        for exp_index in range(len(self.lr_exps)):
            print(f"Checking exp {self.lr_exps[exp_index]}")
            for big_img_path in tqdm(self.lr_img_files[exp_index]):
                for img_path in big_img_path:
                    self.check_img_corr(
                        os.path.join(self.lr_img_dirs[exp_index], img_path),
                        shape=(411, 403),
                    )

        print("Checking hr img files:")
        for big_img_path in tqdm(self.hr_img_files):
            for img_path in big_img_path:
                self.check_img_corr(
                    os.path.join(self.hr_img_dir, img_path),
                    shape=(411 * self.hr_res_mult, 403 * self.hr_res_mult),
                )
        if self.lr_agn or self.hr_agn:
            print("Checking lr agn files:")
            for exp_index in range(len(self.lr_exps)):
                print(f"Checking exp {self.lr_exps[exp_index]}")
                for big_img_path in tqdm(self.lr_agn_files[exp_index]):
                    for img_path in big_img_path:
                        self.check_img_corr(
                            os.path.join(self.lr_agn_dirs[exp_index], img_path),
                            shape=(411, 403),
                        )

            print("Checking hr agn files:")
            for big_img_path in tqdm(self.hr_agn_files):
                for img_path in big_img_path:
                    self.check_img_corr(
                        os.path.join(self.hr_agn_dir, img_path),
                        shape=(411 * self.hr_res_mult, 403 * self.hr_res_mult),
                    )

        if self.lr_background:
            print("Checking lr background files:")
            for exp_index in range(len(self.lr_exps)):
                print(f"Checking exp {self.lr_exps[exp_index]}")
                for img_path in tqdm(self.lr_background_files[exp_index]):
                    self.check_img_corr(
                        os.path.join(self.lr_background_dirs[exp_index], img_path),
                        shape=(411, 403),
                    )

        if self.hr_background:
            print("Checking hr background files:")
            for img_path in tqdm(self.hr_background_files):
                self.check_img_corr(
                    os.path.join(self.hr_background_dir, img_path),
                    shape=(411 * self.hr_res_mult, 403 * self.hr_res_mult),
                )

        print("All files are within specifications!")
