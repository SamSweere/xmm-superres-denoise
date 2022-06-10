import os
from astropy.io import fits
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import torch

from transforms.downsamplesum import DownsampleSum
from transforms.normalize import Normalize
from transforms.crop import Crop
from transforms.randomflip import RandomFlip


class TngDataset(Dataset):
    """Illustrius TNG X ray simulated images dataset"""

    def __init__(self, dataset_name, datasets_dir, cache_dir, dataset_lr_res, dataset_hr_res, lr_res, hr_res, data_scaling=None,
                 transform=None):
        """
        Args:
            dataset_name (string): Name of the dataset
            datasets_dir (string): Directory of the datasets
            cache_dir (string): Directory to save the processed files.
            dataset_lr_res (int): The resolution of the dataset images used as input (low resolution)
            dataset_hr_res (int): The resolution of the dataset images used as output (high resolution)
            lr_res (int): The resolution of the final input images (low resolution)
            hr_res (int): The resolution of the final target images (high resolution)
            data_scaling (string) (optional): The normalization method, options: linear, sqrt.
                                            When None no normalization will be done
            transform (callable) (optional): Optional transform to be applied
                                                on a sample.

        """
        assert isinstance(dataset_lr_res, int)
        assert isinstance(dataset_hr_res, int)

        self.transform = transform
        self.stretch_f = data_scaling

        self.cache_format = '.npz' # Compressed numpy files
        self.lr_res = lr_res
        self.hr_res = hr_res

        # Create the downsample sum classes
        self.downsample_lr = DownsampleSum(output_size=dataset_lr_res)
        self.downsample_hr = DownsampleSum(output_size=dataset_hr_res)

        self.randomflip = RandomFlip()

        if data_scaling:
            # Create the normalization class
            self.normalize = Normalize(data_scaling)
        else:
            self.normalize = None

        if lr_res != dataset_lr_res and hr_res != dataset_hr_res:
            # Create the randomcrop class
            self.randomcrop = Crop(lr_res, hr_res)
        else:
            self.randomcrop = None


        dataset_dir = os.path.join(datasets_dir, dataset_name)

        fits_dir = os.path.join(dataset_dir, "fits")

        if not os.path.exists(dataset_dir):
            raise OSError(f"Dataset directory {dataset_dir} does not exists")

        if not os.path.exists(fits_dir):
            raise OSError(f"Fits directory {fits_dir} does not exists")

        # Create the cache dir if it does not exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.fits_files = []
        self.file_names = []

        # Only save the files that end with .fits
        for file in os.listdir(fits_dir):
            if file.endswith(".fits"):
                self.fits_files.append(file)
                file_name = os.path.splitext(file)[0]
                self.file_names.append(file_name)

        print(f"Detected {len(self.fits_files)} fits files")

        # Check if the dataset specific cache dir exists, if not create it
        dataset_cache_dir = os.path.join(cache_dir, dataset_name)
        if not os.path.exists(dataset_cache_dir):
            os.makedirs(dataset_cache_dir)

        # Check if the low and high resolution caches exits if not create them
        self.hr_cache_dir = os.path.join(dataset_cache_dir, str(dataset_hr_res))
        self.lr_cache_dir = os.path.join(dataset_cache_dir, str(dataset_lr_res))

        if os.path.exists(self.hr_cache_dir):
            print(f"Found existing hr dataset cache dir {self.hr_cache_dir}")
        else:
            print(f"No hr dataset cache dir found, creating cache dir {self.hr_cache_dir}")
            os.makedirs(self.hr_cache_dir)

        if os.path.exists(self.lr_cache_dir):
            print(f"Found existing lr dataset cache dir {self.lr_cache_dir}")
        else:
            print(f"No hr dataset cache dir found, creating cache dir {self.lr_cache_dir}")
            os.makedirs(self.lr_cache_dir)

        hr_cache_files = os.listdir(self.hr_cache_dir)
        hr_cache_file_names = [os.path.splitext(x)[0] for x in hr_cache_files]

        lr_cache_files = os.listdir(self.lr_cache_dir)
        lr_cache_file_names = [os.path.splitext(x)[0] for x in lr_cache_files]

        # Check and generate hr cache files
        print("Checking and generating cache files...")
        for fits_file in tqdm(self.fits_files):
            file_name = os.path.splitext(fits_file)[0]

            file_in_hr_cache = file_name in hr_cache_file_names
            file_in_lr_cache = file_name in lr_cache_file_names

            if file_in_hr_cache and file_in_lr_cache:
                # File present in hr adn lr cache, we do not need to generate it
                continue
            else:
                # File not present, we need to generate it
                # Load the fits image
                img = self.load_fits(os.path.join(fits_dir, fits_file))

                # Downsamplesum to the desired image resolutions
                if not file_in_hr_cache:
                    hr_img = self.downsample_hr(img)
                    # Save as compressed numpy file
                    hr_cache_file_name = os.path.join(self.hr_cache_dir, file_name)
                    np.savez(hr_cache_file_name + self.cache_format, data=hr_img)

                if not file_in_lr_cache:
                    lr_img = self.downsample_lr(img)
                    # Save as compressed numpy file
                    lr_cache_file_name = os.path.join(self.lr_cache_dir, file_name)
                    np.savez(lr_cache_file_name + self.cache_format, data=lr_img)

        # At this point all the cache files are generated, reload them with the new ones
        self.hr_cache_files = os.listdir(self.hr_cache_dir)

        self.lr_cache_files = os.listdir(self.lr_cache_dir)


    def load_fits(self, fits_path):
        hdu = fits.open(fits_path)
        # Extract the image data from the fits file and convert to float
        # (these images will be in int but since we will work with floats in pytorch we convert them to float)
        img = hdu['PRIMARY'].data.astype(np.float32)
        hdu.close()

        return img

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        file_name = self.file_names[idx]

        # Load the lr and hr images from cache
        lr_cache_file_name = os.path.join(self.lr_cache_dir, file_name + self.cache_format)
        lr_img = np.load(lr_cache_file_name)['data']

        hr_cache_file_name = os.path.join(self.hr_cache_dir, file_name + self.cache_format)
        hr_img = np.load(hr_cache_file_name)['data']

        # Do the random cropping to resolution
        if self.randomcrop:
            lr_img, hr_img = self.randomcrop(lr_img, hr_img)

        # Apply the transformations

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        lr_img, hr_img = self.randomflip(lr_img, hr_img)

        # Apply the normalization
        if self.normalize:
            lr_img, lr_max = self.normalize(lr_img)
            hr_img, hr_max = self.normalize(hr_img)
        else:
            lr_max = torch.max(lr_img).detach()
            hr_max = torch.max(hr_img).detach()

        # Torch needs the data to have dimensions [1, x, x]
        # lr_img = torch.unsqueeze(lr_img, axis=0)
        # hr_img = torch.unsqueeze(hr_img, axis=0)

        sample = {'lr': lr_img, 'hr': hr_img, 'source': file_name, 'lr_max': lr_max, 'hr_max': hr_max, 'stretch_mode': self.stretch_f}

        return sample