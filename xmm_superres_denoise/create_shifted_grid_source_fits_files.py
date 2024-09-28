import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from xmm_superres_denoise.datasets.utils import load_fits
from pathlib import Path
from xmm_superres_denoise.transforms.data_scaling_functions import (
    asinh_scale,
    linear_scale,
    log_scale,
    sqrt_scale,
    hist_eq_scale,
)
from astropy.io import fits
from torchvision.transforms.functional import pad
from shifted_grid_sources.utils.utils import return_grid_dir
import argparse


def create_directory_if_not_exists(directory_path: Path):
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def modify_and_save_fits(input_fits_path: Path, output_fits_path: Path, modified_img: np.ndarray):
    try:
        with fits.open(input_fits_path) as hdu:
            # Update the image data with the modified image
            hdu["PRIMARY"].data = modified_img.astype(np.float32)
            header = hdu["PRIMARY"].header
            header.pop('')
            # Save to a new FITS file
            hdu.writeto(output_fits_path, overwrite=True)
    except Exception as e:
        raise IOError(f"Failed to save modified FITS file {output_fits_path} with error:", e)


def shift_and_add_images(img, shift_amount, res, br_diff):
    shift = res * shift_amount if res != 1 else shift_amount

    # Shift the image upwards by shift_amount pixels
    shifted_img = np.pad(img[shift:], ((0, shift), (0, 0)), mode='constant', constant_values=0)

    # Add the shifted image to the original
    combined_img = img + br_diff * shifted_img

    return combined_img


def plot_images(img, combined_img, save_path=None, shifted_save_path=None, scale=sqrt_scale):
    # Create a subplot with specified rows and columns
    fig, ax = plt.subplots(figsize=(9, 9))
    fig_shifted, ax_shifted = plt.subplots(figsize=(9, 9))
    
    # Plot original and shifted images
    ax.imshow(scale(img), cmap='viridis', origin='lower')
    ax_shifted.imshow(scale(combined_img), cmap='viridis', origin='lower')

    # Save the figure as a PDF if save_path is provided
    if save_path:
        fig.savefig(save_path)
        fig_shifted.savefig(shifted_save_path)
    else:
        plt.show()

    plt.close()


def main(base_save_path, down_scales, br_diffs_scales, exposures, resolutions):
    for ds in down_scales:
        for br_diff in br_diffs_scales:
            for exp, res, resl in zip(exposures, resolutions, [False, True]):
                # Define base directory of input fits files
                base_dir = f'/xmm-super/final_data/sim/xmm_sim_dataset/{exp}ks/test_grid/{res}x/'
                
                for shift_amount in np.arange(5):
                    # Define paths to save the images 
                    save_path = Path(base_save_path + f'original_grids/{res}x_{ds:.2f}ds')
                    shifted_save_path = Path(base_save_path + f'shifted_grids/br_diff_sc{br_diff}/{res}x_{ds:.2f}ds/shift{shift_amount}')
                    
                    # Create paths if they dont exist yet
                    create_directory_if_not_exists(save_path)
                    create_directory_if_not_exists(shifted_save_path)

                    for num in range(1):
                        grid_dir = return_grid_dir(num, res, exp, resl)
                        
                        # Add filenames
                        img_save_path = save_path / Path(f'{grid_dir[:-8]}.pdf')
                        img_shifted_save_path = shifted_save_path / Path(f'shifted_{grid_dir[:-8]}.pdf')

                        fits_path = Path(base_dir + grid_dir)
                        fits_shifted_save_path = shifted_save_path / Path(f'shifted_{grid_dir}')
                    
                        # Load FITS file
                        img = load_fits(fits_path)            
                        img = img['img'] * 1000 * exp * ds    

                        # Create shifted image 
                        combined_img = shift_and_add_images(img, shift_amount, res, br_diff)

                        # Save the combined image as a fits file
                        modify_and_save_fits(fits_path, fits_shifted_save_path, combined_img)

                        # Plot the images and save as 'grid_sources.pdf'
                        plot_images(img, combined_img, img_save_path, img_shifted_save_path, sqrt_scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process FITS files with specified parameters.')

    parser.add_argument('--base_save_path', type=str, default='/home/xmmsas/mywork/cleanup_new/xmm-superres-denoise/shifted_grid_sources/results/', help='Base directory for saving results.')
    parser.add_argument('--down_scales', nargs='+', type=float, default=[0.25, 0.5, 0.75, 1], help='List of factors used to downscale the input.')
    parser.add_argument('--br_diffs_scales', nargs='+', type=float, default=[1], help='List of brightness differences between the original and the shifted image.')
    parser.add_argument('--exposures', nargs='+', type=int, default=[20, 100], help='Exposure of low and high resolution imaage, respectively')
    parser.add_argument('--resolutions', nargs='+', type=int, default=[1, 2], help='Resolution of low and high resolution image, respectively')

    args = parser.parse_args()

    main(args.base_save_path, args.down_scales, args.br_diffs_scales, args.exposures, args.resolutions)
