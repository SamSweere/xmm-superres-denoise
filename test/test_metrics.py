import os

import numpy as np
import torch
from PIL import Image
from astropy.io import fits
from matplotlib import pyplot as plt

from utils.ssim import ssim, ms_ssim


def load_fits(fits_path):
    try:
        hdu = fits.open(fits_path)
        # Extract the image data from the fits file and convert to float
        # (these images will be in int but since we will work with floats in pytorch we convert them to float)
        img = hdu['PRIMARY'].data.astype(np.float32)
        hdu.close()

        img = np.flip(img, 0)

        return img
    except Exception as e:
        print("ERROR failed to load fits file with error: ", e)
        print(fits_path)
        raise IOError(e)


def display_tensor(img_tens, title=''):
    plt.imshow(img_tens[0][0])
    plt.title(title)
    plt.colorbar()
    plt.show()


def add_subplot(fig, axs, index, img_tens, title='', vmin=None, vmax=None, cmap='viridis'):

    img = img_tens[0][0].cpu().detach().numpy()



    im1 = axs[index].imshow(img, interpolation='None', vmin=vmin, vmax=vmax, cmap=cmap)

    axs[index].axis('off')

    fig.colorbar(im1, ax=axs[index], fraction=0.046, pad=0.04)
    # fig.title(title, ax=axs[0])
    axs[index].set_title(title)

    return fig, axs


base_img_dir = '/home/sam/Documents/ESA/data/test/display_selection'
out_img_dir = '/home/sam/Documents/ESA/data/test/structure_metric'
scaling_f = '' #'sqrt' #'sqrt' #''
data_range = 50

img_pair_list = os.listdir(base_img_dir)

# TODO: this is an override
# img_pair_list = ['TNG300_1_z_99_subhalos_175556_m_slice_r_2048_w_1000_n_z_p0_0_5ev_p1_2_0ev_sb_22_37_zoom_1_84_offx_-0']

gen_img_paths = []


for img_dir in img_pair_list:
    print("Processing:", img_dir)
    img_dir_path = os.path.join(base_img_dir, img_dir)

    input_img = load_fits(os.path.join(img_dir_path, 'input.fits.gz'))
    pred = load_fits(os.path.join(img_dir_path, 'prediction.fits.gz'))
    label = load_fits(os.path.join(img_dir_path, 'reference.fits.gz'))

    if scaling_f == 'sqrt':
        input_img = np.sqrt(input_img)
        pred = np.sqrt(pred)
        label = np.sqrt(label)
        data_range = np.sqrt(data_range)
    elif scaling_f == 'log':
        input_img = np.sqrt(input_img)
        pred = np.sqrt(pred)
        label = np.sqrt(label)
        data_range = np.sqrt(data_range)

    input_tens = torch.tensor(input_img.copy())
    input_tens = torch.unsqueeze(input_tens, 0)
    input_tens = torch.unsqueeze(input_tens, 0)

    pred_tens = torch.tensor(pred.copy())
    pred_tens = torch.unsqueeze(pred_tens, 0)
    pred_tens = torch.unsqueeze(pred_tens, 0)

    label_tens = torch.tensor(label.copy())
    label_tens = torch.unsqueeze(label_tens, 0)
    label_tens = torch.unsqueeze(label_tens, 0)

    diff_tens = pred_tens - label_tens
    L1_tens = torch.abs(pred_tens - label_tens)

    # TODO: the data range is dependent on the output exposure
    low_winsize = 11
    low_sigma = 1.5
    med_winsize = 13
    med_sigma = 2.5
    high_winsize = 15
    high_sigma = 4.5

    ms_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    K = (0.01, 0.05)

    ssim_low_torch, ssim_low_images = ssim(pred_tens, label_tens, win_size=low_winsize, win_sigma=low_sigma, data_range=data_range, K=K)
    ms_ssim_low_torch = ms_ssim(pred_tens, label_tens, win_size=low_winsize, win_sigma=low_sigma, data_range=data_range, weights=ms_weights, K=K)

    ssim_med_torch, ssim_med_images = ssim(pred_tens, label_tens, win_size=med_winsize, win_sigma=med_sigma, data_range=data_range, K=K)
    ms_ssim_med_torch = ms_ssim(pred_tens, label_tens, win_size=med_winsize, win_sigma=med_sigma, data_range=data_range, weights=ms_weights, K=K)

    ssim_high_torch, ssim_high_images = ssim(pred_tens, label_tens, win_size=high_winsize, win_sigma=high_sigma, data_range=data_range, K=K)
    ms_ssim_high_torch = ms_ssim(pred_tens, label_tens, win_size=high_winsize, win_sigma=high_sigma, data_range=data_range, weights=ms_weights, K=K)

    print(f"SSIM {low_winsize} {low_sigma}: {ssim_low_torch}")
    print(f"MS_SSIM {low_winsize} {low_sigma}: {ms_ssim_low_torch}")
    print(f"SSIM {med_winsize} {med_sigma}: {ssim_med_torch}")
    print(f"MS_SSIM {med_winsize} {med_sigma}: {ms_ssim_med_torch}")
    print(f"SSIM {high_winsize} {high_sigma}: {ssim_high_torch}")
    print(f"MS_SSIM {high_winsize} {high_sigma}: {ms_ssim_high_torch}")

    fig, axs = plt.subplots(1, 8, figsize=(12*2, 2*2))
    fig.tight_layout()

    # left = 0.125  # the left side of the subplots of the figure
    # right = 0.9  # the right side of the subplots of the figure
    # bottom = 0.1  # the bottom of the subplots of the figure
    # top = 0.9  # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    # plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    vmin = 0
    vmax = max(torch.max(pred_tens), torch.max(label_tens))

    fig, axs = add_subplot(fig, axs, index=0, img_tens=input_tens, title='Input', vmin=vmin, vmax=vmax, cmap='plasma')
    fig, axs = add_subplot(fig, axs, index=1, img_tens=pred_tens, title='Prediction', vmin=vmin, vmax=vmax, cmap='plasma')
    fig, axs = add_subplot(fig, axs, index=2, img_tens=label_tens, title='Label', vmin=vmin, vmax=vmax, cmap='plasma')
    fig, axs = add_subplot(fig, axs, index=3, img_tens=L1_tens, title='L1', cmap='afmhot_r')
    fig, axs = add_subplot(fig, axs, index=4, img_tens=ssim_low_images, title=f"ssim_map {low_winsize} {low_sigma}", cmap="viridis_r")
    fig, axs = add_subplot(fig, axs, index=5, img_tens=ssim_med_images, title=f"ssim_map {med_winsize} {med_sigma}", cmap="viridis_r")
    fig, axs = add_subplot(fig, axs, index=6, img_tens=ssim_high_images, title=f"ssim_map {high_winsize} {high_sigma}", cmap="viridis_r")


    # index = 4
    # for key in ssim_images.keys():
    #     fig, axs = add_subplot(fig, axs, index=index, img_tens=ssim_images[key], title=key)
    #     index += 1

    score_index = 7
    # Add the score text
    axs[score_index].axis([0, 10, 0, 10])
    axs[score_index].axis('off')
    axs[score_index].text(1, 8, f'SSIM {low_winsize} {low_sigma}: {round(ssim_low_torch.item(), 4)}', fontsize=12)
    axs[score_index].text(1, 7, f'MS_SSIM {low_winsize} {low_sigma}: {round(ms_ssim_low_torch.item(), 4)}', fontsize=12)
    axs[score_index].text(1, 6, f'SSIM {med_winsize} {med_sigma}: {round(ssim_med_torch.item(), 4)}', fontsize=12)
    axs[score_index].text(1, 5, f'MS_SSIM {med_winsize} {med_sigma}: {round(ms_ssim_med_torch.item(), 4)}', fontsize=12)
    axs[score_index].text(1, 4, f'SSIM {high_winsize} {high_sigma}: {round(ssim_high_torch.item(), 4)}', fontsize=12)
    axs[score_index].text(1, 3, f'MS_SSIM {high_winsize} {high_sigma}: {round(ms_ssim_high_torch.item(), 4)}', fontsize=12)

    out_path = os.path.join(out_img_dir, scaling_f + "_" + img_dir) + '.png'
    plt.savefig(out_path)
    print("Saved results to:", out_path)

    gen_img_paths.append(out_path)
    # plt.show()

    plt.close()


    # im1 = axs[0, 0].imshow(pred_tens[0][0], interpolation='None')
    # fig.colorbar(im1, ax=axs[0, 0])
    # axs[0, 0].title('Pre')

    # display_tensor(pred_tens, 'Prediction')
    # display_tensor(label_tens, 'Label')
    # display_tensor(diff_tens, 'Diff')
    # for key in ssim_images.keys():
    #     display_tensor(ssim_images[key], key)

    #
    # plt.imshow(pred)
    # plt.title("Prediction")
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(label)
    # plt.title("Label")
    # plt.colorbar()
    # plt.show()

    asdf = 123

# From: https://note.nkmk.me/en/python-pillow-concat-images/
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

concat_images = True
if concat_images:
    merged_img = None
    for img_path in gen_img_paths:
        im = Image.open(img_path)

        if merged_img is None:
            merged_img = im
        else:
            merged_img = get_concat_v(merged_img, im)


    merged_img.save(os.path.join(out_img_dir, scaling_f + "_" + 'concat.png'))
    print("Saved combined image to:",os.path.join(out_img_dir, scaling_f + "_" + 'concat.png'))