import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplot

from datasets.utils import load_fits
from transforms.data_scaling_functions import asinh_scale, linear_scale, sqrt_scale, \
    log_scale
from PIL import Image
from matplotlib import cm

def save_img(img, out_path):
    plasma = cm.get_cmap("plasma")
    im = Image.fromarray(np.uint8(plasma(img/np.max(img)) * 255))
    im.save(out_path)

def plot_scale_fits():
    # fits_file_path = '/home/sam/Documents/ESA/data/datasets/xmm_selection_dataset/combined/100ks/img/1x/TNG50_1_z_99_subhalos_382215_m_proj_r_2048_w_400_n_0_1_0_gz_p0_0_5ev_p1_2_0ev_sb_16_2_zoom_1_37_offx_-0_05_offy_-0_01_mult_1_100ks_p_0-0_agn_abs.fits.gz'
    # fits_file_path = '/home/sam/Documents/ESA/data/datasets/xmm_selection_dataset/combined/100ks/img/1x/TNG100_1_z_99_subhalos_207473_m_proj_r_2048_w_1600_n_1_0_0_gz_p0_0_5ev_p1_2_0ev_sb_47_51_zoom_1_75_offx_0_04_offy_-0_03_mult_1_100ks_p_0-0_agn_abs_back_16292226137142594.fits.gz'
    fits_file_path = '/home/sam/Documents/ESA/data/datasets/display_datasets/xmm_split_display_selection_small/full/0824450501_image_500_2000_70.54ks.fits'

    img = load_fits(fits_file_path)['img']

    img = np.flip(img, axis=0)

    # Limit the max value

    img = np.clip(img, a_min=0.0, a_max=0.002)

    img = img/np.max(img)

    linear_img = linear_scale(img)
    sqrt_img = sqrt_scale(img)
    asinh_img = asinh_scale(img)
    log_img = log_scale(img)


    plt.figure(figsize=(40,10))

    total_subplots = 4
    cmap = 'plasma'

    subplot(1, total_subplots, 1)
    plt.imshow(linear_img, cmap=cmap)
    plt.title("No scaling")
    plt.axis('off')
    subplot(1, total_subplots, 2)
    plt.imshow(sqrt_img, cmap=cmap)
    plt.title("Sqrt scaling")
    plt.axis('off')

    subplot(1, total_subplots, 3)
    plt.imshow(asinh_img, cmap=cmap)
    plt.title(f"asinh")
    plt.axis('off')
    subplot(1, total_subplots, 4)
    plt.imshow(log_img, cmap=cmap)
    plt.title("Log scaling")
    plt.axis('off')

    save_img(linear_img, "/home/sam/Documents/ESA/thesis/figures/method/linear_img.png")
    save_img(sqrt_img, "/home/sam/Documents/ESA/thesis/figures/method/sqrt_img.png")
    save_img(asinh_img, "/home/sam/Documents/ESA/thesis/figures/method/asinh_img.png")
    save_img(log_img, "/home/sam/Documents/ESA/thesis/figures/method/log_img.png")

    # subplot(1, total_subplots, 3)
    # plt.imshow(asinh_scale(img, a = 0.01))
    # plt.title(f"asinh a=0.01")
    # plt.axis('off')
    # subplot(1, total_subplots, 4)
    # plt.imshow(asinh_scale(img, a = 0.02))
    # plt.title(f"asinh a=0.02")
    # plt.axis('off')
    # subplot(1, total_subplots, 5)
    # plt.imshow(log_scale(img, a=1000))
    # plt.title("Log scaling, a=1000")
    # plt.axis('off')
    # subplot(1, total_subplots, 6)
    # plt.imshow(log_scale(img, a=10000))
    # plt.title("Log scaling, a=10000")
    # plt.axis('off')

    # for i in range(5):
    #     plt.subplot(1, total_subplots, 5+i)
    #     a = float(i+1)/10.0
    #     img_asinh = np_asinh_scale(img, a=a)
    #     plt.imshow(img_asinh)
    #     plt.title(f"asinh a={a}")


    # plt.subplot(1, 3, 2)
    # plt.imshow(img_asinh_01)
    # plt.title("asinh scaling a=0.1")
    # plt.subplot(1, 3, 3)
    # plt.imshow(img_asinh_03)
    # plt.title("asinh scaling a=0.3")
    plt.tight_layout()
    plt.savefig("/home/sam/Documents/ESA/thesis/figures/method/data_scaling_example.png")
    plt.show()

def data_scaling_graphs():
    x = np.linspace(0, 1, 1000)
    a = asinh_scale(x, a = 0.02)
    # b = np_asinh_scale(x, a = 0.1)
    c = np.sqrt(x)
    d = log_scale(x, a=1000)
    d = np.clip(d, a_min=0.0, a_max=1.0)

    # d = np.log(x+0.000001)
    # d = d - np.min(d)
    # d = d/np.max(d)

    plt.plot(x, x, 'r', label='linear')
    plt.plot(x, c, 'b', label='sqrt') # plotting t, c separately
    plt.plot(x, a, 'g', label='asinh') # plotting t, a separately
    plt.plot(x, d, 'orange', label='log') # plotting t, c separately

    # plt.plot(x, b, 'b', label='asinh a=0.1') # plotting t, b separately

    # plt.plot(x, e, label='log10') # plotting t, c separately
    plt.legend(loc='lower right')
    plt.savefig("/home/sam/Documents/ESA/thesis/figures/method/scaling_functions_graph.pdf")
    plt.show()

data_scaling_graphs()
# plot_scale_fits()

# a_sum = sum(a[200:600])
# b_sum = sum(b[200:600])
#
# print((a_sum - b_sum)/(a_sum))


