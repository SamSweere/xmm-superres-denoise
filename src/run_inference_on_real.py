import argparse
import os
import warnings

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import ImageNormalize, PercentileInterval
from astropy.wcs import WCS

from utils.filehandling import read_yaml
from utils.inference_funcs import run_inference_on_file

warnings.simplefilter("ignore", category=AstropyWarning)

import matplotlib.pylab as plt

# plt.style.use(['seaborn-colorblind','~/presentation.mplstyle'])
#%%
#
# Parse the input
parser = argparse.ArgumentParser(description="Predict XMM SR or DN image")
parser.add_argument(
    "filein", type=str, help="The FITS filename in detxy coordinates with WCS"
)
parser.add_argument(
    "--which", type=str, default="SR", help="Which inference to apply? SR, DN or SR+DN"
)
parser.add_argument(
    "--display",
    default=False,
    action="store_true",
    help="Display the input and predicted image?",
)
#
args = parser.parse_args()

# inference_model_names = ["XMM-SuperRes","XMM-DeNoise"]

if args.which.upper() == "SR":
    model_name = "XMM-SuperRes"
elif args.which.upper() == "DN":
    model_name = "XMM-DeNoise"
else:
    print("Not implemented: either SR or DN")
    raise RuntimeError

# hardcoded settings
root_dir = cwd = os.path.dirname(os.path.abspath(__file__))
print("Running inference using model:", model_name)
#
model_base_path = os.path.join(f"{root_dir}/../models")
onnx_filepath = f"{model_base_path}/{model_name}.onnx"
if not os.path.isfile(onnx_filepath):
    print(f"The model ONNX file not found, cannot continue! ==> {onnx_filepath}")
    raise FileNotFoundError
#
model_data_config_path = os.path.join(model_base_path, f"{model_name}_data_config.yaml")
if not os.path.isfile(model_data_config_path):
    print(
        f"The model config YAML file not found, cannot continue! ==> {model_data_config_path}"
    )
    raise FileNotFoundError
#
real_datasets_dir = os.getcwd()
real_dataset_name = ""
# # Load the dataset config and modify to match the tests
dataset_mode = "real"  # can be sim or real
dataset_config = read_yaml(model_data_config_path)
dataset_config["model_name"] = args.which.upper()
dataset_config["debug"] = True  # Set to True when debugging
dataset_config["dataset_type"] = dataset_mode
dataset_config["dataset_name"] = real_dataset_name
dataset_config["batch_size"] = 1  # For inference we use batch sizes of 1
dataset_config["datasets_dir"] = real_datasets_dir
dataset_config["check_files"] = False
dataset_config["include_hr"] = False
dataset_config["crop_mode"] = "center"
#
# # output folder where the results will be saved

# #
# dataset_out_filepath = args.outdir
# if not os.path.exists(dataset_out_filepath):
#     os.makedirs(dataset_out_filepath)#
# #
# output_path = dataset_out_filepath
#%%
#
# now run the inference
inf, outf = run_inference_on_file(args.filein, dataset_config, onnx_filepath)
#
if args.display:
    #
    # print (f'Now will display both {inf} and {outf}')
    #
    with fits.open(f"{inf}.fits.gz") as hdu1, fits.open(f"{outf}.fits.gz") as hdu2:
        img_in = hdu1[0].data
        wcs_in = WCS(hdu1[0].header)
        img_pred = hdu2[0].data
        wcs_pred = WCS(hdu2[0].header)
    #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 2, 1)
    norm = ImageNormalize(img_in, interval=PercentileInterval(99.5))
    ax.imshow(
        img_in, norm=norm, cmap=plt.cm.plasma, origin="lower", interpolation="nearest"
    )
    ax.set_title("Input")
    # overlay = ax.get_coords_overlay('fk5')
    # overlay.grid(color='white', ls='dotted')
    #
    ax = fig.add_subplot(1, 2, 2)
    norm = ImageNormalize(img_pred, interval=PercentileInterval(99.5))
    ax.imshow(
        img_pred, norm=norm, cmap=plt.cm.plasma, origin="lower", interpolation="nearest"
    )
    ax.set_title(f"Predicted {args.which.upper()}")
    #
    plt.show()
print(f"Predicted XMM-{args.which.upper()} image saved in {outf}.fits.gz")
#
