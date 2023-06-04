# Deep Learning-Based Super-Resolution (SR) and De-Noising (DN) for XMM-Newton Images: running inference

The results of this research are described in the paper <em>``Deep Learning-Based Super-Resolution and De-Noising for XMM-Newton Images"</em>, [2022, MNRAS, 517, 4054](https://doi.org/10.1093/mnras/stac2437). Pre-print available in [arXiv](https://arxiv.org/pdf/2205.01152.pdf).

More implementation details are described in Sam Sweere's master's thesis included in this repository.

Here we only provide the necessary code and models to run inference on XMM-Newton EPIC-pn images (see below for requirements).

## Requirements

The inference does not need GPU to run.

### Python packages

These are listed in [requirements.txt](requirements.txt), the inference code is tested with python 3.10. We strongly suggest to install the packages in a separate conda of virtual python environment.

### XMM-Newton image requirements

The models were trained and validated on EPIC-pn images in energy band `[0.5,2]` in detector coordinates (`DETX`,`DETY`) with pixel size of 4"/pixel.

The training for the denoising part was performed on images of 20 ks exposure, predicting images as if they were exposed for 50 ks. The training images were cleaned for periods of high background using the XMM-Newton pipeline-derived thresholds, before limiting them to 20 ks exposure. You may still be able to run the DN inference on images with exposure different from 20 ks, because the training and validation was done on count-rate images. However, the results may not be as good if the exposure is too different from 20 ks.

The SR-predicted images will have two times smaller pixel size and two times smaller PSF, and will be free of background, although some low level spurious residuals may be present in the output.

Both the SR and DN training were performed on specially cropped and zero-padded images to avoid unnecessary empty spaces. Therefore the output predicted images will not match the default input image in pixel coordinates. That is why we provide on output the cropped and padded input image, so that a direct pixel-to-pixel comparison could be made. 

Both the input and output cropped images are in detector coordinates with proper `RA-DEC` equatorial world coordinate system.

### Creating XMM-Newton images for inference

The python script `make_detxy_image_pps.py` can be used to create an image suitable as an input for the inference. To run it, you will need to have the XMM-SAS and all its requirements installed, as explained in [these pages](https://www.cosmos.esa.int/web/xmm-newton/download-and-install-sas). 

The script uses the python wrapper for running the SAS tasks, `pysas`. If running `python -c 'import pysas'` ends with an error, then you have to fix your XMM-SAS installation.

The script has as an input the `OBS_ID`. It will access the XMM Science Archive and download the pipeline products (PPS) using `astroquery.esa.xmm_newton`, or will use already downloaded PPS files if `--pps_dir` points to a folder with existing PPS files. The user can specify the temporary folder where all the intermediate files will be stored (calibration files index, calibrated event lists, lightcurves etc). 

The PPS calibrated event list will be filtered for periods of high background using the PPS-derived threshold (available in the PPS products) and an image in `DETX`,`DETY` coordinates in energy band `[0.5,2]` keV (`PI >= 500 && PI <= 2000`) will be created with the default pixel size of 4"/pixel, filtering the evens with `FLAG == 0` and `PATTERN <= 4`.

As a final step, the header of the detector coordinates image will be updated with correct `RA-DEC` WCS. This is necessary as the default detector coordinate images from XMM-SAS do not contain usable WCS.

Here is the help for `make_detxy_image_pps.py`: 

```
usage: make_detxy_pn_image_pps.py [-h] [--low_pi LOW_PI] [--high_pi HIGH_PI] [--expo_time EXPO_TIME] [--binSize BINSIZE] [--save_dir SAVE_DIR]
                                  [--pps_dir PPS_DIR]
                                  obsid

XMMSAS generating an image for OBS_ID in DETX,DETY from PPS and GTI filtered event lists

positional arguments:
  obsid                 The OBS_ID to use, PPS will be downloaded

options:
  -h, --help            show this help message and exit
  --low_pi LOW_PI       Low energy (Pulse Intensity, PI) of events to consider, in eV (integer), default 500 eV
  --high_pi HIGH_PI     High energy (PI) of events to consider, in eV (integer), default 2000 eV
  --expo_time EXPO_TIME
                        Will select a sublist of events amounting to this exposure time, in kiloseconds. Default 20 ks. If negative or 0 then no time
                        selection.
  --binSize BINSIZE     The image bin size (integer), default 80 ==> 4 arcsec pixel
  --save_dir SAVE_DIR   Folder where to save the processed/output files
  --pps_dir PPS_DIR     Folder where the PPS files are, if None will download them from the archive

```

**Note:** The default parameters are set to values that will create an EPIC-pn image in detector coordinates that can be used directly as an input to the inference code. If `--save_dir` is not specified then the XMM products will be downloaded and stored in the current working folder. 


### Example #1:

```
python make_detxy_pn_image_pps.py 0840940101 --save_dir /scratch/XMM_data
```

This will download PPS products from the [XMM-Newton archive (XSA)](http://nxsa.esac.esa.int/nxsa-web/#search) and will extract them in a folder `/scratch/XMM_data/0840940101/pps`, then it will filter the event list and create an image with 4"/pixel in detector coordinates in band `[0.5,2] keV` with exposure 20 ks (with tollerance of 15%) in the same folder. The default image name will be `pn_cleaned_20.0ks_detxy_500_2000_80.fits` and can be used as an input to the SR and DN inference.

### Example #2:

If you already have the PPS files, downloaded from the [XSA](http://nxsa.esac.esa.int/nxsa-web/#search) (e.g. using the web interface) and extracted in `/scratch/XMM_pps`: the untaring the XSA .tar file will create a folder `0840940101` with subfolder `pps` where the PPS products will be found.

```
python make_detxy_pn_image_pps.py 0840940101 --pps_dir /scratch/XMM_pps/0840940101/pps
```

This will use the PPS products in a folder `/scratch/XMM_pps/0840940101/pps` and then will filter the event list and create an image with 4"/pixel in detector coordinates in band `[0.5,2] keV` with exposure 20 ks (with tolerance of 15%) and will store those in the same folder. The default image name will be `pn_cleaned_20.0ks_detxy_500_2000_80.fits` and can be used as an input to the SR and DN inference. 


## Install

1. Clone the GitHub repository:<br/>
2. Enter in the repository folder
3. Create a conda environment (or virtualenv) and install the required python packages <br/> `pip install -r requirements.txt`
4. Activate the new environment
5. Test the code using already available input image in `tests` folder (note that this will only test the inference and not the input image creation):

```
python run_inference_on_real.py -h
```
you should see this message:
```
usage: run_inference_on_real.py [-h] [--which WHICH] [--display] filein

Predict XMM SR or DN image

positional arguments:
  filein         The FITS filename in detxy coordinates with WCS

options:
  -h, --help     show this help message and exit
  --which WHICH  Which inference to apply? SR or DN
  --display      Display the input and predicted image?
```
The `filein` is the relative or the full path of input image as generated with `make_detxy_pn_image_pps.py`. The outputs will be stored in the same folder as the input image: with suffix `_input_wcs_SR.fits.gz` for the input image for SR run, or `_input_wcs_DN.fits.gz`  or `_predict_wcs_SR.fits.gz` and `_predict_wcs_SR.fits.gz` for the outputs, depending if DN or SR mode is given as input argument ot the script.

Optionally the input and output images can be displayed using `matplotlib` (if `--display` is set).

If all good then run the inference on a real image:

```
python run_inference_on_real.py --which SR --display \
    tests/0827240501_image_split_500_2000_20ks_1_1.fits.gz

python run_inference_on_real.py --which DN --display \
    tests/0827240501_image_split_500_2000_20ks_1_1.fits.gz
```
---

Ivan Valtchanov, XMM SOC, April 2023 
