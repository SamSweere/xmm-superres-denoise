# Notes on running inference to predict SR and DN images for XMM-Newton EPIC-pn

The training, validation and testing of the models are described in greater detial in [Sweere et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.tmp.2335S/abstract).

This note is to explain how to run the inference and predict super-resolution or denoised XMM-Newton EPIC-pn images.

The way the networks were trained dictates the input for the inference: 

* It must be a pn image in detector coordinates `(DETX,DETY)` with pre-defined pixel binsize of 80 (in units of 0.05 arcsec), with output shape `(403,411)` pixels.
* Only energy band [0.5,2] keV was used in training.
* Exposure 20 ks, shorter or longer exposures may work too, but the networks (both SR and DN) were trained with 20 ks images. Both networks were trained using exposure-corrected images (however, no vignetting correction was applied). Therefore, in a certain range of exposures around 20 ks the results should be good.


## Workflow to produce the input image

The following steps requre XMM-SAS to be available.

1. Download and extract XMM pipeline products (PPS) from the [XMM-Newton archive](http://nxsa.esac.esa.int/nxsa-web/#home).
2. Identify the good-time-intervals (GTI) when the flaring background is not affecting the pn observation. This step will use the PPS flaring background time-series file and the PPS derived count-rate threshold.
3. (Optionally) select only events up to 20 ks exposure (taking into account the bad intervals!)
4. Filter the PPS calibrated event list with the GTI and with some additional criteria, the hardcoded expression in `xmmsas_tools.filter_events_gti()` is 
`(FLAG == 0) && gti({gti_file},TIME) && (PI>150) && (PATTERN <= 4)`
5. Using the filtered event list, produce an image in detector coordinates with `xmmsas_tools.make_detxy_image()`

All these steps can be performed by the user with his own XMM-SAS procedures as long as the output image is in detector coordinates and with dimensions `(403,411)` pixels and exposure around 20 ks. 

Note that by default, the SAS-produced image in detector coordinates will not have proper RA-DEC sky coordinate system. The function in `xmmsas_tools.make_detxy_image()` has an additional step to add a proper RA-DEC WCS.

## Run the inference 

This part does not require XMM-SAS.

Execute `python run_inference_on_real <detxy_file_name> --which SR|DN`

Depending on `--which` the process will produce a predicted image with twice the nominal XMM resolution (SR) or a predicted denoised image (DN).

## Caveats:

### Super-resolution

* The predicted SR image will have x2 smaller pixel size: 2"/pixel compared to the nominal 4"/pixel and ideally should have the PSF twice as smaller, e.g. FWHM should be 3" instead of 6". The predicted image will be with dimensions `(416,416)` (the input `403x411` is zero-padded).
* The SR image will be cleaned from background
* The network was trained on simulated images of 20 ks, predicting SR images with shorter (or longer) exposures should work in principle, but may lead to increased noise and possibly spurious features.
* The network was trained on images in energy band [0.5,2] keV. Predicting SR images in other bands should work in principle but the results may not be as good: increased noise and spurious features.

### Denoising

* The predicted DN image will be with dimensions `(416,416)` (the input `403x411` is zero-padded) and pixel size as the input, i.e. 4"/pixel.
* No changes in the PSF size.
* The network was trained on simulated images with 20ks exposure predicting images with 50ks, i.e. in principle shuold lead to an increase of the signal-to-noise ratio by a factor of 1.6. Using DN model on images with significantly different exposure (smaller or larger) may lead to undesirable effects.
* We are currently training a new network to predict 50ks exposure images from 10ks input (SNR increase a factor of 2.2).


## Practical usage of the code

The simplest use is to run `python inference_end2end_obsid.py`.

```
usage: inference_end2end_obsid.py [-h] [--wdir WDIR] [--expo_time EXPO_TIME] obsid

Predict XMM SR or DN image

positional arguments:
  obsid                 The OBS_ID to process

optional arguments:
  -h, --help            show this help message and exit
  --wdir WDIR           The working top folder name, must exist
  --expo_time EXPO_TIME
                        Will extract only this exposure time (in ks) from the event list. Set it to negative to use
                        the GTI one.

```

It will follow the steps in the workflow as described above, and will produce predicted images using the SR and the DN models.

Some comments:

* The `obsid` must be public.
* Downloading the PPS products and saving them in `<wdir>/<obsid>/pps` may take time, be patient.
* If `wdir` is not set, the current folder wil lbe used.
* The `expo_time` is set to 20 ks by default (see above why). You can use a different exposure or set `--expo_time -1` to use the full GTI-filtered one.
* All products will be saved in `<wdir>/<obsid>/proc` (hardcoded for now): filtered event lists, GTI files, GTI diagnostc plot and the images.


_Ivan Valtchanov_, XMM SOC, Oct 2022
