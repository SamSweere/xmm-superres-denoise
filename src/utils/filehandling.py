import math
import os
from datetime import datetime

import numpy as np
import yaml
from astropy.io import fits


#%%
def write_xmm_file_to_fits(
    img,
    output_dir,
    source_file_name,
    res_mult,
    exposure,
    comment=None,
    out_file_name=None,
    in_header=None,
):
    header = fits.Header()

    header["IMG_FILE"] = (source_file_name, "Input source file")

    # Add header information
    if in_header is not None:
        header_keys_to_omit = [
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "NAXIS1",
            "NAXIS2",
            "EXTEND",
            "XPROC0",
            "XDAL0",
            "CREATOR",
            "DATE",
            "CTYPE1",
            "CRPIX1",
            "CRVAL1",
            "CDELT1",
            "CTYPE1L",
            "CRPIX1L",
            "CRVAL1L",
            "CDELT1L",
            "LTV1",
            "LTM1_1",
            "CTYPE2",
            "CRPIX2",
            "CRVAL2",
            "CDELT2",
            "CTYPE2L",
            "CRPIX2L",
            "CRVAL2L",
            "CDELT2L",
            "LTV2",
            "LTM2_2",
            "LTM1_2",
            "LTM2_1",
            "ONTIME01",
            "ONTIME02",
            "ONTIME03",
            "ONTIME04",
            "ONTIME05",
            "ONTIME06",
            "ONTIME07",
            "ONTIME08",
            "ONTIME09",
            "ONTIME10",
            "ONTIME11",
            "ONTIME12",
            "EXPOSURE",
            "DURATION",
        ]

        header_to_add = in_header
        for omit_key in header_keys_to_omit:
            header_to_add.pop(omit_key, None)

        for key in header_to_add.keys():
            header[key] = header_to_add[key]

    header["EXPOSURE"] = exposure

    # TODO: put the real WCS values in here, these are centered on 0 0 and based on simulated images
    # Update the cdelt values
    header["CDELT1"] = -0.00111111113801599 / res_mult
    # header['CDELT1L'] = 80.0/res_mult
    header["CDELT2"] = 0.00111111113801599 / res_mult
    # header['CDELT2L'] = 80.0/res_mult

    if res_mult == 1:
        header["CRPIX1"] = 244.0
        header["CRPIX2"] = 224.0
    elif res_mult == 2:
        header["CRPIX1"] = 487.5
        header["CRPIX2"] = 447.5

    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0

    header["CTYPE1"] = "RA---TAN"
    # header['CTYPE1L'] = 'RA---TAN'
    header["CTYPE2"] = "DEC---TAN"
    # header['CTYPE2L'] = 'DEC---TAN'

    if comment is not None:
        header["COMMENT"] = comment

    if out_file_name is None:
        out_file_name = source_file_name

    header["COMMENT"] = (
        f"Code Created by Sam Sweere (samsweere@gmail.com) for ESAC. File created at "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )

    hdu = fits.PrimaryHDU(img, header=header)
    # We limit the file name such that is stays usable
    compressed_out_path = os.path.join(output_dir, out_file_name + ".fits.gz")
    hdu.writeto(compressed_out_path, overwrite=True)


#%%
# Modified by IVAN
#
#
# Empirically figuring this out (IvanV), from the original DETXY *with* proper WCS
#
# CRPIX1 = CRPIX1 + 6
# CRPIX2 = CRPIX2 + 2
#
def write_xmm_file_to_fits_wcs(
    img,
    output_dir,
    source_file_name,
    res_mult,
    exposure,
    comment=None,
    out_file_name=None,
    in_header=None,
):
    header = fits.Header()

    header["IMG_FILE"] = (source_file_name, "Input source file")

    # Add header information
    if in_header is not None:
        header_keys_to_omit = [
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "NAXIS1",
            "NAXIS2",
            "EXTEND",
            "XPROC0",
            "XDAL0",
            "CREATOR",
            "DATE",
            # "CTYPE1",
            # "CRPIX1",
            # "CRVAL1",
            # "CDELT1",
            "CTYPE1L",
            "CRPIX1L",
            "CRVAL1L",
            "CDELT1L",
            "LTV1",
            "LTM1_1",
            # "CTYPE2",
            # "CRPIX2",
            # "CRVAL2",
            # "CDELT2",
            "CTYPE2L",
            "CRPIX2L",
            "CRVAL2L",
            "CDELT2L",
            "LTV2",
            "LTM2_2",
            "LTM1_2",
            "LTM2_1",
            "ONTIME01",
            "ONTIME02",
            "ONTIME03",
            "ONTIME04",
            "ONTIME05",
            "ONTIME06",
            "ONTIME07",
            "ONTIME08",
            "ONTIME09",
            "ONTIME10",
            "ONTIME11",
            "ONTIME12",
            "EXPOSURE",
            "DURATION",
        ]

        header_to_add = in_header
        for omit_key in header_keys_to_omit:
            header_to_add.pop(omit_key, None)

        for key in header_to_add.keys():
            header[key] = header_to_add[key]

    header["EXPOSURE"] = exposure
    #
    # now update the reference pixel only!
    #
    crpix1_new = header["CRPIX1"] + 6
    crpix2_new = header["CRPIX2"] + 2
    header["CRPIX1"] = crpix1_new
    header["CRPIX2"] = crpix2_new
    if res_mult == 2:
        header["CRPIX1"] = res_mult * crpix1_new + 0.5
        header["CRPIX2"] = res_mult * crpix2_new + 0.5
        cdelt1 = header["CDELT1"] / res_mult
        cdelt2 = header["CDELT2"] / res_mult
        header["CDELT1"] = cdelt1
        header["CDELT2"] = cdelt2
        #
        crota2 = 90.0 - float(header["PA_PNT"])
        header["CROT2"] = crota2
        crota2_rad = math.radians(crota2)
        # add the CD matrix, just in case?
        cd1_1 = cdelt1 * math.cos(crota2_rad)
        cd1_2 = -1.0 * cdelt2 * math.sin(crota2_rad)
        cd2_1 = cdelt1 * math.sin(crota2_rad)
        cd2_2 = cdelt2 * math.cos(crota2_rad)
        header["CD1_1"] = cd1_1
        header["CD1_2"] = cd1_2
        header["CD2_1"] = cd2_1
        header["CD2_2"] = cd2_2

    if comment is not None:
        header["COMMENT"] = comment

    if out_file_name is None:
        out_file_name = f"{source_file_name.replace('.fits','')}_sr_predict"
    #
    header["COMMENT"] = "Code Created by Sam Sweere (samsweere@gmail.com) for ESAC"
    header["COMMENT"] = "WCS code written by Ivan V"
    header[
        "COMMENT"
    ] = f"File created on {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    #
    hdu = fits.PrimaryHDU(img, header=header)
    # We limit the file name such that is stays usable
    compressed_out_path = f"{output_dir}/{out_file_name}.fits.gz"
    hdu.writeto(compressed_out_path, overwrite=True)


#%%
def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
