#!/usr/bin/env python
# coding: utf-8

# # Processing XMM-Newton observations starting from the pipeline produced products (PPS)
#
# 1. Download the PPS (only the FTZ files) and extract it in a folder=<obsid> and a subfolder=pps (default structure of the XSA downloaded tar file with PPS.)
# 2. Derive the good-time-intervals (GTI)
#   2a. using the PPS flaring background file *FBKTSR* and the PPS derived threshold kept
#     in the *FBKTSR* header keyword FLCUTTHR, method='pps'
#   2b. using the 'standard' procedure with high energy events E>10 keV and nominal thresholds per MOS and PN, method='sas'
#   2c. using the 'standard' procedure of using high energy events only and threshold derived by my procedure (mode + scale*MAD), method='mine'
# 3. Filter the PPS event list for the GTI, FLAG and PATTERN. Note that the GTI file prodused with method will be used in the filtering.
#
# ## Modification history
#
# * Created: 12 Jul 2021, Ivan Valtchanov
# * Adapted for Abell 85 observations

#%%


import glob
import math
import os
import sys
import tarfile
from contextlib import contextmanager

import matplotlib as mpl
import numpy as np
from astropy.io import fits
# from astropy import wcs
from astropy.table import Table
from astroquery.esa.xmm_newton import XMMNewton as xmm
from pysas.wrapper import Wrapper
from scipy import stats

mpl.use("Agg")
import matplotlib.pylab as plt

# import argparse

# globals
home = os.path.expanduser("~")
xinst = {"EMOS1": "m1", "EMOS2": "m2", "EPN": "pn"}

#%%
# redirect temporarily the stdout, useful for pysas Wrapper output
#
# taken from answer #29 https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
# https://stackoverflow.com/a/14707227
#
@contextmanager
def stdout_redirected(new_stdout):
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout


#%%
def get_pps_nxsa(obsid, wdir=None, skip=True, keeptar=False):
    """
    Download PPS files from the XMM archive, only the FTZ files.

    Parameters
    ----------

    obsid: str,
        The XMM OBS_ID for the PPS download
    wdir: str,
        If None, then will use the current folder, else it will be the folder where the PPS tar file will be saved to.
    skip: bool, default True
        If folder 'wdir/obsid/pps' already exists, then the download will be skipped.
    keeptar: bool, default False
        If the downloaded tar.gz file is to be kept in `wdir`.
    verbose: bool, default False
        If verbose info is to be printed on stdout

    Returns
    -------

    The path to the pps folder 'wdir/obsid/pps'

    Method
    ------

    Will use `astroquery.esa.xmm_newton` package to download only the FTZ files at level PPS

    """
    #
    if not os.path.isdir(wdir):
        print(
            f"Warning! The working dir {wdir} does not exist. Will use the current dir."
        )
        wdir = os.getcwd()
    #
    #
    # check if subfolder pps already exists
    #
    ppsdir = f"{wdir}/{obsid}/pps"
    if os.path.isdir(ppsdir):
        if skip:
            print(
                f"Will skip PPS download for {obsid} as {ppsdir} already exists and skip flag is {skip}"
            )
            return ppsdir
        else:
            print(
                f"Warning! Found an already existing folder {ppsdir} and skip flag is {skip} => files will be overwritten!"
            )
    #
    pps_tar_file = f"{wdir}/{obsid}_PPS_nxsa"
    # due to a bug (or feature) in xmm_newton the tar file name appends .tar at the end
    xmm.download_data(obsid, level="PPS", extension="FTZ", filename=pps_tar_file)
    pps_tar_file = f"{wdir}/{obsid}_PPS_nxsa.tar"
    print(f"Extracting {pps_tar_file}")
    if not tarfile.is_tarfile(pps_tar_file):
        print(
            f"Downloaded file from NXSA {pps_tar_file} does not look like tar file. Cannot continue!"
        )
        return None
    #
    with tarfile.open(pps_tar_file, "r") as tar:
        tar.extractall(path=wdir)
    #
    if not keeptar:
        os.remove(pps_tar_file)
    return ppsdir


#%%
def gen_gti_pps(event_list, out_dir=None, plot_it=False, save_plot=None):
    """

    Will use the threshold from the PPS flared background file (*FBKTSR*) header and save the GTI file in outdir

    Parameters
    ----------

    event_list: str,
        The full path to the event list filename
    out_dir: str, default None
        Where to save the GTI file, if None then will use the same folder as the `event_list`
    plot_it: bool, default False
        If to plot the GTI intervals
    save_plot: str, default None
        The PNG filename where to save the plot, if None, no saving.

    Returns
    -------

    The GTI filename with full path

    Method
    ------

    Will find the flaring background light curve (pattern FBKTSR in filename) in the folder where the input event list is and then read the header where
    the keyword 'FLCUTTHR' encodes the pipeline-derived flaring background threshold, all time intervals there the lightcurve is below this interval will
    be included in the GTI file

    """
    pps_dir = os.path.dirname(event_list)
    xname = os.path.basename(event_list)[0:13]
    fbk_file = glob.glob(f"{pps_dir}/{xname}*FBKTSR*.FTZ")
    if len(fbk_file) < 1:
        print(f"Flaring background PPS file {xname}*FBKTSR* not found in {pps_dir}")
        return None
    #
    with fits.open(fbk_file[0]) as hdu:
        inst = hdu[0].header["INSTRUME"]
        if inst not in ["EMOS1", "EMOS2", "EPN"]:
            print(f"Cannot do instrument {inst}")
            return None
        obsid = hdu[0].header["OBS_ID"]
        x = hdu["RATE"].data["TIME"]
        y = hdu["RATE"].data["RATE"]
        if "FLCUTTHR" not in hdu["RATE"].header.keys():
            print(
                f"Cannot find threshold FLCUTTHR for {obsid}: {inst}. Cannot use method='pps'"
            )
            return None
        else:
            rate_lim = hdu["RATE"].header["FLCUTTHR"]
    #
    #
    # run gtigen
    #
    if out_dir is None:
        out_dir = pps_dir
    gti_name = f"{out_dir}/{xinst[inst]}_pps.gti"
    args = [
        f"table={fbk_file[0]}",
        f"expression=RATE<={rate_lim}",
        f"gtiset={gti_name}",
    ]
    p = Wrapper("tabgtigen", args)
    #
    logout = gti_name + ".log"
    with open(logout, "w") as f:
        with stdout_redirected(f):
            p.run()
    #
    with fits.open(gti_name, mode="update") as hdu:
        hdu["STDGTI"].header["METHOD"] = (
            "pps",
            "Method used to derive the rate threshold",
        )
        hdu["STDGTI"].header["RLIM"] = (rate_lim, "The PPS derived threshold")
    #
    if plot_it:
        #
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(x, y, label="GTI", zorder=0)
        ax.axhline(
            rate_lim,
            color="red",
            linewidth=3,
            linestyle="dashed",
            label=f"GTI threshold {rate_lim:.2f} cts/s",
            zorder=2,
        )
        ax.set_xlabel("Relative time (s)")
        ax.set_ylabel("Count-rate (cts/s)")
        ax.grid()
        ax.legend()
        ax.set_title(f"{obsid}: {inst}")
        if save_plot is not None:
            plt.savefig(save_plot, dpi=100)
            # plt.show()
            plt.close()
        else:
            plt.close()
    return gti_name


#%%
def gen_gti_sas(
    event_list,
    method="sas",
    time_bin=100.0,
    scale=2.0,
    rate_curve=None,
    out_dir=None,
    plot_it=False,
    save_plot=None,
):
    """

    Will use the SAS recommended way to filter for high background periods and create a GTI file.

    Parameters
    ----------

    event_list: str,
        The full path to the event list filename
    method: str, default 'sas'
        If `method='sas'` then it will create light curve and will use the default values of the threshold 0.35 for MOS and 0.40 for PN. If it is not 'sas' then
        it will use my method of cderiving the threshold based on the mode and the median-absolute-deviation:
        `threshold = mode + `scale` * MAD`
    time_bin: float, in seconds, default 100.0 s
        If time bin to use to build the light curve
    scale: float, default 2.0,
        The scale to use for the threshold derivation (only used if `method != 'sas'`)
    rate_curve: str, default None,
        If None then will save the rate curve with a generic name '<instrument>_rate.fits', else then will use this filename
        to save the rate curve in the same folder as the `event_list`
    out_dir: str, default None,
        If None then will use the same folder as the `event_list` to save the files.
    plot_it: bool, default False,
        If to produce a plot of the rate curve and the used thresholds for the good-time-intervals.
    save_plot: str, default None
        The PNG filename where to save the plot, if None, no saving.

    Returns
    -------

    The GTI filename with full path


    """
    if not os.path.isfile(event_list):
        print(f"Event list {event_list} not found.")
        return None
    #
    pps_dir = os.path.dirname(event_list)
    #
    hdr = fits.getheader(event_list, 0)
    inst = hdr["INSTRUME"]
    obsid = hdr["OBS_ID"]
    #
    if "EMOS1" in inst or "EMOS2" in inst:
        expr = "#XMMEA_EM && (PI>10000) && (PATTERN==0)"
    elif "EPN" in inst:
        expr = "#XMMEA_EP && (PI>10000&&PI<12000) && (PATTERN==0)"
    else:
        print(f"Cannot build rate curve for instrument {inst}")
        return None
    #
    #
    if out_dir is None:
        out_dir = pps_dir
    if rate_curve is None:
        # use dfault name
        rate_curve = f"{pps_dir}/{xinst[inst]}_rate.fits"
    else:
        rate_curve = f"{pps_dir}/{rate_curve}"
    #
    args = [
        f"table={event_list}",
        "withrateset=Y",
        f"rateset={rate_curve}",
        "maketimecolumn=Y",
        f"timebinsize={time_bin}",
        "makeratecolumn=Y",
        f"expression={expr}",
    ]
    #
    p = Wrapper("evselect", args)
    logout = rate_curve + ".log"
    with open(logout, "w") as f:
        with stdout_redirected(f):
            p.run()
    #
    # now generate the GTI
    #
    gti_name = f"{out_dir}/{xinst[inst]}_{method}.gti"
    #
    if method == "sas":
        if "MOS" in inst:
            rate_lim = 0.35
        else:
            rate_lim = 0.40
    else:
        with fits.open(rate_curve) as hdu:
            x = hdu["RATE"].data["TIME"]
            y = hdu["RATE"].data["RATE"]
            xmode = stats.mode(y, axis=None)
            xmad = stats.median_abs_deviation(y, axis=None)
            rate_lim = xmode.mode[0] + scale * xmad
    #
    args = [f"table={rate_curve}", f"expression=RATE<={rate_lim}", f"gtiset={gti_name}"]
    p = Wrapper("tabgtigen", args)
    logout = gti_name + ".log"
    with open(logout, "w") as f:
        with stdout_redirected(f):
            p.run()
    #
    with fits.open(gti_name, mode="update") as hdu:
        hdu["STDGTI"].header["METHOD"] = (
            method,
            "Method used to derive the rate threshold",
        )
        hdu["STDGTI"].header["RLIM"] = (rate_lim, "The derived rate threshold")
    if plot_it:
        with fits.open(rate_curve) as hdu:
            x = hdu["RATE"].data["TIME"]
            y = hdu["RATE"].data["RATE"]
        #
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(x, y, label="GTI", zorder=0)
        ax.axhline(
            rate_lim,
            color="red",
            linewidth=3,
            linestyle="dashed",
            label=f"GTI threshold {rate_lim:.2f} cts/s",
            zorder=2,
        )
        if method != "sas":
            #
            # show the cannonical thresholds
            #
            if "MOS" in inst:
                ax.axhline(
                    0.35,
                    color="magenta",
                    linewidth=2,
                    linestyle="dotted",
                    label=f"Nominal threshold 0.35 cts/s",
                    zorder=3,
                )
            else:
                ax.axhline(
                    0.40,
                    color="magenta",
                    linewidth=2,
                    linestyle="dotted",
                    label=f"Nominal threshold 0.40 cts/s",
                    zorder=3,
                )
        #
        ax.set_xlabel("Relative time (s)")
        ax.set_ylabel("Count-rate (cts/s)")
        ax.grid()
        ax.legend()
        ax.set_title(f"{obsid}: {inst}")
        if save_plot is not None:
            plt.savefig(save_plot, dpi=100)
            # plt.show()
            plt.close()
        else:
            plt.close()
    return gti_name


#%%


def filter_events_gti(event_list, gti_input, output_name=None, verbose=False):
    """
    Using an input event list, filter them with the input GTI and save to a new cleaned event list.

    Parameters
    ----------

    event_list: str,
        The full path to the event list filename
    gti_input: str,
        The input GTI file
    output_name: str, default None,
        If name of the cleaned event list, it will be save in the same folder as the input one. If None, will use a generic name
        of this pattern: `<instrument>_<method>_clean.evl' where <method> will be read from the GTI file header.
    verbose: bool, default False,
        If to produce some verbose info
    Returns
    -------

    The full path to the cleaned event list
    """
    #
    # check if the input files exist
    if not os.path.isfile(event_list):
        print(f"Input event list {event_list} not found.")
        return None
    #
    if not os.path.isfile(gti_input):
        print(f"GTI file {gti_input} not found.")
        return None
    #
    out_dir = os.path.dirname(event_list)
    #
    # read some keywords for housekeeping
    hdr = fits.getheader(event_list, "EVENTS")
    inst = hdr["INSTRUME"]
    obsid = hdr["OBS_ID"]
    ontime0 = hdr["ONTIME"]
    #
    # read some keywords for housekeeping from the GTI file, method and threshold
    hdr2 = fits.getheader(gti_input, "STDGTI")
    rlim = hdr2["RLIM"]
    method = hdr2["METHOD"]
    #
    # mapping the instrument names
    xinst = {"EMOS1": "m1", "EMOS2": "m2", "EPN": "pn"}
    #
    # now filtering the event lists with the GTI
    #
    if "EMOS" in inst:
        expr = f"#XMMEA_EM && gti({gti_input},TIME) && (PI>150) && (PATTERN <= 12)"
    elif "EPN" in inst:
        expr = f"(RAWY > 12) && (FLAG == 0 || FLAG & 0x10000 != 0) && (PATTERN in [0:4]) &&  gti({gti_input},TIME)"
        # expr = f"(FLAG == 0) && gti({gti_input},TIME) && (PI>150) && (PATTERN <= 4)"
    #
    if output_name is None:
        output_name = f"{out_dir}/{xinst[inst]}_{method}_clean.evl"
    #
    args = [
        f"table={event_list}",
        "withfilteredset=Y",
        f"filteredset={output_name}",
        "destruct=Y",
        "keepfilteroutput=Y",
        f"expression={expr}",
    ]
    #
    p = Wrapper("evselect", args)
    logout = output_name + ".log"
    with open(logout, "w") as f:
        with stdout_redirected(f):
            p.run()
    #
    if verbose:
        hdr1 = fits.getheader(output_name, "EVENTS")
        ontime1 = hdr1["ONTIME"]
        print(
            f"Input event list on-time {ontime0:.1f} s, \n filtered list on-time {ontime1:.1f} s, \n good time fraction {(100*ontime1/ontime0):.1f} %"
        )
    #
    return output_name


#
#%%
def split_event_list(
    event_list, gti_file=None, expo=20.0, output_filename=None, verbose=False
):
    """
    Extract a sublist of `expo` ks from an input `event_list`

    Parameters
    ----------
        event_list: str,
            The event list filename
        gti_file: str, optional,
            The name of the GTI file to be used to skip masked periods
        expo: float, in kseconds
            The extracted  sublist will have this exposure time, *taking into account the GTI*
        verbose: bool, default False,
            Print some verbose info
    """
    if not os.path.isfile(event_list):
        print(f"Cannot find the input event list {event_list}")
        return None
    proc_dir = os.path.dirname(event_list)
    expo_s = expo * 1000.0  # in seconds
    #
    with fits.open(event_list) as hdu:
        hdr = hdu["EVENTS"].header
        # within +/- 15% of the required expo
        ontime = hdu["EVENTS"].header["ONTIME"]  # in seconds
        if (ontime < expo_s * 1.15) and (ontime > expo_s * 0.85):
            print(
                f"No sublist extraction as the input filtered event list has exposure time is with +/-15% of {expo} ks"
            )
            return event_list
        else:
            t0 = hdu["EVENTS"].data["TIME"].min()
            t1 = t0 + expo * 1000.0  # in seconds
            expr = f"TIME in [{t0}:{t1}]"
    #
    # now, taking into account the GTI
    #
    if gti_file is not None and os.path.isfile(gti_file):
        #
        # will read it as Table (no units needed)
        qt = Table.read(gti_file)
        qt["DELTAT"] = qt["STOP"] - qt["START"]
        # sort in reverse order of time GTI, i.e. largest GTI first
        qt.sort("DELTAT")
        qt.reverse()
        qt["CUMTIME"] = np.cumsum(qt["DELTAT"])
        ilim = (
            len(np.where(qt["CUMTIME"] <= expo_s)[0]) + 1
        )  # add done more to include time range
        tx = qt[0:ilim]
        #
        print(tx)
        expr = ""
        for i in range(ilim):
            ct = tx["CUMTIME"][i]
            if ct <= expo_s:
                if "TIME" in expr:
                    expr += f' || (TIME IN [{tx["START"][i]}:{tx["STOP"][i]}])'
                else:
                    expr = f'(TIME IN [{tx["START"][i]}:{tx["STOP"][i]}])'
            else:
                if "TIME" in expr:
                    expr += f' || (TIME IN [{tx["START"][i]}:{tx["START"][i] + (ct - expo_s)}])'
                else:
                    expr = f'(TIME IN [{tx["START"][i]}:{tx["START"][i] + expo_s}])'
                break
    if verbose:
        print(f"Will filter on time with this expression: {expr}")
    #
    if output_filename is None:
        output_filename = f"extracted_{expo}ks.evl"
    args = [
        f"table={event_list}",
        "withfilteredset=Y",
        f"filteredset={output_filename}",
        "destruct=Y",
        "keepfilteroutput=Y",
        f"expression={expr}",
    ]
    #
    p = Wrapper("evselect", args)
    logout = output_filename + ".log"
    with open(logout, "w") as f:
        with stdout_redirected(f):
            p.run()
    if verbose:
        hdr = fits.getheader(output_filename, extname="EVENTS")
        print(f'Filtered ONTIME for {expo_s} s: {hdr["ONTIME"]} seconds')

    return output_filename


#%%
#
def make_detxy_images(
    event_list,
    low_e=500,
    high_e=2000,
    binsize=80,
    out_prefix=None,
    expo_image=False,
    mask_image=False,
    cr_image=False,
    verbose=True,
    skip=True,
):
    """
    Make images in detector coordinates from event list

    Parameters
    ----------

    event_list: str,
        The full path to the event list filename
    low_e: int, default 500 eV,
        The low energy (PI) of the events to consider, in eV
    high_e: int, default 2000 eV,
        The high energy (PI) of the events to consider, in eV
    binsize: int, default 80,
        The pixel size of the uotput image, it is binsize*0.05 arcsec
    out_prefix: str, default None,
        The prefix for the output FITS filename, if None it will be generic
    expo_image: bool, default False,
        If an exposure image is to be produced.
    mask_image: bool, default False,
        If a detector mask image is to be produced, if True then this implies `expo_image`=True
    cr_image: bool, default False,
        If a count-rate image is to be produced, implies `expo_image`=True
    verbose: bool, default False,
        If to produce some verbose info
    skip: bool, default True,
        Will skip if files with matching names already exist

    Returns
    -------

    A dict:
        {'detxy_image': None, 'detxy_expo_image': None, 'detxy_mask_image': None, 'detxy_cr_image': None}

    """
    #
    proc_dir = os.path.dirname(event_list)
    if out_prefix is None:
        outpr = os.path.splitext(event_list)[0]
    else:
        outpr = f"{proc_dir}/{out_prefix}"
    #
    if mask_image or cr_image:
        expo_image = True
    #
    image_detxy_name = f"{outpr}_detxy_{low_e}_{high_e}_{binsize}.fits"
    expo_detxy_name = f"{outpr}_detxy_expo_{low_e}_{high_e}_{binsize}.fits"
    mask_detxy_name = f"{outpr}_detxy_mask_{low_e}_{high_e}_{binsize}.fits"
    cr_detxy_name = f"{outpr}_detxy_cr_{low_e}_{high_e}_{binsize}.fits"
    #
    output_dict = {
        "detxy": None,
        "expo_detxy": None,
        "mask_detxy": None,
        "cr_detxy": None,
    }
    #
    # image in DETXY coordinates
    #
    if verbose:
        print(
            f'Generating DETXY image in band [{low_e},{high_e}] eV with {binsize*0.05}"/pixel'
        )
    #
    if os.path.isfile(image_detxy_name) and skip:
        print(
            f"Found an already existing image in DETX,DETY and skip is {skip}, so will skip then."
        )
    else:
        xargs = [
            f"table={event_list}",
            "xcolumn=DETX",
            "ycolumn=DETY",
            "imagebinning=binSize",
            f"ximagebinsize={binsize}",
            f"yimagebinsize={binsize}",
            f"expression=(PI in [{low_e}:{high_e}])",
            "squarepixels=yes",
            f"withimageset=true",
            f"imageset={image_detxy_name}",
        ]
        #
        logfile = image_detxy_name.replace(".fits", ".log")
        p = Wrapper("evselect", xargs)
        with open(logfile, "w") as f:
            with stdout_redirected(f):
                p.run()
        #
        # add proper WCS which SAS currenty does not do
        #
        status = update_detxy_wcs(image_detxy_name, binsize)
        #
        # now add an ERROR extension (neede in some cases for data analysis)
        #
        with fits.open(image_detxy_name, mode="update") as hdu:
            # adding Poissonian errors as an extension to the image file
            err_image = np.sqrt(hdu[0].data)
            hdu_err = fits.ImageHDU(data=err_image, header=hdu[0].header, name="ERROR")
            hdu.append(hdu_err)
    output_dict["detxy"] = image_detxy_name
    #
    if expo_image:
        if os.path.isfile(expo_detxy_name) and skip:
            print(
                f"Found an already existing exposure map in DETX,DETY and skip is {skip}, so will skip then."
            )
        else:
            if verbose:
                print(
                    f'Generating expo image in band [{low_e},{high_e}] eV with {binsize*0.05}"/pixel'
                )
            atthk = glob.glob(f"{proc_dir}/*ATT*")[0]
            #
            args = [
                f"imageset={image_detxy_name}",
                f"attitudeset={atthk}",
                f"eventset={event_list}",
                f"expimageset={expo_detxy_name}",
                f"pimin={low_e}",
                f"pimax={high_e}",
                "withdetcoords=Y",
            ]
            logfile = expo_detxy_name.replace(".fits", ".log")
            p = Wrapper("eexpmap", args)
            with open(logfile, "w") as f:
                with stdout_redirected(f):
                    p.run()
    output_dict["expo_detxy"] = expo_detxy_name
    #
    if mask_image:
        if os.path.isfile(mask_detxy_name) and skip:
            print(
                f"Found an already existing detector mask in DETX,DETY and skip is {skip}, so will skip then."
            )
        else:
            if verbose:
                print(
                    f"Generating detector mask image from exposure map {expo_detxy_name}"
                )
            args = [
                f"expimageset={expo_detxy_name}",
                f"detmaskset={mask_detxy_name}",
                "threshold1=0.3",
                "threshold2=0.5",
            ]
            logfile = mask_detxy_name.replace(".fits", ".log")
            p = Wrapper("emask", args)
            with open(logfile, "w") as f:
                with stdout_redirected(f):
                    p.run()
    output_dict["mask_detxy"] = mask_detxy_name
    #
    if cr_image:
        if os.path.isfile(cr_detxy_name) and skip:
            print(
                f"Found an already existing count-rate image in DETX,DETY and skip is {skip}, so will skip then."
            )
        else:
            hq = make_cr_image(
                image_detxy_name,
                expo_detxy_name,
                cr_filename=cr_detxy_name,
                overwrite=True,
            )
    output_dict["cr_detxy"] = cr_detxy_name
    #
    if verbose:
        print("make_detxy_images terminated")
    return output_dict


#
#
#%%
#
def make_sky_images(
    event_list,
    low_e=500,
    high_e=2000,
    binsize=80,
    out_prefix=None,
    expo_image=False,
    mask_image=False,
    cr_image=False,
    verbose=True,
    skip=True,
):
    """
    Make images in SKY coordinates from event list

    Parameters
    ----------

    event_list: str,
        The full path to the event list filename
    low_e: int, default 500 eV,
        The low energy (PI) of the events to consider, in eV
    high_e: int, default 2000 eV,
        The high energy (PI) of the events to consider, in eV
    binsize: int, default 80,
        The pixel size of the uotput image, it is binsize*0.05 arcsec
    out_prefix: str, default None,
        The prefix for the output FITS filename, if None it will be generic
    expo_image: bool, default False,
        If an exposure image is to be produced.
    mask_image: bool, default False,
        If a detector mask image is to be produced, if True then this implies `expo_image`=True
    cr_image: bool, default False,
        If a count-rate image is to be produced, implies `expo_image`=True
    verbose: bool, default False,
        If to produce some verbose info
    skip: bool, default True,
        Will skip if files with matching names already exist

    Returns
    -------

    A dict:
        {'detxy_image': None, 'detxy_expo_image': None, 'detxy_mask_image': None, 'detxy_cr_image': None}

    """
    #
    proc_dir = os.path.dirname(event_list)
    if out_prefix is None:
        outpr = os.path.splitext(event_list)[0]
    else:
        outpr = f"{proc_dir}/{out_prefix}"
    #
    if mask_image or cr_image:
        expo_image = True
    #
    image_xy_name = f"{outpr}_xy_{low_e}_{high_e}_{binsize}.fits"
    expo_xy_name = f"{outpr}_xy_expo_{low_e}_{high_e}_{binsize}.fits"
    mask_xy_name = f"{outpr}_xy_mask_{low_e}_{high_e}_{binsize}.fits"
    cr_xy_name = f"{outpr}_xy_cr_{low_e}_{high_e}_{binsize}.fits"
    #
    output_dict = {"xy": None, "expo_xy": None, "mask_xy": None, "cr_xy": None}
    #
    # image in XY (SKY) coordinates
    #
    if verbose:
        print(
            f'Generating SKY image in band [{low_e},{high_e}] eV with {binsize*0.05}"/pixel'
        )
    #
    if os.path.isfile(image_xy_name) and skip:
        print(
            f"Found an already existing image in X,Y and skip is {skip}, so will skip then."
        )
    else:
        xargs = [
            f"table={event_list}",
            "xcolumn=X",
            "ycolumn=Y",
            "imagebinning=binSize",
            f"ximagebinsize={binsize}",
            f"yimagebinsize={binsize}",
            f"expression=(PI in [{low_e}:{high_e}])",
            "squarepixels=yes",
            f"withimageset=true",
            f"imageset={image_xy_name}",
        ]
        #
        logfile = image_xy_name.replace(".fits", ".log")
        p = Wrapper("evselect", xargs)
        with open(logfile, "w") as f:
            with stdout_redirected(f):
                p.run()
        #
        # now add an ERROR extension (neede in some cases for data analysis)
        #
        with fits.open(image_xy_name, mode="update") as hdu:
            # adding Poissonian errors as an extension to the image file
            err_image = np.sqrt(hdu[0].data)
            hdu_err = fits.ImageHDU(data=err_image, header=hdu[0].header, name="ERROR")
            hdu.append(hdu_err)
    output_dict["xy"] = image_xy_name
    #
    if expo_image:
        if os.path.isfile(expo_xy_name) and skip:
            print(
                f"Found an already existing exposure map in X,Y and skip is {skip}, so will skip then."
            )
        else:
            if verbose:
                print(
                    f'Generating expo SKY image in band [{low_e},{high_e}] eV with {binsize*0.05}"/pixel'
                )
            atthk = glob.glob(f"{proc_dir}/*ATT*")[0]
            #
            args = [
                f"imageset={image_xy_name}",
                f"attitudeset={atthk}",
                f"eventset={event_list}",
                f"expimageset={expo_xy_name}",
                f"pimin={low_e}",
                f"pimax={high_e}",
                "withdetcoords=N",
            ]
            logfile = expo_xy_name.replace(".fits", ".log")
            p = Wrapper("eexpmap", args)
            with open(logfile, "w") as f:
                with stdout_redirected(f):
                    p.run()
    output_dict["expo_xy"] = expo_xy_name
    #
    if mask_image:
        if os.path.isfile(mask_xy_name) and skip:
            print(
                f"Found an already existing detector mask in X,Y and skip is {skip}, so will skip then."
            )
        else:
            if verbose:
                print(
                    f"Generating detector mask image from exposure map {expo_xy_name}"
                )
            args = [
                f"expimageset={expo_xy_name}",
                f"detmaskset={mask_xy_name}",
                "threshold1=0.3",
                "threshold2=0.5",
            ]
            logfile = mask_xy_name.replace(".fits", ".log")
            p = Wrapper("emask", args)
            with open(logfile, "w") as f:
                with stdout_redirected(f):
                    p.run()
    output_dict["mask_xy"] = mask_xy_name
    #
    if cr_image:
        if os.path.isfile(cr_xy_name) and skip:
            print(
                f"Found an already existing count-rate image in X,Y and skip is {skip}, so will skip then."
            )
        else:
            hq = make_cr_image(
                image_xy_name, expo_xy_name, cr_filename=cr_xy_name, overwrite=True
            )
    output_dict["cr_xy"] = cr_xy_name
    #
    if verbose:
        print("make_sky_images terminated")
    return output_dict


#
#%%
def update_detxy_wcs(
    detxy_image_filename, binsize, mode="update", outname=None, verbose=False
):
    """Update the FITS file in detector coordinates with proper RA-DEC WCS keywords

    Parameters
    ----------
        detxy_image_filename: str,
            The FITS file name for the image in detector coordinates
        mode: str, default 'update'
            The file will be changed in place if `mode='update'` else it will save it under outname or
            return then HDU if outname is None.
        outname: str or None,
            If str and mode is not 'update' then this will be used as a filename to save the result
            If None, then the HDU will be returned (no saving)
        verbose: bool, default False,
            If to print verbose info

    Returns
    -------
        HDU in all cases

    Method
    ------
        Will use `ecoordconv` with X,Y from header REFXCRPIX,REFYCRPX and get the IMAGE_X, IMAGE_Y pixel coordinates and the corresponding RA, DEC
    """
    #
    if not os.path.isfile(detxy_image_filename):
        print(
            f"Cannot find the FITS file with the image in detector coordinates: {detxy_image_filename}"
        )
        return None
    #
    if verbose:
        print("\t Running ecoordconv")
    #
    hdr = fits.getheader(detxy_image_filename)
    x = hdr["REFXCRPX"]
    y = hdr["REFYCRPX"]
    ra = hdr["REFXCRVL"]
    dec = hdr["REFYCRVL"]
    logfile = detxy_image_filename.replace(".fits", "_ecoordconv.log")
    xargs = [f"imageset={detxy_image_filename}", f"x={x}", f"y={y}", "coordtype=pos"]
    p = Wrapper("ecoordconv", xargs)
    with open(logfile, "w") as f:
        with stdout_redirected(f):
            p.run()
    #
    with open(logfile, "r") as f:
        lines = f.readlines()
    #
    for iline in lines:
        if "IM_X:" in iline:
            q = iline.split()
            xima = q[2]
            yima = q[3]
    #     if ('DEC:' in iline):
    #         q = iline.split()
    #         ra = q[2]
    #         dec = q[3]
    #
    # print (xima,yima,ra,dec)
    #
    if verbose:
        print("\t Update the header with a new WCS")
    if mode == "update":
        with fits.open(detxy_image_filename, mode="update") as hdu:
            header = hdu[0].header
            # Create a new WCS object.  The number of axes must be set
            # from the start
            header["CRVAL1"] = float(ra)
            header["CRVAL2"] = float(dec)
            header["CRPIX1"] = float(xima)
            header["CRPIX2"] = float(yima)
            cdelt1 = binsize * header["REFYCDLT"]
            cdelt2 = -binsize * header["REFXCDLT"]
            header["CDELT1"] = cdelt1
            header["CDELT2"] = cdelt2
            header["CTYPE1"] = "RA---TAN"
            header["CTYPE2"] = "DEC--TAN"
            #
            # rotation
            #
            crota2 = 90.0 - float(header["PA_PNT"])
            header["CROTA2"] = crota2
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
            header["COMMENT"] = "WCS added by IvanV"
    else:
        hdu = fits.open(detxy_image_filename)
        header = hdu[0].header
        # Create a new WCS object.  The number of axes must be set
        # from the start
        header["CRVAL1"] = float(ra)
        header["CRVAL2"] = float(dec)
        header["CRPIX1"] = float(xima)
        header["CRPIX2"] = float(yima)
        cdelt1 = binsize * header["REFYCDLT"]
        cdelt2 = -binsize * header["REFXCDLT"]
        header["CDELT1"] = cdelt1
        header["CDELT2"] = cdelt2
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        #
        # rotation
        #
        crota2 = 90.0 - float(header["PA_PNT"])
        header["CROTA2"] = crota2
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
        header["COMMENT"] = "WCS added by IvanV"
        if outname is not None:
            hdu.writeto(outname, overwrite=True)
        return hdu
    return detxy_image_filename


#%%


def make_cr_image(infile, expo, cr_filename=None, overwrite=False):
    """Make a count-rate image by dividing the counts image by the exposure map

    Parameters
    ----------
        infile: str
            The input FITS file with the counts image
        expo: str or float
            The input exposure map, must have the same shape as the image
            If scalar then it is the exposure time in seconds (constant) and will be used to derive the count-rate
        cr_filename: str or None
            The filename to save the counts-rate image to. If None, no saving but the HDU list will be returned

    Returns
    -------
        FITS HDU list, optionally if `cr_filename` is set then saving the FITS HDU list to that file.

    Notes
    -----
        If the infile containe extension `ERROR` then it will be also calculated
    """
    if not os.path.isfile(infile):
        print(f"Error! Input image FITS file {infile} not fuond. Cannot continue")
        return None
    #
    have_err = False
    rate_err = None
    #
    with fits.open(infile) as hdu:
        image = hdu[0].data
        header = hdu[0].header
        try:
            err = hdu["ERROR"].data
            herr = hdu["ERROR"].header
            have_err = True
        except:
            have_err = False
    #
    #
    if not os.path.isfile(expo):
        if expo.isdigit():
            print(f"Exposure time: {expo} seconds")
            texpo = float(expo)
            rate = np.divide(image, texpo)
            if have_err:
                rate_err = np.divide(err, texpo)
        else:
            print(
                f"Error! Parameter expo={expo} is neither a FITS file with exposure map nor a scalar exposure time."
            )
            print("Cannot make count-rate image")
            return None
    else:
        with fits.open(expo) as expo_hdu:
            expo_map = expo_hdu[0].data
        #
        # check if shapes match
        #
        if np.shape(image) == np.shape(expo_map):
            rate = np.true_divide(image, expo_map, where=expo_map > 0.0)
            if have_err and (np.shape(err) == np.shape(expo_map)):
                rate_err = np.true_divide(err, expo_map, where=expo_map > 0.0)
        else:
            print(
                f"Input image and exposure map shapes do not match: {np.shape(image)} and {np.shape(expo_map)}"
            )
            return None
    #
    hdu = fits.PrimaryHDU(header=header)
    hdu_img = fits.ImageHDU(data=rate, name="CR_IMAGE", header=header)
    hdul = fits.HDUList([hdu, hdu_img])
    if have_err and (rate_err is not None):
        hdu_err = fits.ImageHDU(data=rate_err, header=herr, name="ERROR")
        hdul = fits.HDUList([hdu, hdu_img, hdu_err])
    #
    if cr_filename is not None:
        print(hdul.info())
        hdul.writeto(cr_filename, overwrite=overwrite)
    return hdul
