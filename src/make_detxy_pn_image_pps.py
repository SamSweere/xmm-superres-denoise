#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Make a EPIC-pn image in DETX, DETY coordinates for a given energy band and OBS_ID

The image is suitable to be used for XMm super-resolution and denoising inference.

Workflow:

1. Download the PPS 
2. Clean the pn event list for GTI using the PPS derived threshold
3. Extract a sub-list with user-provided exposure, we use 20 ks for the super-resolution and denoising inference
4. Make an image in DETX,DETY coordinates and add proper WCS with the help of `ecoordconv`

Notes:

Will use pysas.wrapper for XMM-SAS command-line execution, this has some limitations and a lot of functions from datasets/xmm_tools.py

@author: ivaltchanov
"""

import argparse
import glob
import os

from astropy.io import fits

from utils.xmm_tools import (filter_events_gti, gen_gti_pps, get_pps_nxsa,
                             make_detxy_images, split_event_list)

#%%
#
parser = argparse.ArgumentParser(
    description="XMMSAS generating an image for OBS_ID in DETX,DETY from PPS and GTI filtered event lists"
)
parser.add_argument("obsid", type=str, help="The OBS_ID to use, PPS will be downloaded")
parser.add_argument(
    "--low_pi",
    type=int,
    default=500,
    help="Low energy (Pulse Intensity, PI) of events to consider, in eV (integer), default 500 eV",
)
parser.add_argument(
    "--high_pi",
    type=int,
    default=2000.0,
    help="High energy (PI) of events to consider, in eV (integer), default 2000 eV",
)
parser.add_argument(
    "--expo_time",
    type=float,
    default=20.0,
    help="Will select a sublist of events amounting to this exposure time, in kiloseconds. Default 20 ks. If negative or 0 then no time selection.",
)
parser.add_argument(
    "--binSize",
    type=int,
    default=80,
    help="The image bin size (integer), default 80 ==> 4 arcsec pixel",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=".",
    help="Top level folder name from where to extract the PPS files",
)
parser.add_argument(
    "--pps_dir",
    type=str,
    default="",
    help="Folder where the PPS files are, if not set or folder does not exist then it will download them from the XMM archive",
)
#
args = parser.parse_args()
#%%
inst = "pn"
obsid = args.obsid
top_dir = args.save_dir
if args.pps_dir != "":
    pps_dir = os.path.join(top_dir, obsid, args.pps_dir)
else:
    pps_dir = os.path.join(top_dir, obsid, "pps")
#
if not os.path.isdir(top_dir):
    os.makedirs(top_dir, exist_ok=True)
else:
    print(f"INFO: save folder {top_dir} already exists")
#
pi0 = args.low_pi
pi1 = args.high_pi
bin_size = args.binSize
expo_time = args.expo_time
#
#%%
#
# 1. get PPS from the NXSA (only if --pps_dir is not set at call)
#
if not os.path.isdir(pps_dir):
    print("INFO: Downloading data from NXSA, please be patient...")
    pps_dir = get_pps_nxsa(obsid, wdir=top_dir)
else:
    print(
        f"INFO: No downloading from XSA, will use the PPS files in already existing folder {pps_dir}"
    )
#
# 2. Set SAS environment variables
#
f1 = glob.glob(f"{pps_dir}/*CALIND*")
if len(f1) < 1:
    print(f"CALINDEX CCF file not found in {pps_dir}. Cannot continue.")
    raise FileNotFoundError
ccf_file = f1[0]

os.environ["SAS_ODF"] = pps_dir  # type: ignore
os.environ["SAS_CCF"] = ccf_file
#
# 3. select the event list to use
#
# the event lists, select only the PN
# if more than one exposure then will pick up the one with largest ONTIME
evlists = glob.glob(f"{pps_dir}/*IEVL*.FTZ")
evl = None
if len(evlists) == 0:
    print(
        f"No calibrated PPS-produced event lists (*IEVL* pattern) found in folder {pps_dir}. Cannot continue."
    )
    raise FileNotFoundError
elif len(evlists) > 1:
    # print (f'Found more than one exposure for instrument {inst.upper()}, will select the one with largest ONTIME.')
    ontime_max = 0.0
    for ex in evlists:
        if "PN" in ex:
            hdr = fits.getheader(ex, extname="EVENTS")
            ontime = hdr["ONTIME"]
            if ontime >= ontime_max:
                ontime_max = ontime
                evl = ex
else:
    # only one found
    evl = evlists[0]
if evl is None:
    print("No PPS-produced event list for PN is available. Cannot continue.")
    raise FileNotFoundError
#
#
evl_cleaned_file = f"{pps_dir}/{inst}_cleaned.evl"
#
# 4. generate GTI using the PPS derived threshold for periods of high background
#
gti_file = gen_gti_pps(evl, out_dir=pps_dir, plot_it=False)
#
# 5. Filter the events
#
evl_clean = filter_events_gti(evl, gti_file, verbose=True, output_name=evl_cleaned_file)
#
# 6. Extract a sublist with the required exposure time, 20 ks is the one we used for SR and DN
#
if args.expo_time <= 0:
    # no sub-list is needed
    out = make_detxy_images(
        evl_cleaned_file,
        low_e=pi0,
        high_e=pi1,
        binsize=bin_size,
        expo_image=False,
        mask_image=False,
        cr_image=False,
        verbose=True,
        skip=False,
    )
else:
    #
    # sublist will be created
    evl_cleaned_expo = f"{pps_dir}/{inst}_cleaned_{expo_time:.1f}ks.evl"
    subevl = split_event_list(
        evl_cleaned_file,
        gti_file=gti_file,
        expo=expo_time,
        verbose=True,
        output_filename=evl_cleaned_expo,
    )
    out = make_detxy_images(
        subevl,
        low_e=pi0,
        high_e=pi1,
        binsize=bin_size,
        expo_image=False,
        mask_image=False,
        cr_image=False,
        verbose=True,
        skip=False,
    )
#
print(f"All done in {pps_dir}, {out}")
