#
# Using XMM SAS xmmsas_tools in utils
#
# The workflow is the following:
#
# 1. Download the PPS files for an OBSID from the XMM archive
# 1. Generate GTI files based on PPS flaring background and threshold
# 1. Generate a new GTI with max 20ks exposure
# 1. Filter the PPS event lists with the GTI
# 1. Using the cleaned event lists, generate images in a given energy band
#
# Created Oct 2022, _Ivan Valtchanov_, XMM SOC
#

import os

from utils.xmmsas_tools import (check_pps_dir, filter_events_gti, get_pps_nxsa,
                                make_detxy_image, make_gti_pps)

# Set top_d to a folder where to download the PPS
top_d = os.getcwd()
if not os.path.isdir(top_d):
    print("Error: working folder {top_d} does not exist, cannot continue!")
    raise FileNotFoundError
#
obsid = "0852030101"  # M51 (from paper, Fig. 7 bottom)
pps_d = f"{top_d}/{obsid}/pps"
#
pps_files = get_pps_nxsa(obsid, wdir=top_d, skip=True, keeptar=False, verbose=True)
#
proc_d = f"{top_d}/{obsid}/proc"
if not os.path.isdir(proc_d):
    os.mkdir(proc_d)
#
os.chdir(proc_d)

pps_files = check_pps_dir(pps_dir=pps_d, verbose=True)

expo_time = 20  # ks
gtis = make_gti_pps(
    pps_d,
    instrument="all",
    out_dir=proc_d,
    max_expo=expo_time,
    plot_it=True,
    save_plot=f"all_gti_{expo_time}ks_figure.png",
)
if len(gtis) < 1:
    raise RuntimeError
# select the PN event list in the PPS
for ifile in pps_files["evl_files"]:
    if "PN" in ifile.upper():
        evl = ifile
        break
for ifile in gtis:
    if "PN" in ifile.upper():
        pn_gti = ifile
        break
#
evl_filt = filter_events_gti(
    evl,
    pn_gti,
    pps_dir=pps_d,
    output_name=f"pn_cleaned_evl_{expo_time}ks.fits",
    verbose=True,
)
if evl_filt is None:
    raise RuntimeError
#
e1 = 500
e2 = 2000
detxy = make_detxy_image(
    evl_filt,
    pps_dir=pps_d,
    output_name=f"pn_{e1}_{e2}_detxy_image_{expo_time}ks.fits",
    low_energy=e1,
    high_energy=e2,
    bin_size=80,
    radec_image=True,
    verbose=True,
)
if detxy is None:
    raise RuntimeError
#
print("Test done")
