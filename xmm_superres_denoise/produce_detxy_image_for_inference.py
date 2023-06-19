#
# Using XMM SAS xmmsas_tools in xmm_superres_denoise.utils
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

from argparse import ArgumentParser
from pathlib import Path

from xmm_superres_denoise.utils.xmmsas_tools import (
    filter_events_gti,
    get_pps_nxsa,
    make_detxy_image,
    make_gti_pps,
)


def get_detxy_for_obs_id(
    obs_id: str,
    obs_dir: Path,
    proc_dir: Path,
    pps_dir: Path,
    instrument: str,
    max_expo: int,
):
    proc_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{max_expo}ks.fits" if max_expo > 0 else "full.fits"
    pps_files = get_pps_nxsa(obs_id, w_dir=obs_dir)
    png_name = f"{instrument}_gti_{suffix.replace('fits', 'png')}"
    gtis = make_gti_pps(
        pps_files=pps_files,
        instrument=instrument,
        out_dir=proc_dir,
        max_expo=max_expo,
        plot_it=True,
        save_plot=png_name,
    )

    if len(gtis) < 1:
        raise RuntimeError(f"No GTIs found.")

    evl = None
    pn_gti = None
    # select the PN event list in the PPS
    for file in pps_files["evl_files"]:
        if "PN" in file.stem.upper():
            evl = file
            break

    for file in gtis:
        if "PN" in file.stem.upper():
            pn_gti = file
            break

    evl_filtered = filter_events_gti(
        evl,
        pn_gti,
        pps_files=pps_files,
        verbose=True,
        output_name=f"{instrument}_cleaned_evl_{suffix}",
        w_dir=obs_dir,
    )

    if evl_filtered is None:
        raise RuntimeError("No filtered event lists found!")

    det_xy = make_detxy_image(
        event_list=evl_filtered,
        pps_dir=pps_dir,
        pps_files=pps_files,
        low_energy=500,
        high_energy=2000,
        bin_size=80,
        radec_image=False,
        verbose=True,
        output_name=f"{instrument}_500_2000_detxy_image_{suffix}",
        w_dir=obs_dir,
    )

    return det_xy


if __name__ == "__main__":
    parser = ArgumentParser(description="Create det_xy image for given observation")
    parser.add_argument("obsid", type=str, help="The OBS_ID to process")
    parser.add_argument(
        "--wdir",
        type=Path,
        default=Path.cwd(),
        help="The working top folder name, must exist",
    )
    parser.add_argument(
        "--expo_time",
        type=float,
        default=20,
        help="Will extract only this exposure time (in ks) from the event list. "
        "Set it to negative to use the GTI one.",
    )
    args = parser.parse_args()

    get_detxy_for_obs_id(
        obs_id=args.obsid,
        obs_dir=args.wdir / args.obsid,
        proc_dir=args.wdir / args.obsid / "proc",
        pps_dir=args.wdir / args.obsid / "pps",
        instrument="all",
        max_expo=args.expo_time,
    )
