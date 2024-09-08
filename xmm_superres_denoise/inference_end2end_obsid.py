#
# Example script to run the inference and predict super-resolution (SR) or denoised (DN) XMM EPIC-pn image
#
# Using XMM SAS xmmsas_tools in xmm_superres_denoise.utils
#
# The workflow is the following:
#
# 1. Download the PPS files for an OBSID from the XMM archive
# 1. Generate GTI files based on PPS flaring background and threshold
# 1. Generate a new GTI with max 20ks exposure
# 1. Filter the PPS event lists with the GTI
# 1. Using the cleaned event lists, generate DETX,DETY image in a given energy band
# 1. Run the inference on this image and produce the predicted SR and DN images

import argparse
from pathlib import Path

from produce_detxy_image_for_inference import get_detxy_for_obs_id
from utils.run_inference_on_file import run_on_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict XMM SR or DN image")
    parser.add_argument("obsid", type=str, help="The OBS_ID to process")
    parser.add_argument("checkpoint", type=Path, help="Path to the checkpoint file")
    parser.add_argument("--run_config", type=Path, required=True)
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

    # obs_id = '0852030101' # M51 (from paper, Fig. 7 bottom)
    obs_id = args.obsid
    expo_time = args.expo_time
    w_dir: Path = args.wdir
    if not w_dir.exists():
        raise NotADirectoryError(f"Directory {w_dir} does not exist!")

    obs_dir = w_dir / obs_id
    pps_dir = obs_dir / "pps"
    proc_dir = obs_dir / "proc"

    det_xy = get_detxy_for_obs_id(
        obs_id=obs_id,
        obs_dir=obs_dir,
        proc_dir=proc_dir,
        pps_dir=pps_dir,
        instrument="pn",
        max_expo=expo_time,
    )

    # Run inference
    run_on_file(
        fits_file=det_xy,
        checkpoint=args.checkpoint,
        out=obs_dir,
        run_config=args.run_config,
        plot=True,
    )
