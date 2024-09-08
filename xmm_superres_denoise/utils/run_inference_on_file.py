from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Tuple
from warnings import warn

import matplotlib.pyplot as plt
# import onnxruntime as onnxruntime
import torch
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval
from matplotlib import colormaps

from data.utils import (
    apply_transform,
    load_det_mask,
    load_fits,
    reshape_img_to_res,
)
from models import Model
from transforms import Crop, Normalize
from transforms.totensor import ToTensor
from utils.filehandling import (
    read_yaml,
    write_xmm_file_to_fits_wcs,
)


def _infer_from_ckpt(
    checkpoint_path: Path,
    model_config: dict,
    img: torch.Tensor,
    lr_shape: Tuple[int, int],
    hr_shape: Tuple[int, int],
) -> torch.Tensor:
    model = Model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=model_config,
        lr_shape=lr_shape,
        hr_shape=hr_shape,
        loss=None,
        metrics=None,
    )
    with torch.no_grad():
        model.eval()
        model.freeze()
        return model(img[None, None, :].to(model.device))


def _infer_from_onnx(img: torch.Tensor, checkpoint_path: Path) -> torch.Tensor:
    ort_session = onnxruntime.InferenceSession(checkpoint_path.as_posix())
    ort_outs = ort_session.run(
        ["name"], {"input": img[None, None, :].detach().cpu().numpy()}
    )
    output = ort_outs[0][0][0]
    return output


def run_on_file(
    fits_file: Path, checkpoint: Path, out: Path, run_config: Path, plot: bool
):
    if not fits_file.exists():
        raise FileNotFoundError(f"File {fits_file} not found!")

    out.mkdir(parents=True, exist_ok=True)

    run_config: dict = read_yaml(run_config)
    dataset_config: dict = run_config["dataset"]
    model_config: dict = run_config["model"]
    model_config.update(
        read_yaml(Path("res") / "configs" / "model" / f"{model_config['name']}.yaml")
    )
    model_config["batch_size"] = dataset_config["batch_size"]

    in_name, pred_name = _infer_file(
        fits_file=fits_file,
        dataset_config=dataset_config,
        checkpoint_path=checkpoint,
        out_path=out,
        model_config=model_config,
    )

    if plot:
        with fits.open(out / f"{in_name}.fits.gz") as f1, fits.open(
            out / f"{pred_name}.fits.gz"
        ) as f2:
            img_in = f1[0].data
            img_out = f2[0].data
        norm = ImageNormalize(img_in, interval=PercentileInterval(99.5))
        plt.imshow(
            img_in,
            norm=norm,
            cmap=colormaps["plasma"],
            origin="lower",
            interpolation="nearest",
        )
        plt.savefig(out / "plot_in.png")
        norm = ImageNormalize(img_out, interval=PercentileInterval(99.5))
        plt.imshow(
            img_out,
            norm=norm,
            cmap=colormaps["plasma"],
            origin="lower",
            interpolation="nearest",
        )
        plt.savefig(out / "plot_out.png")


def _infer_file(
    fits_file: Path,
    dataset_config: dict,
    checkpoint_path: Path,
    out_path: Path,
    model_config: dict,
) -> Tuple[str, str]:
    """
    Purpose:
        Run SR or DN inference on an input FITS file with real XMM-Newton image
    Inputs:
        fits_file - str,
            input FITS file absolute path
        dataset_config - dict
            The configuration for running the inference (this will dictate if it's SR or DN model)
    Outputs:
        input_filename - str,
            The absolute path to the saved input file
        predicted_filename - str,
            The absolute path to the saved predicted file

    Notes:
        * The input FITS file must be in detector coordinates **with good WCS** and shape (403,411),
        ideally made with xmmsas_tools.make_detxy_image.py
    """
    # raise a warning if the exposure (ONTIME) in the inpt fitsfile is outside 20 ks +/- 5ks
    hdr = fits.getheader(fits_file)
    ontime = hdr["EXPOSURE"] / 1000.0  # in ks
    if ontime >= 25.0 or ontime <= 15.0:
        warn(
            f"The networks were trained on 20 ks exposure images, "
            f"the exposure time of the input image is {ontime:.2f}ks."
        )
    else:
        print(f"Info: the exposure time of the input image is {ontime:.2f} ks.")

    # Load and prepare the image
    loaded = load_fits(fits_file)
    det_mask = load_det_mask(1)
    img = loaded["img"]
    img = img * det_mask
    img = reshape_img_to_res(dataset_lr_res=416, img=img, res_mult=1)

    transform = [Crop(crop_p=1.0, mode=dataset_config["crop_mode"]), ToTensor()]  # TODO
    normalize = Normalize(
        lr_max=dataset_config["lr"]["max"],
        hr_max=dataset_config["hr"]["max"],
        stretch_mode=dataset_config["scaling"],
    )

    img = apply_transform(img, transform)
    img = normalize.normalize_lr_image(img)

    # Load model and infer on file
    if checkpoint_path.suffix == ".onnx":
        output = _infer_from_onnx(img, checkpoint_path=checkpoint_path)
    elif checkpoint_path.suffix == ".ckpt":
        output = _infer_from_ckpt(
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            img=img,
            lr_shape=(dataset_config["lr"]["res"], dataset_config["lr"]["res"]),
            hr_shape=(dataset_config["hr"]["res"], dataset_config["hr"]["res"]),
        )
        output = output.squeeze()
    else:
        raise ValueError

    in_denorm = normalize.denormalize_lr_image(img)
    out_denorm = normalize.denormalize_hr_image(output)

    # Save input and predicted
    input_out_name = f"{fits_file.stem}_input_wcs"
    predict_out_name = input_out_name.replace("input", "predict")

    res_mult = out_denorm.shape[0] // in_denorm.shape[0]
    # input, padded
    write_xmm_file_to_fits_wcs(
        img=in_denorm.detach().cpu().numpy(),
        output_dir=out_path,
        source_file_name=loaded["file_name"],
        res_mult=1,
        exposure=loaded["exp"],
        comment="Input image padded and WCS aligned. Needs to be multiplied by exposure.",
        out_file_name=input_out_name,
        in_header=loaded["header"],
    )
    # output
    write_xmm_file_to_fits_wcs(
        img=out_denorm.detach().cpu().numpy(),
        output_dir=out_path,
        source_file_name=loaded["file_name"],
        res_mult=res_mult,
        exposure=dataset_config["hr"]["exp"],
        comment=f"XMM {model_config['name']} model prediction. Needs to be multiplied by exposure. It's possible that"
        f"the given exposure is not correctly calculated so take care.",
        out_file_name=predict_out_name,
        in_header=loaded["header"],
    )
    return input_out_name, predict_out_name


if __name__ == "__main__":
    parser = ArgumentParser(description="Predict XMM SR or DN image")
    parser.add_argument(
        "--fits",
        type=Path,
        help="The FITS filename in detxy coordinates with WCS",
        required=True,
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Path to the checkpoint file", required=True
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Path where the created files should be stored",
        required=True,
    )
    parser.add_argument("--run_config", type=Path, required=True)
    parser.add_argument("--plot", default=True, action=BooleanOptionalAction)
    args = parser.parse_args()

    run_on_file(
        fits_file=args.fits,
        checkpoint=args.checkpoint,
        out=args.out,
        run_config=args.run_config,
        plot=args.plot,
    )
