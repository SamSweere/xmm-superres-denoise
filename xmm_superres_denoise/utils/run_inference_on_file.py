from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Tuple
from warnings import warn

import matplotlib.pyplot as plt
import onnxruntime as onnxruntime
import torch
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval
from astropy.visualization.stretch import SqrtStretch
from matplotlib import colormaps
import numpy as np 

from xmm_superres_denoise.datasets.utils import (
    apply_transform,
    load_det_mask,
    load_fits,
    reshape_img_to_res,
)
from xmm_superres_denoise.models import Model
from xmm_superres_denoise.transforms import Crop, Normalize
from xmm_superres_denoise.transforms.totensor import ToTensor
from xmm_superres_denoise.utils.filehandling import (
    read_yaml,
    write_xmm_file_to_fits_wcs,
)
import pandas as pd
import os


def _infer_from_ckpt(
    checkpoint_path: Path,
    model_config: dict,
    img: torch.Tensor,
    lr_shape: Tuple[int, int],
    hr_shape: Tuple[int, int],
) -> torch.Tensor:
    
    loss_config: dict = read_yaml(Path("res") / "configs" / "loss_functions.yaml")
    
    model = Model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=model_config,
        lr_shape=lr_shape,
        hr_shape=hr_shape,
        loss=None,
        loss_config = loss_config , 
        metrics=None,
        extended_metrics=None,
        in_metrics=None,
        in_extended_metrics=None,
    )
    with torch.no_grad():
        model.eval()
        model.freeze()
        return model(img[None, None, :].to(model.device))


def _infer_from_onnx(img: torch.Tensor, checkpoint_path: Path) -> torch.Tensor:
    ort_session = onnxruntime.InferenceSession(checkpoint_path.as_posix())
    # add extra dimension to the input to allow using the old onnx file 
    img = torch.stack((img, torch.zeros_like(img)))
    ort_outs = ort_session.run(
    ["output"], {"input": img[ None, :].detach().cpu().numpy()}
    )
    output = ort_outs[0][0][0]
    return torch.tensor(output)


def run_on_file(
    fits_file: Path, checkpoint: Path, out: Path, run_config: Path, plot: bool, clamping_tr_nn: str = None, shift_amount: int = None
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
    model_config["H_in"] = dataset_config["lr"]["res"]
    model_config["W_in"] = dataset_config["lr"]["res"]
    model_config["clamp"] = dataset_config["clamp"]

    in_name, pred_name = _infer_file(
        fits_file=fits_file,
        dataset_config=dataset_config,
        checkpoint_path=checkpoint,
        out_path=out,
        model_config=model_config,
        clamping_tr_nn = clamping_tr_nn, 
        shift_amount = shift_amount,
    )


    if plot:
        with fits.open(out / f"{in_name}.fits.gz") as f1, fits.open(
            out / f"{pred_name}.fits.gz"
        ) as f2:
            img_in = f1[0].data
            img_out = f2[0].data

        norm = ImageNormalize(img_in, stretch = SqrtStretch())
        # norm = ImageNormalize(img_in, interval=PercentileInterval(99.5))
        plt.imshow(
            img_in,
            norm=norm,
            # cmap=colormaps["plasma"],
            cmap = 'viridis',
            origin="lower",
            interpolation="None",
        )
        plt.savefig(out / f"{in_name}_plot_in.pdf")
        norm = ImageNormalize(img_out, stretch = SqrtStretch())
        # norm = ImageNormalize(img_out, interval=PercentileInterval(99.5))
        plt.imshow(
            img_out,
            norm=norm,
            # cmap=colormaps["plasma"],
            cmap = 'viridis',
            origin="lower",
            interpolation="None",
        )
        plt.savefig(out / f"{pred_name}_plot_out.pdf")

def compute_statistics(img):


    maxes = np.max(img)
    means = np.mean(img)
    variances = np.var(img)
    quantile_values = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999]
    quantiles = np.quantile(img, np.array(quantile_values)).reshape(1, 8)

    # Create a dictionary with fractions as keys and quantiles as values
    # quantiles_dict = dict(zip(map(str, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]), quantiles))

    # Make dataframe from maxes, means, and variances
    df = pd.DataFrame({"Maxes": [maxes], "Means": [means], "Variances": [variances]})
   
    # Make quantiles dataframe
    columns = ['quantile_' + str(quantile_value) for quantile_value in quantile_values]
    df_2d = pd.DataFrame(quantiles, columns= columns)
    
    # Combine quantile and other dataframe
    result_df = pd.concat([df, df_2d], axis=1)
    
    return result_df
            


def _infer_file(
    fits_file: Path,
    dataset_config: dict,
    checkpoint_path: Path,
    out_path: Path,
    model_config: dict,
    clamping_tr_nn: str = None, 
    shift_amount: int = None,
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
    img = reshape_img_to_res(dataset_lr_res=256, img=img, res_mult=1)

    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # fig.savefig('test.pdf')

    

    # Compute the clamping value 
    lr_statistics = compute_statistics(img)


    transform = [Crop(crop_p=1.0, mode=dataset_config["crop_mode"]), ToTensor()]  # TODO
    normalize = Normalize(
        lr_max=dataset_config["lr"]["max"],
        hr_max=dataset_config["hr"]["max"],
        config = dataset_config,
        lr_statistics=lr_statistics,
        stretch_mode=dataset_config["scaling"],
        clamp = dataset_config["clamp"],
        sigma_clamp = dataset_config["sigma_clamp"],
        quantile_clamp = dataset_config["quantile_clamp"],
    )

   
    img = apply_transform(img, transform)  
    img = normalize.normalize_lr_image(img, idx = 0)
    
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # fig.savefig('test12.pdf')

    

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

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(img)
    # ax[0].set_title("input")
    # ax[1].imshow(output)
    # ax[1].set_title("output")

    # fig.savefig('clamping_test_quantile.pdf')

    in_denorm = normalize.denormalize_lr_image(img, idx = 0) # --> Isn't that just the image before applying the norm?
    out_denorm = normalize.denormalize_hr_image(output, idx = 0)

    
   
    # Save input and predicted
    input_out_name = f"{clamping_tr_nn}_{ Path(fits_file.stem).stem}_input_wcs"
    # input_out_name = f"{fits_file.stem}_input_wcs"
    predict_out_name = input_out_name.replace("input", "predict")

    
    plt.close()

    res_mult = out_denorm.shape[0] // in_denorm.shape[0]


   
    # fig, ax = plt.subplots()
    # ax.imshow(in_denorm)
    # fig.savefig('test2.pdf')

    

    write_xmm_file_to_fits_wcs(
        img=in_denorm.detach().cpu().numpy()*hdr["EXPOSURE"],
        # img=in_denorm.detach().cpu().numpy(),
        output_dir=out_path,
        source_file_name=loaded["file_name"],
        res_mult=1,
        exposure=loaded["exp"],
        comment="Input image padded and WCS aligned. Needs to be multiplied by exposure.",
        out_file_name=input_out_name,
        in_header=loaded["header"],
    )
    # output
    #TODO: multplication is hardcoded, just assuming a five times increase in exposure!
    write_xmm_file_to_fits_wcs(
        img=out_denorm.detach().cpu().numpy()*hdr["EXPOSURE"]*5,
        # img=out_denorm.detach().cpu().numpy(),
        output_dir=out_path,
        source_file_name=loaded["file_name"],
        res_mult=res_mult,
        exposure=dataset_config["hr"]["exp"]*1000, # convert to seconds
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
