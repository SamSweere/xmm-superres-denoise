from typing import Dict, Union

from torchmetrics import (
    MeanAbsoluteError,
    Metric,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from xmm_superres_denoise.metrics import PoissonNLLLoss, VGGLoss

# The scaling and the scaled loss functions
# These are based on randomly initialized untrained models
_zero_epoch = {
    "linear": {
        "l1": 0.05746,
        "poisson": 0.3323,
        "psnr": 22.189,
        "ssim": 0.3856,
        "ms_ssim": 0.6093,
    },
    "sqrt": {
        "l1": 0.1573,
        "poisson": 0.5002,
        "psnr": 14.761,
        "ssim": 0.1362,
        "ms_ssim": 0.5425,
    },
    "asinh": {
        "l1": 0.2573,
        "poisson": 2.801,
        "psnr": 10.464,
        "ssim": 0.06155,
        "ms_ssim": 0.2081,
    },
    "log": {
        "l1": 0.3528,
        "poisson": 3.167,
        "psnr": 7.817,
        "ssim": 0.05174,
        "ms_ssim": 0.3088,
    },
}

# Epoch 38 (training was not completely stable for all runs, probably due to the ADAM optimiser) of the runs:
# silver-butterfly0=-101, graceful-flower-100, cool-universe-99 and stellar-cloud-98
_last_epoch = {
    "linear": {
        "l1": 0.02097,
        "poisson": 0.1804,
        "psnr": 30.565,
        "ssim": 0.7218,
        "ms_ssim": 0.96,
    },
    "sqrt": {
        "l1": 0.05374,
        "poisson": 0.4187,
        "psnr": 22.977,
        "ssim": 0.4621,
        "ms_ssim": 0.874,
    },
    "asinh": {
        "l1": 0.08037,
        "poisson": 0.5223,
        "psnr": 19.52,
        "ssim": 0.3662,
        "ms_ssim": 0.8258,
    },
    "log": {
        "l1": 0.1072,
        "poisson": 0.6567,
        "psnr": 16.838,
        "ssim": 0.3446,
        "ms_ssim": 0.7982,
    },
}


def _get_scaling(x1, x2, y1=1.0, y2=0.0):
    # Based on linear formula y=ax+b
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    return a, b


def create_loss(
    data_scaling,
    loss_config: Dict[str, Union[int, dict]],
) -> Metric:
    l1_p = loss_config["l1"]
    poisson_p = loss_config["poisson"]
    psnr_p = loss_config["psnr"]
    ssim_p = loss_config["ssim"]
    ms_ssim_p = loss_config["ms_ssim"]

    vgg_config = loss_config["vgg"]
    vgg_p = vgg_config["p"]
    total = l1_p + poisson_p + psnr_p + ssim_p + ms_ssim_p + vgg_p
    if not 0.0 < total <= 1.0:
        raise ValueError(
            f"Sum of l1_p={l1_p}, poisson_p={poisson_p}, psnr_p={psnr_p}, ssim_p={ssim_p}, "
            f"ms_ssim_p={ms_ssim_p}, vgg_p={vgg_p} has to be in ]0.0, 1.0] but is {total}!"
        )

    metrics = []

    if l1_p > 0.0:
        scaling, correction = _get_scaling(
            x1=_zero_epoch[data_scaling]["l1"], x2=_last_epoch[data_scaling]["l1"]
        )
        metrics.append(l1_p * scaling * MeanAbsoluteError() + correction)

    if poisson_p > 0.0:
        scaling, correction = _get_scaling(
            x1=_zero_epoch[data_scaling]["poisson"],
            x2=_last_epoch[data_scaling]["poisson"],
        )
        m = PoissonNLLLoss(scaling=poisson_p * scaling, correction=correction)
        metrics = m if metrics is None else metrics + m

    if psnr_p > 0.0:
        scaling, correction = _get_scaling(
            x1=_zero_epoch[data_scaling]["psnr"], x2=_last_epoch[data_scaling]["psnr"]
        )
        metrics.append(psnr_p * scaling * PeakSignalNoiseRatio() + correction)

    if ssim_p > 0.0:
        scaling, correction = _get_scaling(
            x1=_zero_epoch[data_scaling]["ssim"], x2=_last_epoch[data_scaling]["ssim"]
        )
        metrics.append(
            ssim_p
            * scaling
            * StructuralSimilarityIndexMeasure(kernel_size=13, sigma=2.5, k2=0.05)
            + correction
        )

    if ms_ssim_p > 0.0:
        scaling, correction = _get_scaling(
            x1=_zero_epoch[data_scaling]["ms_ssim"],
            x2=_last_epoch[data_scaling]["ms_ssim"],
        )
        metrics.append(
            ms_ssim_p
            * scaling
            * MultiScaleStructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            )
            + correction
        )

    if vgg_p:
        m = VGGLoss(
            scaling=vgg_p,
            vgg_model=vgg_config["vgg_model"],
            batch_norm=vgg_config["batch_norm"],
            layers=vgg_config["layers"],
        )
        metrics = m if metrics is None else metrics + m

    final_metric = metrics[0]
    for i in range(1, len(metrics)):
        final_metric = final_metric + metrics[i]

    return final_metric
