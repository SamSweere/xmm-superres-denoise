from typing import Optional

from piq import psnr, multi_scale_ssim, ssim
from torch.nn.functional import l1_loss, poisson_nll_loss
from torchmetrics import Metric

from metrics import MAE, PoissonNLLLoss, PSNR, SSIM, MultiScaleSSIM

_loss = {
    "l1": l1_loss,
    "poisson": poisson_nll_loss,
    "psnr": psnr,
    "ssim": ssim,
    "ms_ssim": multi_scale_ssim
}

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
        l1_p: float = 0.0,
        poisson_p: float = 0.0,
        psnr_p: float = 0.0,
        ssim_p: float = 0.0,
        ms_ssim_p: float = 0.0
) -> Metric:
    total = l1_p + poisson_p + psnr_p + ssim_p + ms_ssim_p
    if not 0.0 < total <= 1.0:
        raise ValueError(f"Sum of l1_p={l1_p}, poisson_p={poisson_p}, psnr_p={psnr_p}, ssim_p={ssim_p}, "
                         f"ms_ssim_p={ms_ssim_p} has to be in ]0.0, 1.0] but is {total}!")

    metrics: Optional[Metric] = None

    if l1_p > 0.0:
        # L1 loss
        scaling, correction = _get_scaling(x1=_zero_epoch[data_scaling]["l1"], x2=_last_epoch[data_scaling]["l1"])
        metrics = MAE(scaling=l1_p * scaling, correction=correction)

    if poisson_p > 0.0:
        # Poisson loss
        scaling, correction = _get_scaling(x1=_zero_epoch[data_scaling]["poisson"],
                                           x2=_last_epoch[data_scaling]["poisson"])
        if metrics is None:
            metrics = PoissonNLLLoss(scaling=poisson_p * scaling, correction=correction)
        else:
            metrics = metrics + PoissonNLLLoss(scaling=poisson_p * scaling, correction=correction)

    if psnr_p > 0.0:
        # PSNR loss
        # psnr is a rising metric, therefore inverse the scale
        scaling, correction = _get_scaling(x1=_zero_epoch[data_scaling]["psnr"],
                                           x2=_last_epoch[data_scaling]["psnr"])
        if metrics is None:
            metrics = PSNR(scaling=psnr_p * scaling, correction=correction)
        else:
            metrics = metrics + PSNR(scaling=psnr_p * scaling, correction=correction)

    if ssim_p > 0.0 or ms_ssim_p > 0.0:
        # SSIM settings
        if ssim_p > 0.0:
            # ssim is a rising metric, therefore inverse the scale

            # SSIM loss
            scaling, correction = _get_scaling(x1=_zero_epoch[data_scaling]["ssim"],
                                               x2=_last_epoch[data_scaling]["ssim"])
            if metrics is None:
                metrics = SSIM(scaling=ssim_p * scaling, correction=correction)
            else:
                metrics = metrics + SSIM(scaling=ssim_p * scaling, correction=correction)

        if ms_ssim_p > 0.0:
            # MS_SSIM loss
            scaling, correction = _get_scaling(x1=_zero_epoch[data_scaling]["ms_ssim"],
                                               x2=_last_epoch[data_scaling]["ms_ssim"])
            if metrics is None:
                metrics = MultiScaleSSIM(scaling=ms_ssim_p * scaling, correction=correction)
            else:
                metrics = metrics + MultiScaleSSIM(scaling=ms_ssim_p * scaling, correction=correction)
    return metrics
