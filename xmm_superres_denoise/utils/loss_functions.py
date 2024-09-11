from typing import Dict, Union

from metrics import PoissonNLLLoss, VGGLoss
from torchmetrics import (
    MeanAbsoluteError,
    Metric,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


def create_loss(
    data_scaling: str,
    loss_config: Dict[str, Union[int, dict]],
) -> Metric:
    sc_dict = loss_config[data_scaling] if loss_config["use_scaling"] else None

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
        m = MeanAbsoluteError() * l1_p
        if sc_dict:
            scaling, correction = sc_dict["l1"]["scaling"], sc_dict["l1"]["correction"]
            m = m * scaling + correction
        metrics.append(m)

    if poisson_p > 0.0:
        m = PoissonNLLLoss() * poisson_p
        if sc_dict:
            scaling, correction = (
                sc_dict["poisson"]["scaling"],
                sc_dict["poisson"]["correction"],
            )
            m = m * scaling + correction
        metrics.append(m)

    if psnr_p > 0.0:
        m = PeakSignalNoiseRatio() * psnr_p
        if sc_dict:
            scaling, correction = (
                sc_dict["psnr"]["scaling"],
                sc_dict["psnr"]["correction"],
            )
            m = m * scaling + correction
        metrics.append(m)

    if ssim_p > 0.0:
        m = (
            StructuralSimilarityIndexMeasure(kernel_size=13, sigma=2.5, k2=0.05)
            * ssim_p
        )
        if sc_dict:
            scaling, correction = (
                sc_dict["ssim"]["scaling"],
                sc_dict["ssim"]["correction"],
            )
            m = m * scaling + correction
        metrics.append(m)

    if ms_ssim_p > 0.0:
        m = (
            MultiScaleStructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            )
            * ms_ssim_p
        )
        if sc_dict:
            scaling, correction = (
                sc_dict["ms_ssim"]["scaling"],
                sc_dict["ms_ssim"]["correction"],
            )
            m = m * scaling + correction
        metrics.append(m)

    if vgg_p > 0.0:
        m = (
            VGGLoss(
                vgg_model=vgg_config["vgg_model"],
                batch_norm=vgg_config["batch_norm"],
                layers=vgg_config["layers"],
            )
            * vgg_p
        )
        metrics.append(m)

    final_metric = metrics[0]
    for i in range(1, len(metrics)):
        final_metric = final_metric + metrics[i]

    return final_metric
