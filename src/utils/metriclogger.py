from collections import defaultdict
from typing import Callable

import piq
import torch
import torch.nn as nn
from torch.nn.functional import l1_loss, mse_loss, poisson_nll_loss

from transforms import ImageUpsample


def _get_scores(
        x: torch.Tensor,
        pred: torch.Tensor,
        y: torch.Tensor,
        data_range,
        log_input=False,
        log_extended=False,
        reduction: str = "mean"
):
    x = x.to(pred.device)
    y = y.to(pred.device)

    kernel_size = 13
    kernel_sigma = 2.5
    k1 = 0.01
    k2 = 0.05

    # try:
    # Metrics calculated using package https://github.com/photosynthesis-team/piq
    metrics = {
        "psnr": piq.psnr(x=pred, y=y, data_range=data_range, reduction=reduction),
        "ssim": piq.ssim(x=pred, y=y, kernel_size=kernel_size, kernel_sigma=kernel_sigma, data_range=data_range,
                         reduction=reduction, k1=k1, k2=k2),
        "l1": l1_loss(input=pred, target=y, reduction=reduction),
        "l2": mse_loss(input=pred, target=y, reduction=reduction),
        "poisson": poisson_nll_loss(input=pred, target=y, log_input=False, reduction=reduction)
    }

    if pred.shape[-1] > 192:
        # We can only calculate ms_ssim if the size is bigger than 192 due to the different scales in ms_ssim
        metrics["ms_ssim"] = piq.multi_scale_ssim(x=pred, y=y, kernel_size=kernel_size, kernel_sigma=kernel_sigma,
                                                  data_range=data_range, k1=k1, k2=k2,
                                                  reduction=reduction)

    if log_extended:
        metrics["vif_p"] = piq.vif_p(x=pred, y=y, data_range=data_range, reduction=reduction)
        metrics["fsim"] = piq.fsim(x=pred, y=y, data_range=data_range, chromatic=False, reduction=reduction)
        metrics["gmsd"] = piq.gmsd(pred, y, data_range=data_range, reduction=reduction)
        metrics["ms_gmsd"] = piq.multi_scale_gmsd(pred, y, data_range=data_range, chromatic=False,
                                                  reduction=reduction)
        metrics["haarpsi"] = piq.haarpsi(x=pred, y=y, data_range=data_range, reduction=reduction)
        metrics["mdsi"] = piq.mdsi(x=pred, y=y, data_range=data_range, reduction=reduction)

    if log_input:
        # Relative psnr in comparison to the input
        metrics["psnr_in"] = piq.psnr(x=x, y=y, data_range=data_range, reduction=reduction)
        metrics["ssim_in"] = piq.ssim(x=x, y=y, kernel_size=kernel_size, kernel_sigma=kernel_sigma,
                                      data_range=data_range,
                                      reduction=reduction, k1=k1, k2=k2)
        metrics["l1_in"] = l1_loss(input=x, target=y, reduction=reduction)
        metrics["l2_in"] = mse_loss(input=x, target=y, reduction=reduction)
        metrics["poisson_in"] = poisson_nll_loss(input=x, target=y, log_input=False, reduction=reduction)

        if log_extended:
            metrics["vif_p_in"] = piq.vif_p(x=x, y=y, data_range=data_range, reduction=reduction)
            metrics["fsim_in"] = piq.fsim(x=x, y=y, data_range=data_range, chromatic=False,
                                          reduction=reduction)
            metrics["gmsd_in"] = piq.gmsd(x, y, data_range=data_range, reduction=reduction)
            metrics["ms_gmsd_in"] = piq.multi_scale_gmsd(x, y, data_range=data_range, chromatic=False,
                                                         reduction=reduction)
            metrics["haarpsi_in"] = piq.haarpsi(x=x, y=y, data_range=data_range, reduction=reduction)
            metrics["mdsi_in"] = piq.mdsi(x=x, y=y, data_range=data_range, reduction=reduction)

        if pred.shape[-1] > 192:
            metrics["ms_ssim_in"] = piq.multi_scale_ssim(x=x, y=y, kernel_size=kernel_size,
                                                         kernel_sigma=kernel_sigma,
                                                         data_range=data_range, k1=k1, k2=k2,
                                                         reduction=reduction)

    return metrics


class MetricsCalculator(Callable):
    def __init__(
            self,
            data_range,
            scaling_normalizers,
            lr_shape,
            hr_shape,
            normalize
    ):
        super(MetricsCalculator, self).__init__()

        self.data_range = data_range
        self.normalize = normalize
        self.scaling_normalizers = scaling_normalizers

        if lr_shape[-1] != hr_shape[-1]:
            # Create the upsample class
            self.upsample = ImageUpsample(scale_factor=int(hr_shape[-1] / lr_shape[-1]))
        else:
            self.upsample = nn.Identity()

    def __call__(
            self,
            lr: torch.Tensor,
            pred: torch.Tensor,
            true: torch.Tensor,
            prefix: str = "",
            batch: bool = True,
            log_inputs=False,
            log_extended=False
    ):
        return self.get_metrics(lr, pred, true, prefix, batch, log_inputs, log_extended)

    def get_metrics(
            self,
            lr: torch.Tensor,
            pred: torch.Tensor,
            true: torch.Tensor,
            prefix: str = "",
            batch: bool = True,
            log_inputs: bool = False,
            log_extended: bool = False
    ):
        lr = self.normalize.denormalize_lr_image(lr)
        pred = self.normalize.denormalize_hr_image(pred)
        true = self.normalize.denormalize_hr_image(true)

        lr = self.upsample(lr)

        # This should be the case anyway, but sometimes a small negative number gets through somehow
        # Maybe because of the upsample
        lr = torch.clamp(lr, min=0.0, max=self.data_range)
        pred = torch.clamp(pred, min=0.0, max=self.data_range)
        true = torch.clamp(true, min=0.0, max=self.data_range)

        # Depending on the batch size, one batch could contain images from different TNG sets
        # We need to take this into consideration while calculating the metrics
        # If all images are from the same set, then just take the mean of the scores
        # Otherwise, take the mean of the scores for the corresponding TNG set
        metrics = defaultdict(defaultdict)
        for scale_normalizer in self.scaling_normalizers:
            stretch_name = scale_normalizer.stretch_mode
            x_s = scale_normalizer.normalize_hr_image(lr)
            pred_s = scale_normalizer.normalize_hr_image(pred)
            y_s = scale_normalizer.normalize_hr_image(true)

            metrics[f"{prefix}{stretch_name}"].update(
                _get_scores(
                    x_s,
                    pred_s,
                    y_s,
                    data_range=1.0,
                    log_input=log_inputs,
                    log_extended=log_extended,
                    reduction="mean" if batch else "none"
                )
            )
        return metrics
