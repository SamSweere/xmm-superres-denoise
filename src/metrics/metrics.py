from typing import Union

import piq
import torch
from torch.nn.functional import mse_loss, poisson_nll_loss, l1_loss
from torchmetrics import Metric


class _Metric(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        super(_Metric, self).__init__()
        self.add_state("metric", default=[], dist_reduce_fx="cat")
        self.scaling = scaling
        self.correction = correction

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        unscaled = torch.cat([values.flatten() for values in self.metric])
        scaled = self.scaling * unscaled + self.correction
        return torch.mean(scaled)

    def __repr__(self):
        return f"{self.scaling} * {self.__class__.__name__} + {self.correction}"


class VIF(_Metric):
    """
    Compute Visual Information Fidelity in pixel domain for a batch of images.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(piq.vif_p(x=preds, y=target, data_range=data_range, reduction=reduction))


class SSIM(_Metric):
    """
    Interface of Structural Similarity (SSIM) index.
    """

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0,
            kernel_size: int = 13,
            kernel_sigma: float = 2.5,
            k1: float = 0.01,
            k2: float = 0.05
    ):
        super(SSIM, self).__init__(scaling=scaling, correction=correction)
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(piq.ssim(x=preds, y=target, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
                                    data_range=data_range, reduction=reduction, k1=self.k1, k2=self.k2))


class MultiScaleSSIM(_Metric):
    """
    Interface of Multi-scale Structural Similarity (MS-SSIM) index.
    """

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0,
            kernel_size: int = 13,
            kernel_sigma: float = 2.5,
            k1: float = 0.01,
            k2: float = 0.05
    ):
        super(MultiScaleSSIM, self).__init__(scaling=scaling, correction=correction)

        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(piq.multi_scale_ssim(x=preds, y=target, kernel_size=self.kernel_size,
                                                kernel_sigma=self.kernel_sigma, data_range=data_range,
                                                reduction=reduction,
                                                k1=self.k1, k2=self.k2))


class PSNR(_Metric):
    """
    Compute Peak Signal-to-Noise Ratio for a batch of images.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(piq.psnr(x=preds, y=target, data_range=data_range, reduction=reduction))


class PoissonNLLLoss(_Metric):
    higher_is_better = False

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(poisson_nll_loss(input=preds, target=target, log_input=False, reduction=reduction))


class MSE(_Metric):
    """
    Measures the element-wise mean squared error.
    """
    higher_is_better = False

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(mse_loss(input=preds, target=target, reduction=reduction))


class MDSI(_Metric):
    """
    Compute Mean Deviation Similarity Index (MDSI) for a batch of images.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(piq.mdsi(x=preds, y=target, data_range=data_range, reduction=reduction))


class MAE(_Metric):
    """
    Function that takes the mean element-wise absolute value difference.
    """
    higher_is_better: bool = False

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(l1_loss(input=preds, target=target, reduction=reduction))


class HaarPSI(_Metric):
    """
    Compute Haar Wavelet-Based Perceptual Similarity Inputs
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(piq.haarpsi(x=preds, y=target, data_range=data_range, reduction=reduction))


class GMSD(_Metric):
    """
    Compute Gradient Magnitude Similarity Deviation
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(piq.gmsd(x=preds, y=target, data_range=data_range, reduction=reduction))


class MultiScaleGMSD(_Metric):
    """
    Computation of Multi scale GMSD.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(piq.multi_scale_gmsd(x=preds, y=target, data_range=data_range, chromatic=False,
                                                reduction=reduction))


class FSIM(_Metric):
    """
    Compute Feature Similarity Index Measure for a batch of images.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            data_range: Union[int, float] = 1.0,
            reduction: str = "mean") -> None:
        self.metric.append(piq.fsim(x=preds, y=target, data_range=data_range, chromatic=False, reduction=reduction))
