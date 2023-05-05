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
            correction: float = 0.0,
    ):
        super(_Metric, self).__init__()
        self.add_state("metric", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.scaling = scaling
        self.correction = correction

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        scaled = self.scaling * self.metric
        mean_scaled = scaled / self.total
        corrected = mean_scaled + self.correction
        return corrected

    def __repr__(self):
        scaling = f"{self.scaling} * " if self.scaling != 1.0 else ""
        name = {self.__class__.__name__}
        correction = f"+ {self.correction}" if self.correction != 0.0 else ""
        return f"{scaling}{name}{correction}"


class VIF(_Metric):
    """
    Compute Visual Information Fidelity in pixel domain for a batch of images.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += piq.vif_p(x=preds, y=target, reduction=reduction)
        self.total += preds.size()[0]


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
            reduction: str = "mean") -> None:
        self.metric += piq.ssim(x=preds, y=target, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
                                reduction=reduction, k1=self.k1, k2=self.k2)
        self.total += preds.size()[0]


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
            reduction: str = "mean") -> None:
        self.metric += piq.multi_scale_ssim(x=preds, y=target, kernel_size=self.kernel_size,
                                            kernel_sigma=self.kernel_sigma,
                                            reduction=reduction,
                                            k1=self.k1, k2=self.k2)
        self.total += preds.size()[0]


class PSNR(_Metric):
    """
    Compute Peak Signal-to-Noise Ratio for a batch of images.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += piq.psnr(x=preds, y=target, reduction=reduction)
        self.total += preds.size()[0]


class PoissonNLLLoss(_Metric):
    higher_is_better = False

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += poisson_nll_loss(input=preds, target=target, log_input=False, reduction=reduction)
        self.total += preds.size()[0]


class MSE(_Metric):
    """
    Measures the element-wise mean squared error.
    """
    higher_is_better = False

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += mse_loss(input=preds, target=target, reduction=reduction)
        self.total += preds.size()[0]


class MDSI(_Metric):
    """
    Compute Mean Deviation Similarity Index (MDSI) for a batch of images.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += piq.mdsi(x=preds, y=target, reduction=reduction)
        self.total += preds.size()[0]


class MAE(_Metric):
    """
    Function that takes the mean element-wise absolute value difference.
    """
    higher_is_better: bool = False

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += l1_loss(input=preds, target=target, reduction=reduction)
        self.total += preds.size()[0]


class HaarPSI(_Metric):
    """
    Compute Haar Wavelet-Based Perceptual Similarity Inputs
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += piq.haarpsi(x=preds, y=target, reduction=reduction)
        self.total += preds.size()[0]


class GMSD(_Metric):
    """
    Compute Gradient Magnitude Similarity Deviation
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += piq.gmsd(x=preds, y=target, reduction=reduction)
        self.total += preds.size()[0]


class MultiScaleGMSD(_Metric):
    """
    Computation of Multi scale GMSD.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += piq.multi_scale_gmsd(x=preds, y=target, chromatic=False,
                                            reduction=reduction)
        self.total += preds.size()[0]


class FSIM(_Metric):
    """
    Compute Feature Similarity Index Measure for a batch of images.
    """

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean") -> None:
        self.metric += piq.fsim(x=preds, y=target, chromatic=False, reduction=reduction)
        self.total += preds.size()[0]
