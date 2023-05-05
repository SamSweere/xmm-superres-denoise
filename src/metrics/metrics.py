import piq
import torch
import torch.nn as nn
from torchmetrics import Metric


class _Metric(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(
            self,
            fn,
            scaling: float = 1.0,
            correction: float = 0.0,
    ):
        super(_Metric, self).__init__()
        self.add_state("metric", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.scaling = scaling
        self.correction = correction
        self.fn = fn

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor) -> None:
        self.metric += self.fn(x=preds, y=target)
        self.total += preds.size()[0]

    def compute(self) -> torch.Tensor:
        scaled = self.scaling * self.metric
        mean_scaled = scaled / self.total
        corrected = mean_scaled + self.correction
        return corrected

    def __repr__(self):
        scaling = f"{self.scaling} * " if self.scaling != 1.0 else ""
        correction = f" + {self.correction}" if self.correction != 0.0 else ""
        return f"{scaling}{self.fn}{correction}"


class VIF(_Metric):
    """
    Compute Visual Information Fidelity in pixel domain for a batch of images.
    """

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = piq.VIFLoss()
        super(VIF, self).__init__(fn=fn, scaling=scaling, correction=correction)


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
        fn = piq.SSIMLoss(kernel_size=kernel_size, kernel_sigma=kernel_sigma, k1=k1, k2=k2)
        super(SSIM, self).__init__(fn=fn, scaling=scaling, correction=correction)


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
            k2: float = 0.05,
            scale_weights: torch.Tensor = None
    ):
        fn = piq.MultiScaleSSIMLoss(kernel_size=kernel_size, kernel_sigma=kernel_sigma, k1=k1, k2=k2,
                                    scale_weights=scale_weights)
        super(MultiScaleSSIM, self).__init__(fn=fn, scaling=scaling, correction=correction)


class PSNR(_Metric):
    """
    Compute Peak Signal-to-Noise Ratio for a batch of images.
    """

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = piq.psnr
        super(PSNR, self).__init__(fn=fn, scaling=scaling, correction=correction)


class PoissonNLLLoss(_Metric):
    higher_is_better = False

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = nn.PoissonNLLLoss(log_input=False)
        super(PoissonNLLLoss, self).__init__(fn=fn, scaling=scaling, correction=correction)


class MSE(_Metric):
    """
    Measures the element-wise mean squared error.
    """
    higher_is_better = False

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = nn.MSELoss()
        super(MSE, self).__init__(fn=fn, scaling=scaling, correction=correction)


class MDSI(_Metric):
    """
    Compute Mean Deviation Similarity Index (MDSI) for a batch of images.
    """

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = piq.MDSILoss()
        super(MDSI, self).__init__(fn=fn, scaling=scaling, correction=correction)


class MAE(_Metric):
    """
    Function that takes the mean element-wise absolute value difference.
    """
    higher_is_better: bool = False

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = nn.L1Loss()
        super(MAE, self).__init__(fn=fn, scaling=scaling, correction=correction)


class HaarPSI(_Metric):
    """
    Compute Haar Wavelet-Based Perceptual Similarity Inputs
    """

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = piq.HaarPSILoss()
        super(HaarPSI, self).__init__(fn=fn, scaling=scaling, correction=correction)


class GMSD(_Metric):
    """
    Compute Gradient Magnitude Similarity Deviation
    """

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = piq.GMSDLoss()
        super(GMSD, self).__init__(fn=fn, scaling=scaling, correction=correction)


class MultiScaleGMSD(_Metric):
    """
    Computation of Multi scale GMSD.
    """

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = piq.MultiScaleGMSDLoss(chromatic=False)
        super(MultiScaleGMSD, self).__init__(fn=fn, scaling=scaling, correction=correction)


class FSIM(_Metric):
    """
    Compute Feature Similarity Index Measure for a batch of images.
    """

    def __init__(
            self,
            scaling: float = 1.0,
            correction: float = 0.0
    ):
        fn = piq.FSIMLoss(chromatic=False)
        super(FSIM, self).__init__(fn=fn, scaling=scaling, correction=correction)
