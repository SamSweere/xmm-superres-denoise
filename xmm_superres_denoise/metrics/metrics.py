import piq
import torch
from torch.nn.functional import mse_loss, poisson_nll_loss
from torchmetrics import Metric
from torchvision import transforms
from torchvision.models import VGG, vgg


class _Metric(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super(_Metric, self).__init__()
        self.add_state(
            "metric", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        return self.metric / self.total


class VIF(_Metric):
    """
    Compute Visual Information Fidelity in pixel domain for a batch of images.
    """

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.vif_p(x=preds, y=target, reduction=reduction)
        self.total += preds.size()[0]


class PoissonNLLLoss(_Metric):
    higher_is_better = False

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += poisson_nll_loss(
            input=preds, target=target, log_input=False, reduction=reduction
        )
        self.total += preds.size()[0]


class MDSI(_Metric):
    """
    Compute Mean Deviation Similarity Index (MDSI) for a batch of images.
    """

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.mdsi(x=preds, y=target, reduction=reduction)
        self.total += preds.size()[0]


class HaarPSI(_Metric):
    """
    Compute Haar Wavelet-Based Perceptual Similarity Inputs
    """

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.haarpsi(x=preds, y=target, reduction=reduction)
        self.total += preds.size()[0]


class GMSD(_Metric):
    """
    Compute Gradient Magnitude Similarity Deviation
    """

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.gmsd(x=preds, y=target, reduction=reduction)
        self.total += preds.size()[0]


class MultiScaleGMSD(_Metric):
    """
    Computation of Multi scale GMSD.
    """

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.multi_scale_gmsd(
            x=preds, y=target, chromatic=False, reduction=reduction
        )
        self.total += preds.size()[0]


class FSIM(_Metric):
    """
    Compute Feature Similarity Index Measure for a batch of images.
    """

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.fsim(x=preds, y=target, chromatic=False, reduction=reduction)
        self.total += preds.size()[0]


class VGGLoss(_Metric):
    def __init__(
        self,
        vgg_model: str = "vgg19",
        batch_norm: bool = False,
        layers: int = 8,
    ):
        super(VGGLoss, self).__init__()

        if vgg_model not in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            raise ValueError(f"Unknown vgg-model: {vgg_model}")

        if batch_norm:
            vgg_model = f"{vgg_model}_bn"

        models = {
            "vgg11": vgg.vgg11,
            "vgg11_bn": vgg.vgg11_bn,
            "vgg13": vgg.vgg13,
            "vgg13_bn": vgg.vgg13_bn,
            "vgg16": vgg.vgg16,
            "vgg16_bn": vgg.vgg16_bn,
            "vgg19": vgg.vgg19,
            "vgg19_bn": vgg.vgg19_bn,
        }

        # mean and std come from ImageNet dataset since VGG is trained on that data
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.m: VGG = models[vgg_model]().features[: layers + 1]
        self.m.eval()
        self.m.requires_grad_(False)

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        if preds.shape[1] != 3:
            preds = preds.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        preds = self.m(self.normalize(preds))
        target = self.m(self.normalize(target))

        self.metric += mse_loss(input=preds, target=target, reduction=reduction)
        self.total += preds.size()[0]
