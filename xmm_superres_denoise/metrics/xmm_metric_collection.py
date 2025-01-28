from typing import List

from metrics import FSIM, GMSD, MDSI, HaarPSI, MultiScaleGMSD, PoissonNLLLoss
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)
from transforms import Normalize


def get_metrics(
    dataset_normalizer: Normalize,
    scaling_normalizers: List[Normalize],
    prefix: str,
):
    metrics = MetricCollection(
        {
            "psnr": PeakSignalNoiseRatio(),
            "ssim": StructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            ),
            "ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            ),
            "l1": MeanAbsoluteError(),
            "l2": MeanSquaredError(),
            "poisson": PoissonNLLLoss(),
        }
    )
    return XMMMetricCollection(
        metrics=metrics,
        dataset_normalizer=dataset_normalizer,
        scaling_normalizers=scaling_normalizers,
        prefix=prefix,
    )


def get_ext_metrics(
    dataset_normalizer: Normalize,
    scaling_normalizers: List[Normalize],
    prefix: str,
):
    metrics = MetricCollection(
        {
            "vif_p": VisualInformationFidelity(),
            "fsim": FSIM(),
            "gmsd": GMSD(),
            "ms_gmsd": MultiScaleGMSD(),
            "haarpsi": HaarPSI(),
            "msdi": MDSI(),
        }
    )
    return XMMMetricCollection(
        metrics=metrics,
        dataset_normalizer=dataset_normalizer,
        scaling_normalizers=scaling_normalizers,
        prefix=prefix,
    )


def get_in_metrics(
    dataset_normalizer: Normalize,
    scaling_normalizers: List[Normalize],
    prefix: str,
):
    metrics = MetricCollection(
        {
            "in/psnr": PeakSignalNoiseRatio(),
            "in/ssim": StructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            ),
            "in/ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            ),
            "in/l1": MeanAbsoluteError(),
            "in/l2": MeanSquaredError(),
            "in/poisson": PoissonNLLLoss(),
        }
    )
    return XMMMetricCollection(
        metrics=metrics,
        dataset_normalizer=dataset_normalizer,
        scaling_normalizers=scaling_normalizers,
        prefix=prefix,
    )


def get_in_ext_metrics(
    dataset_normalizer: Normalize,
    scaling_normalizers: List[Normalize],
    prefix: str,
):
    metrics = MetricCollection(
        {
            "in/vif_p": VisualInformationFidelity(),
            "in/fsim": FSIM(),
            "in/gmsd": GMSD(),
            "in/ms_gmsd": MultiScaleGMSD(),
            "in/haarpsi": HaarPSI(),
            "in/msdi": MDSI(),
        }
    )
    return XMMMetricCollection(
        metrics=metrics,
        dataset_normalizer=dataset_normalizer,
        scaling_normalizers=scaling_normalizers,
        prefix=prefix,
    )


class XMMMetricCollection(MetricCollection):
    def __init__(
        self,
        metrics: MetricCollection,
        dataset_normalizer: Normalize,
        scaling_normalizers: List[Normalize],
        prefix: str,
    ):
        self.dataset_normalizer = dataset_normalizer

        metric_list = []
        normalizer_dict = {}

        for normalizer in scaling_normalizers:
            mode = normalizer.stretch_mode
            normalizer_dict[mode] = normalizer
            metric_list.append(metrics.clone(prefix=f"{mode}/"))

        self.normalizer_dict = normalizer_dict
        super().__init__(metrics=metric_list, prefix=f"{prefix}/")

    def update(self, preds, target) -> None:
        preds = self.dataset_normalizer.denorm(preds)
        target = self.dataset_normalizer.denorm(target)
        for metric_name, metric in self.items(copy_state=False):
            mode = metric_name.split("/")[1]
            normalizer = self.normalizer_dict[mode]
            preds_scaled = normalizer.norm(preds)
            target_scaled = normalizer.norm(target)
            metric.update(preds=preds_scaled, target=target_scaled)
