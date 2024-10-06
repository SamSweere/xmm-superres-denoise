from typing import List, Union

import torch
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from xmm_superres_denoise.metrics import (
    FSIM,
    GMSD,
    MDSI,
    VIF,
    HaarPSI,
    MultiScaleGMSD,
    PoissonNLLLoss,
)
from xmm_superres_denoise.transforms import Normalize


def get_metrics(
    data_range: Union[int, float],
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
        data_range=data_range,
        metrics=metrics,
        dataset_normalizer=dataset_normalizer,
        scaling_normalizers=scaling_normalizers,
        prefix=prefix,
    )


def get_ext_metrics(
    data_range: Union[int, float],
    dataset_normalizer: Normalize,
    scaling_normalizers: List[Normalize],
    prefix: str,
):
    metrics = MetricCollection(
        {
            "vif_p": VIF(),
            "fsim": FSIM(),
            "gmsd": GMSD(),
            "ms_gmsd": MultiScaleGMSD(),
            "haarpsi": HaarPSI(),
            "msdi": MDSI(),
        }
    )
    return XMMMetricCollection(
        data_range=data_range,
        metrics=metrics,
        dataset_normalizer=dataset_normalizer,
        scaling_normalizers=scaling_normalizers,
        prefix=prefix,
    )


def get_in_metrics(
    data_range: Union[int, float],
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
        data_range=data_range,
        metrics=metrics,
        dataset_normalizer=dataset_normalizer,
        scaling_normalizers=scaling_normalizers,
        prefix=prefix,
    )


def get_in_ext_metrics(
    data_range: Union[int, float],
    dataset_normalizer: Normalize,
    scaling_normalizers: List[Normalize],
    prefix: str,
):
    metrics = MetricCollection(
        {
            "in/vif_p": VIF(),
            "in/fsim": FSIM(),
            "in/gmsd": GMSD(),
            "in/ms_gmsd": MultiScaleGMSD(),
            "in/haarpsi": HaarPSI(),
            "in/msdi": MDSI(),
        }
    )
    return XMMMetricCollection(
        data_range=data_range,
        metrics=metrics,
        dataset_normalizer=dataset_normalizer,
        scaling_normalizers=scaling_normalizers,
        prefix=prefix,
    )


class XMMMetricCollection(MetricCollection):
    def __init__(
        self,
        data_range: Union[int, float],
        metrics: MetricCollection,
        dataset_normalizer: Normalize,
        scaling_normalizers: List[Normalize],
        prefix: str,
    ):
        self.data_range = data_range
        self.dataset_normalizer = dataset_normalizer

        metric_list = []
        normalizer_dict = {}

        for normalizer in scaling_normalizers:
            mode = normalizer.stretch_mode
            normalizer_dict[mode] = normalizer
            metric_list.append(metrics.clone(prefix=f"{mode}/"))

        self.normalizer_dict = normalizer_dict
        super(XMMMetricCollection, self).__init__(
            metrics=metric_list, prefix=f"{prefix}/"
        )
  
    def lr_update(self, preds, target, idx) -> None:
      
        preds = self.dataset_normalizer.denormalize_lr_image(preds, idx)
        target = self.dataset_normalizer.denormalize_hr_image(target, idx)
       
        for metric_name, metric in self.items(copy_state=False):
            mode = metric_name.split("/")[1]
            normalizer = self.normalizer_dict[mode]
            
            preds_scaled = normalizer.normalize_lr_image(preds, idx)
            target_scaled = normalizer.normalize_hr_image(target, idx)
            metric.update(preds=preds_scaled, target=target_scaled)


    def hr_update(self, preds, target, idx) -> None:
      
        preds = self.dataset_normalizer.denormalize_hr_image(preds, idx)
        target = self.dataset_normalizer.denormalize_hr_image(target, idx)
        
        for metric_name, metric in self.items(copy_state=False):
            mode = metric_name.split("/")[1]
            normalizer = self.normalizer_dict[mode]
            
            preds_scaled = normalizer.normalize_hr_image(preds, idx)
            target_scaled = normalizer.normalize_hr_image(target, idx)
            metric.update(preds=preds_scaled, target=target_scaled)

