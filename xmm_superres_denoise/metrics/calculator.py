from typing import List, Optional, Union

import torch
from metrics.metrics import (FSIM, GMSD, MDSI, VIF, HaarPSI, MultiScaleGMSD,
                             PoissonNLLLoss)
from torchmetrics import (MeanAbsoluteError, MeanSquaredError,
                          MetricCollection,
                          MultiScaleStructuralSimilarityIndexMeasure,
                          PeakSignalNoiseRatio,
                          StructuralSimilarityIndexMeasure)
from transforms import ImageUpsample, Normalize


class MetricsCalculator:
    def __init__(
        self,
        data_range: Union[int, float],
        dataset_normalizer: Normalize,
        scaling_normalizers: List[Normalize],
        upsample: ImageUpsample,
        prefix: str,
    ):
        super(MetricsCalculator, self).__init__()

        self.data_range = data_range
        self.dataset_normalizer = dataset_normalizer
        self.scaling_normalizers = scaling_normalizers
        self.upsample = upsample

        metric_template = MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(data_range=data_range),
                "ssim": StructuralSimilarityIndexMeasure(
                    kernel_size=13, sigma=2.5, k2=0.05, data_range=data_range
                ),
                "ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(
                    kernel_size=13, sigma=2.5, k2=0.05, data_range=data_range
                ),
                "l1": MeanAbsoluteError(),
                "l2": MeanSquaredError(),
                "poisson": PoissonNLLLoss(),
            }
        )

        extended_metric_template = MetricCollection(
            {
                "vif_p": VIF(),
                "fsim": FSIM(),
                "gmsd": GMSD(),
                "ms_gmsd": MultiScaleGMSD(),
                "haarpsi": HaarPSI(),
                "msdi": MDSI(),
            }
        )

        metrics = []
        extended_metrics = []
        normalizer_dict = {}

        for normalizer in scaling_normalizers:
            mode = normalizer.stretch_mode
            normalizer_dict[mode] = normalizer
            metrics.append(metric_template.clone(prefix=f"{prefix}/{mode}/"))
            extended_metrics.append(
                extended_metric_template.clone(prefix=f"{prefix}/{mode}/")
            )

        self.metrics = MetricCollection(metrics)
        self.input_metrics = self.metrics.clone(postfix="_in")
        self.extended_metrics = MetricCollection(extended_metrics)
        self.input_extended_metrics = self.extended_metrics.clone(postfix="_in")

        self.normalizer_dict = normalizer_dict

    def _update(
        self,
        preds: torch.Tensor,
        lr: Optional[torch.Tensor],
        target: torch.Tensor,
        extended: bool = False,
    ):
        p = self.dataset_normalizer.denormalize_hr_image(preds)
        t = self.dataset_normalizer.denormalize_hr_image(target)
        self._update_metric_collection(
            preds=p, target=t, metric_collection=self.metrics
        )

        if extended:
            self._update_metric_collection(
                preds=p, target=t, metric_collection=self.extended_metrics
            )

        if lr is not None:
            l = self.dataset_normalizer.denormalize_lr_image(lr)
            l = self.upsample(l)
            self._update_metric_collection(
                preds=l, target=t, metric_collection=self.input_metrics
            )

            if extended:
                self._update_metric_collection(
                    preds=l, target=t, metric_collection=self.input_extended_metrics
                )

    def _update_metric_collection(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        metric_collection: MetricCollection,
    ):
        p = torch.clamp(preds, min=0.0, max=self.data_range)
        t = torch.clamp(target, min=0.0, max=self.data_range)
        for metric_name, metric in metric_collection.items(copy_state=False):
            mode = metric_name.split("/")[1]
            normalizer = self.normalizer_dict[mode]
            preds_scaled = normalizer.normalize_hr_image(p)
            target_scaled = normalizer.normalize_hr_image(t)
            metric.update(preds=preds_scaled, target=target_scaled)

    def update(
        self,
        preds: torch.Tensor,
        lr: Optional[torch.Tensor],
        target: torch.Tensor,
        stage: str,
    ):
        if stage == "val":
            self._update(preds=preds, lr=lr, target=target, extended=False)
        elif stage == "test":
            self._update(preds=preds, lr=lr, target=target, extended=True)