from typing import List, Union
from typing import Optional

import torch
from torchmetrics import MetricCollection

from metrics.metrics import MDSI, PSNR, SSIM, MultiScaleSSIM, MAE, MSE, PoissonNLLLoss, VIF, FSIM, GMSD, \
    MultiScaleGMSD, HaarPSI
from transforms import ImageUpsample, Normalize


class MetricsCalculator:
    def __init__(
            self,
            data_range: Union[int, float],
            dataset_normalizer: Normalize,
            scaling_normalizers: List[Normalize],
            upsample: ImageUpsample,
            prefix: str
    ):
        super(MetricsCalculator, self).__init__()

        self.data_range = data_range
        self.dataset_normalizer = dataset_normalizer
        self.scaling_normalizers = scaling_normalizers
        self.upsample = upsample

        metrics = MetricCollection({
            "psnr": PSNR(),
            "ssim": SSIM(),
            "ms_ssim": MultiScaleSSIM(),
            "l1": MAE(),
            "l2": MSE(),
            "poisson": PoissonNLLLoss()
        })

        extended_metrics = MetricCollection({
            "vif_p": VIF(),
            "fsim": FSIM(),
            "gmsd": GMSD(),
            "ms_gmsd": MultiScaleGMSD(),
            "haarpsi": HaarPSI(),
            "msdi": MDSI()
        })

        input_metrics = MetricCollection({
            "psnr_in": PSNR(),
            "ssim_in": SSIM(),
            "l1_in": MAE(),
            "l2_in": MSE(),
            "poisson_in": PoissonNLLLoss()
        })

        input_extended_metrics = MetricCollection({
            "vif_p_in": VIF(),
            "fsim_in": FSIM(),
            "gmsd_in": GMSD(),
            "ms_gmsd_in": MultiScaleGMSD(),
            "haarpsi_in": HaarPSI(),
            "msdi_in": MDSI()
        })

        self.metrics = []
        self.extended_metrics = []
        self.input_metrics = []
        self.input_extended_metrics = []
        self.normalizer_dict = {scaling_normalizer.stretch_mode: scaling_normalizer
                                for scaling_normalizer in scaling_normalizers}

        for normalizer in scaling_normalizers:
            mode = normalizer.stretch_mode
            self.metrics.append(metrics.clone(prefix=f"{mode}/"))
            self.extended_metrics.append(extended_metrics.clone(prefix=f"{mode}/"))
            self.input_metrics.append(input_metrics.clone(prefix=f"{mode}/"))
            self.input_extended_metrics.append(input_extended_metrics.clone(prefix=f"{mode}/"))

        self.metrics = MetricCollection(*self.metrics, prefix=f"{prefix}/")
        self.extended_metrics = MetricCollection(*self.extended_metrics, prefix=f"{prefix}/")
        self.input_metrics = MetricCollection(*self.input_metrics, prefix=f"{prefix}/")
        self.input_extended_metrics = MetricCollection(*self.input_extended_metrics, prefix=f"{prefix}/")

    def _update(
            self,
            preds: torch.Tensor,
            lr: Optional[torch.Tensor],
            target: torch.Tensor,
            extended: bool = False
    ):
        self._update_metric_collection(preds=preds, target=target, metric_collection=self.metrics)

        if extended:
            self._update_metric_collection(preds=preds, target=target, metric_collection=self.extended_metrics)

        if lr is not None:
            lr = self.dataset_normalizer.denormalize_lr_image(lr)
            lr = self.upsample(lr)
            self._update_metric_collection(preds=lr, target=target, metric_collection=self.input_metrics)

            if extended:
                self._update_metric_collection(preds=lr, target=target, metric_collection=self.input_extended_metrics)

    def _update_metric_collection(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            metric_collection: MetricCollection
    ):
        preds = preds.clamp(min=0.0, max=self.data_range)
        target = target.clamp(min=0.0, max=self.data_range)

        for name, metric in metric_collection.items(copy_state=False):
            mode = name.split("/")[1]
            normalizer = self.normalizer_dict[mode]
            preds_scaled = normalizer.normalize_hr_image(preds)
            target_scaled = normalizer.normalize_hr_image(target)
            metric = metric.to(preds_scaled.device)
            metric.update(preds=preds_scaled, target=target_scaled)

    def update_val(
            self,
            preds: torch.Tensor,
            lr: Optional[torch.Tensor],
            target: torch.Tensor
    ) -> None:
        self._update(
            preds=preds,
            lr=lr,
            target=target,
            extended=False
        )

    def update_test(
            self,
            preds: torch.Tensor,
            lr: Optional[torch.Tensor],
            target: torch.Tensor
    ) -> None:
        self._update(
            preds=preds,
            lr=lr,
            target=target,
            extended=True
        )
        self.update_val(preds=preds, lr=lr, target=target)
