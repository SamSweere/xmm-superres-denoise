from collections import defaultdict
from typing import Dict, List, Union
from typing import Optional, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Metric

from metrics.metrics import MDSI, PSNR, SSIM, MultiScaleSSIM, MAE, MSE, PoissonNLLLoss, VIF, FSIM, GMSD, \
    MultiScaleGMSD, HaarPSI
from transforms import ImageUpsample, Normalize


class MetricsCalculator(pl.Callback):
    def __init__(
            self,
            data_range: Union[int, float],
            dataset_normalizer: Normalize,
            scaling_normalizers: List[Normalize],
            upsample: ImageUpsample
    ):
        super(MetricsCalculator, self).__init__()

        self.data_range = data_range
        self.dataset_normalizer = dataset_normalizer
        self.scaling_normalizers = scaling_normalizers
        self.upsample = upsample

        self.metrics: Dict[str, Metric] = {}
        self.extended_metrics: Dict[str, Metric] = {}
        self.input_metrics: Dict[str, Metric] = {}
        self.input_extended_metrics: Dict[str, Metric] = {}

        for normalizer in scaling_normalizers:
            mode = normalizer.stretch_mode

            self.metrics.update({
                f"{mode}/psnr": PSNR(),
                f"{mode}/ssim": SSIM(),
                f"{mode}/ms_ssim": MultiScaleSSIM(),
                f"{mode}/l1": MAE(),
                f"{mode}/l2": MSE(),
                f"{mode}/poisson": PoissonNLLLoss()
            })

            self.extended_metrics.update({
                f"{mode}/vif_p": VIF(),
                f"{mode}/fsim": FSIM(),
                f"{mode}/gmsd": GMSD(),
                f"{mode}/ms_gmsd": MultiScaleGMSD(),
                f"{mode}/haarpsi": HaarPSI(),
                f"{mode}/msdi": MDSI()
            })

            self.input_metrics.update({
                f"{mode}/psnr_in": PSNR(),
                f"{mode}/ssim_in": SSIM(),
                f"{mode}/l1_in": MAE(),
                f"{mode}/l2_in": MSE(),
                f"{mode}/poisson_in": PoissonNLLLoss()
            })

            self.input_extended_metrics.update({
                f"{mode}/vif_p_in": VIF(),
                f"{mode}/fsim_in": FSIM(),
                f"{mode}/gmsd_in": GMSD(),
                f"{mode}/ms_gmsd_in": MultiScaleGMSD(),
                f"{mode}/haarpsi_in": HaarPSI(),
                f"{mode}/msdi_in": MDSI()
            })

    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()
        for metric in self.extended_metrics.values():
            metric.reset()
        for metric in self.input_metrics.values():
            metric.reset()
        for metric in self.input_extended_metrics.values():
            metric.reset()

    def _on_batch_end(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            metric_dicts: List[Dict[str, Metric]]
    ):
        if preds.size()[-1] != target.size()[-1]:
            preds = self.dataset_normalizer.denormalize_lr_image(preds)
            preds = self.upsample(preds)
        else:
            preds = self.dataset_normalizer.denormalize_hr_image(preds)

        target = self.dataset_normalizer.denormalize_hr_image(target)

        preds.clamp_(min=0.0, max=self.data_range)
        target.clamp_(min=0.0, max=self.data_range)

        for normalizer in self.scaling_normalizers:
            mode = normalizer.stretch_mode
            preds = normalizer.normalize_hr_image(preds)
            target = normalizer.normalize_hr_image(target)
            for metric_dict in metric_dicts:
                for name, metric in metric_dict.items():
                    if name.startswith(mode):
                        metric = metric.to(preds.device)
                        metric.update(preds=preds, target=target, data_range=1.0, reduction="mean")

    def _on_end(
            self,
            logger: Logger,
            stage: str,
            metric_dicts: List[Dict[str, Metric]]
    ):
        metrics = defaultdict(float)
        for metric_dict in metric_dicts:
            for name, metric in metric_dict.items():
                metrics[f"{stage}/{name}"] = metric.compute().float()
        logger.log_metrics(metrics)
        self._reset_metrics()

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        if not trainer.sanity_checking:
            self._on_batch_end(
                preds=outputs["preds"],
                target=outputs["target"],
                metric_dicts=[self.metrics]
            )

            if trainer.current_epoch == 0:
                self._on_batch_end(
                    preds=outputs["lr"],
                    target=outputs["target"],
                    metric_dicts=[self.input_metrics]
                )

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking:
            metrics = [self.input_metrics, self.metrics] if trainer.current_epoch == 0 else [self.metrics]
            self._on_end(trainer.logger, "val", metrics)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self._on_batch_end(
            preds=outputs["preds"],
            target=outputs["target"],
            metric_dicts=[self.metrics, self.extended_metrics]
        )
        self._on_batch_end(
            preds=outputs["lr"],
            target=outputs["target"],
            metric_dicts=[self.input_metrics, self.input_extended_metrics]
        )

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._on_end(trainer.logger, "test",
                     [self.metrics, self.extended_metrics, self.input_metrics, self.input_extended_metrics])
