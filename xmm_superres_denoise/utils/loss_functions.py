from config.config import LossCfg
from metrics import PoissonNLLLoss
from torchmetrics import MeanAbsoluteError, Metric
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


def create_loss(
    sc_dict: dict[str, dict[str, float]] | None,
    loss_config: LossCfg,
) -> Metric:
    modules = {
        "l1": MeanAbsoluteError,
        "poisson": PoissonNLLLoss,
        "psnr": PeakSignalNoiseRatio,
        "ssim": StructuralSimilarityIndexMeasure,
        "ms_ssim": MultiScaleStructuralSimilarityIndexMeasure,
    }

    correction = 0.0
    metrics = []

    for loss, p in iter(loss_config):
        if p > 0.0:
            if sc_dict is not None and loss in sc_dict:
                p = p * sc_dict[loss]["scaling"]
                correction = correction + sc_dict[loss]["correction"]

            if loss == "ssim" or loss == "ms_ssim":
                module = modules[loss](kernel_size=13, sigma=2.5, k2=0.05)
            else:
                module = modules[loss]()
            metrics.append(module * p)

    assert metrics

    final_metric = metrics[0]
    for i in range(1, len(metrics)):
        final_metric = final_metric + metrics[i]

    if correction > 0.0:
        final_metric = final_metric + correction

    return final_metric
