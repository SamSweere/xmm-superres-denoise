# Based off: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan
from typing import List, Optional, Tuple, Dict, Union


import pytorch_lightning as pl
import torch
from torch import Tensor
from torchmetrics import Metric

from xmm_superres_denoise.metrics import XMMMetricCollection
from xmm_superres_denoise.transforms import ImageUpsample, CustomSigmoid
from datetime import datetime

from pytorch_lightning.utilities import rank_zero_warn

class Model(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        lr_shape: Tuple[int, int],
        hr_shape: Tuple[int, int],
        loss: Optional[Metric],
        loss_config: Dict[str, Union[int, dict]],
        metrics: Optional[XMMMetricCollection],
        extended_metrics: Optional[XMMMetricCollection],
        in_metrics: Optional[XMMMetricCollection],
        in_extended_metrics: Optional[XMMMetricCollection],
    ):
        super(Model, self).__init__()

        self.metrics = metrics
        self.ext_metrics = extended_metrics
        self.in_metrics = in_metrics
        self.in_ext_metrics = in_extended_metrics
        self.config = config
        
        # Tried to apply the loss_normalizer in the loss_definition, but that's not possible since you cannot input a CompositionalMetric into torch functions
        # Parameters for sigmoid applied to loss function 
        if loss_config['apply_sigmoid_to_loss']:
            
            rank_zero_warn(
                "You are applying a Sigmoid function to the trianing loss. Make sure that the Sigmoid function parameters in the loss_functions.yaml file match the expected statistics of the chosen loss function(s)."
            )
            self.sigmoid_loss_normalizer = CustomSigmoid(loss_config['k'], loss_config['x0'])
            
        # Optimizer parameters
        self.learning_rate = config["learning_rate"]
        self.betas = (config["b1"], config["b2"])

        # Model and model parameters
        self.memory_efficient = config["memory_efficient"]
        self.model_name = config["name"]
        self.loss = loss 
        self.loss_config = loss_config
        self.clamp = config["clamp"]
        self.batch_size = config["batch_size"]
        self.model: torch.nn.Module
        if self.model_name == "esr_gen":
            from xmm_superres_denoise.models import GeneratorRRDB_SR

            up_scale = hr_shape[0] / lr_shape[0]
            if up_scale % 2 != 0:
                raise ValueError(
                    f"Upscaling is not a multiple of two but {up_scale}, "
                    f"based on in_dims {lr_shape} and out_dims {hr_shape}"
                )

            up_scale = int(up_scale / 2)
            # Initialize generator and discriminator
            self.model = GeneratorRRDB_SR(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                num_filters=config["filters"],
                num_res_blocks=config["residual_blocks"],
                H_in = config["H_in"], 
                W_in = config["W_in"],
                num_upsample=up_scale,
                memory_efficient=self.memory_efficient,
                normalization_layer=config["normalization_layer"]
            )
        elif self.model_name == "rrdb_denoise":
            from xmm_superres_denoise.models import GeneratorRRDB_DN

            self.model = GeneratorRRDB_DN(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                num_filters=config["filters"],
                num_res_blocks=config["residual_blocks"],
                H_in = config["H_in"], 
                W_in = config["W_in"],
                memory_efficient=self.memory_efficient,
                normalization_layer=config["normalization_layer"]
            )
        elif self.model_name == "swinir":
            from xmm_superres_denoise.models import SwinIR

            self.model = SwinIR(
                img_size=config["img_size"],
                window_size=config["window_size"],
                embed_dim=config["embed_dim"],
                num_heads=config["num_heads"],
                depths=config["depths"],
                upsampler=config["upsampler"],
                in_chans=config["in_channels"],
                use_checkpoint=self.memory_efficient,
            )
        else:
            raise ValueError(
                f"Base model name {self.model_name} is not a valid model name!"
            )

    def forward(self, x) -> torch.Tensor:
        #TODO: figure out if this should be included in clamping or not
        return torch.clamp(self.model(x), min=0.0, max=1.0)

    def training_step(self, batch, batch_idx):
        return self._on_step(batch, "train")

    def on_validation_start(self) -> None:
        self._on_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        self._on_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end("val")

    def test_step(self, batch, batch_idx):
        self._on_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end("test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch["preds"] = self(batch["lr"])

    def _on_step(self, batch, stage) -> Optional[Tensor]:
        lr = batch["lr"]
        idx = batch["idx"].to()
        preds = self(lr)
        target = batch.get("hr", preds)

        if stage == "train":
            loss = self.loss(preds=preds, target=target)
            loss = self.sigmoid_loss_normalizer(loss) if self.loss_config['apply_sigmoid_to_loss'] else loss
            self.log(
                f"{stage}/loss",
                loss,
                batch_size=self.batch_size,
                on_step=True,
                on_epoch=False,
            )
            return loss
        else:
            self.loss.update(preds=preds, target=target)

            if self.in_metrics is not None or self.ext_metrics is not None:
                scale_factor = target.shape[2] / lr.shape[2]
                if scale_factor != 1.0:
                    lr = ImageUpsample(scale_factor=scale_factor)(lr)

            if self.metrics is not None:
                self.metrics.hr_update(preds=preds, target=target, idx = batch["idx"])

            if self.in_metrics is not None:
                self.in_metrics.lr_update(preds=lr, target=target, idx = batch["idx"])

            if self.ext_metrics is not None:
                self.ext_metrics.hr_update(preds=preds, target=target, idx = batch["idx"])

            if self.in_ext_metrics is not None:
                self.in_ext_metrics.lr_update(preds=lr, target=target, idx = batch["idx"])

    def _on_epoch_end(self, stage):
        if stage == "train":
            self.loss.reset()
        else:
            loss = self.loss.compute()
            self.log(
                f"{stage}/loss",
                loss,
                batch_size=self.batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.loss.reset()

            to_log: List[dict] = []

            if self.metrics is not None:
                to_log.append(self.metrics.compute())
                self.metrics.reset()

            if self.ext_metrics is not None:
                to_log.append(self.ext_metrics.compute())
                self.ext_metrics.reset()

            if self.in_metrics is not None:
                to_log.append(self.in_metrics.compute())
                self.in_metrics.reset()
                if not self.trainer.sanity_checking:
                    self.in_metrics = None

            if self.in_ext_metrics is not None:
                to_log.append(self.in_ext_metrics.compute())
                self.in_ext_metrics.reset()
                if not self.trainer.sanity_checking:
                    self.in_ext_metrics = None

            for log in to_log:
                self.log_dict(
                    log,
                    batch_size=self.batch_size,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                
            # Get the current date and time
            current_datetime = datetime.now()

            # Format the date and time as a string (e.g., "2023-12-04_12-34-56")
            formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            checkpoint_path = f"checkpoints/model_epoch_{self.current_epoch}_{self.config['name']}_{formatted_datetime}.ckpt"
            self.trainer.save_checkpoint(checkpoint_path)


    def configure_optimizers(self):
        # Optimizers
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, betas=self.betas
        )

        return optimizer
