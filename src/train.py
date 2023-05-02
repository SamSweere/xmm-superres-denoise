from argparse import ArgumentParser
from pathlib import Path

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info

from datasets import XmmDataModule, XmmDisplayDataModule
from metrics import MetricsCalculator
from models import Model
from transforms import Normalize, ImageUpsample
from utils import ImageLogger
from utils.filehandling import read_yaml
from utils.loss_functions import create_loss

if __name__ == "__main__":  # This is needed in order to run it on multiple gpu's
    parser = ArgumentParser()
    parser.add_argument("path_to_run_config", type=Path, help="Path to the run config yaml.")
    args = parser.parse_args()

    run_config: dict = read_yaml(args.path_to_run_config)
    wandb_config: dict = run_config["wandb"]

    dataset_config: dict = run_config["dataset"]
    dataset_config.update(read_yaml(Path("res") / "configs" / "dataset" / f"{dataset_config['name']}.yaml"))

    model_config: dict = run_config["model"]
    model_config.update(read_yaml(Path("res") / "configs" / "model" / f"{model_config['name']}.yaml"))
    model_config["batch_size"] = dataset_config["batch_size"]

    trainer_config: dict = run_config["trainer"]

    rank_zero_info("Creating data module...")
    datamodule = XmmDataModule(dataset_config)

    loss = create_loss(data_scaling=dataset_config["data_scaling"],
                       l1_p=model_config["loss_l1"],
                       poisson_p=model_config["loss_poisson"],
                       psnr_p=model_config["loss_psnr"],
                       ssim_p=model_config["loss_ssim"],
                       ms_ssim_p=model_config["loss_ms_ssim"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="res/checkpoints/",
        filename=model_config["name"] + "-{epoch:05d}-{val/loss:.5f}",
        mode="min"
    )

    lr_max = dataset_config["lr_max"]
    hr_max = dataset_config["hr_max"]
    lr_shape = (dataset_config["lr_res"], dataset_config["lr_res"])
    hr_shape = (dataset_config["hr_res"], dataset_config["hr_res"])
    scaling_normalizers = [
        Normalize(lr_max=lr_max, hr_max=hr_max, stretch_mode=s_mode) for s_mode in ["linear", "sqrt", "asinh", "log"]
    ]

    upsample = None
    if dataset_config["lr_res"] != dataset_config["hr_res"]:
        upsample = ImageUpsample(scale_factor=int(dataset_config["hr_res"] / dataset_config["lr_res"]))

    mc = MetricsCalculator(
        data_range=hr_max,
        dataset_normalizer=datamodule.normalize,
        scaling_normalizers=scaling_normalizers,
        upsample=upsample,
        prefix="test" if trainer_config["checkpoint_path"] else "val"
    )

    il = ImageLogger(
        datamodule=XmmDisplayDataModule(dataset_config),
        log_every_n_epochs=trainer_config["log_images_every_n_epochs"],
        normalize=datamodule.normalize,
        scaling_normalizers=scaling_normalizers,
        data_range=hr_max,
        metrics_calculator=mc
    )

    model = Model(
        model_config,
        lr_shape=lr_shape,
        hr_shape=hr_shape,
        loss=loss,
        metrics=mc
    )

    # model = model.to(torch.device("cuda"))

    if wandb_config["online"]:
        wandb.login(key=wandb_config["api_key"])

    if trainer_config["checkpoint_path"]:
        wandb_logger = WandbLogger(
            project=wandb_config["project"],
            log_model=wandb_config["online"] and wandb_config["log_model"],
            save_dir=wandb_config["log_dir"],
            offline=not wandb_config["online"],
            config=run_config,
            resume="allow",
            id=wandb_config["run"]["id"]
        )
        trainer = Trainer(
            logger=wandb_logger,  # W&B integration
            accelerator=trainer_config["accelerator"],
            devices=1,
            callbacks=[il]
        )
        trainer.test(model, datamodule=datamodule, ckpt_path=trainer_config["checkpoint_path"])
    else:
        wandb_logger = WandbLogger(
            project=wandb_config["project"],
            log_model=wandb_config["online"] and wandb_config["log_model"],
            save_dir=wandb_config["log_dir"],
            offline=not wandb_config["online"],
            config=run_config
        )
        trainer = Trainer(
            logger=wandb_logger,  # W&B integration
            accelerator=trainer_config["accelerator"],
            devices=trainer_config["devices"],
            max_epochs=trainer_config["epochs"],
            strategy="ddp_find_unused_parameters_false",
            # deterministic=True,  # keep it deterministic
            # benchmark=(not config["debug"]) and True,
            # fast_dev_run=config["fast_dev_run"],
            callbacks=[checkpoint_callback, il]
        )
        trainer.fit(model, datamodule=datamodule)
