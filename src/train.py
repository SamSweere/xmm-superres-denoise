from argparse import ArgumentParser
from pathlib import Path

import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn

from datasets import XmmDataModule, XmmDisplayDataModule
from metrics import MetricsCalculator
from models import Model
from transforms import Normalize, ImageUpsample
from utils import ImageLogger
from utils.filehandling import read_yaml
from utils.loss_functions import create_loss

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("routine", choices=["fit", "test"], help="What routine to execute")
    parser.add_argument("run_config", type=Path, help="Path to the run config yaml.")
    args = parser.parse_args()

    run_config: dict = read_yaml(args.run_config)
    wandb_config: dict = run_config["wandb"]

    dataset_config: dict = run_config["dataset"]

    model_config: dict = run_config["model"]
    model_config.update(read_yaml(Path("res") / "configs" / "model" / f"{model_config['name']}.yaml"))
    model_config["batch_size"] = dataset_config["batch_size"]

    loss_config = model_config["loss"]

    trainer_config: dict = run_config["trainer"]

    if wandb_config["online"]:
        wandb.login(key=wandb_config["api_key"])

    wandb_logger = WandbLogger(
        project=wandb_config["project"],
        log_model=wandb_config["online"] and wandb_config["log_model"],
        offline=not wandb_config["online"],
        config=run_config,
        resume="must" if wandb_config["run"]["id"] is not None else None,
        id=wandb_config["run"]["id"]
    )

    rank_zero_info("Creating data module...")
    datamodule = XmmDataModule(dataset_config)

    loss = create_loss(data_scaling=dataset_config["scaling"], loss_config=loss_config)

    rank_zero_info(f"Created loss function {loss}")

    lr_max = dataset_config["lr"]["max"]
    hr_max = dataset_config["hr"]["max"]
    lr_shape = (dataset_config["lr"]["res"], dataset_config["lr"]["res"])
    hr_shape = (dataset_config["hr"]["res"], dataset_config["hr"]["res"])
    scaling_normalizers = [
        Normalize(lr_max=lr_max, hr_max=hr_max, stretch_mode=s_mode) for s_mode in ["linear", "sqrt", "asinh", "log"]
    ]

    upsample = None
    if dataset_config["lr"]["res"] != dataset_config["hr"]["res"]:
        upsample = ImageUpsample(scale_factor=int(dataset_config["hr"]["res"] / dataset_config["lr"]["res"]))

    mc = MetricsCalculator(
        data_range=hr_max,
        dataset_normalizer=datamodule.normalize,
        scaling_normalizers=scaling_normalizers,
        upsample=upsample,
        prefix="val" if args.routine == "fit" else "test"
    )

    model = Model(
        model_config,
        lr_shape=lr_shape,
        hr_shape=hr_shape,
        loss=loss,
        metrics=mc
    )

    callbacks = None

    if args.routine == "fit":
        il = ImageLogger(
            datamodule=XmmDisplayDataModule(dataset_config),
            log_every_n_epochs=trainer_config["log_images_every_n_epochs"],
            normalize=datamodule.normalize,
            scaling_normalizers=scaling_normalizers,
            data_range=hr_max
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=f"{wandb_logger.experiment.dir}/checkpoints",
            filename=f"epoch:{{epoch:05d}}-val_loss:{{val/loss:.5f}}",
            mode="min",
            auto_insert_metric_name=False
        )
        callbacks = [il, checkpoint_callback]

    trainer = Trainer(
        logger=wandb_logger,
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"] if args.routine == "fit" else 1,
        max_epochs=trainer_config["epochs"],
        strategy=trainer_config["strategy"],
        callbacks=callbacks
    )

    if args.routine == "fit":
        if trainer_config["checkpoint_path"] is not None:
            rank_zero_warn("You have given a checkpoint_path in the trainer config! If it was on purpose, then "
                           "you can ignore this warning.")
        trainer.fit(model, datamodule=datamodule, ckpt_path=trainer_config["checkpoint_path"])
    else:
        trainer.test(model, datamodule=datamodule, ckpt_path=trainer_config["checkpoint_path"])
