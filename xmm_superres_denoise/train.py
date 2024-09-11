import tomllib
from argparse import ArgumentParser
from pathlib import Path

from config.config import DatasetCfg, TrainerCfg, WandbCfg
from loguru import logger
from metrics import get_ext_metrics, get_in_ext_metrics, get_in_metrics, get_metrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from transforms import Normalize
from utils import ImageLogger
from utils.filehandling import read_yaml
from utils.loss_functions import create_loss

import wandb
from data import XmmDataModule, XmmDisplayDataModule
from models import Model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "routine", choices=["fit", "test"], help="What routine to execute"
    )
    parser.add_argument("run_config", type=Path, help="Path to the run config yaml.")
    parser.add_argument("run_config_v2", type=Path, help="Path to the run config toml.")
    args = parser.parse_args()

    with open(args.run_config_v2, "rb") as file:
        cfg: dict[str, dict] = tomllib.load(file)

    run_config: dict = read_yaml(args.run_config)
    wandb_config: WandbCfg = WandbCfg(**cfg["wandb"])

    dataset_config: DatasetCfg = DatasetCfg(**cfg["dataset"])

    model_config: dict = run_config["model"]
    model_config.update(
        read_yaml(Path("res") / "configs" / "model" / f"{model_config['name']}.yaml")
    )
    model_config["batch_size"] = dataset_config.batch_size

    loss_config: dict = read_yaml(Path("res") / "configs" / "loss_functions.yaml")

    trainer_config: TrainerCfg = TrainerCfg(**cfg["trainer"])

    # --- Initialise the logger --- #
    logger.info("Creating the WandbLogger...")
    if wandb_config.online:
        wandb.login(key=wandb_config.api_key)

    wandb_logger = WandbLogger(
        project=wandb_config.project,
        log_model=wandb_config.online and wandb_config.log_model,
        offline=not wandb_config.online,
        config=cfg,
        resume="must" if wandb_config.run_id else None,
        id=wandb_config.run_id if wandb_config.run_id else None,
    )
    del wandb_config, cfg
    logger.success("Created WandbLogger!")

    # --- Initialise the XmmDataModule --- #
    logger.info("Creating data module...")
    datamodule = XmmDataModule(dataset_config)
    logger.success("Created the data module!")

    # --- Create the loss function --- #
    loss = create_loss(data_scaling=dataset_config.scaling, loss_config=loss_config)

    logger.info(f"Created loss function {loss}")

    lr_max = dataset_config.lr.clamp_max
    hr_max = dataset_config.hr.clamp_max
    lr_shape = (dataset_config.lr.res, dataset_config.lr.res)
    hr_shape = (dataset_config.hr.res, dataset_config.hr.res)
    scaling_normalizers = [
        Normalize(lr_max=lr_max, hr_max=hr_max, stretch_mode=s_mode)
        for s_mode in ["linear", "sqrt", "asinh", "log"]
    ]

    pre = "val" if args.routine == "fit" else "test"
    metrics = get_metrics(
        data_range=hr_max,
        dataset_normalizer=datamodule.normalize,
        scaling_normalizers=scaling_normalizers,
        prefix=pre,
    )

    in_metrics = get_in_metrics(
        data_range=hr_max,
        dataset_normalizer=datamodule.normalize,
        scaling_normalizers=scaling_normalizers,
        prefix=pre,
    )

    ext_metrics = in_ext_metrics = None
    if args.routine == "test":
        ext_metrics = get_ext_metrics(
            data_range=hr_max,
            dataset_normalizer=datamodule.normalize,
            scaling_normalizers=scaling_normalizers,
            prefix=pre,
        )

        in_ext_metrics = get_in_ext_metrics(
            data_range=hr_max,
            dataset_normalizer=datamodule.normalize,
            scaling_normalizers=scaling_normalizers,
            prefix=pre,
        )

    model = Model(
        model_config,
        lr_shape=lr_shape,
        hr_shape=hr_shape,
        loss=loss,
        metrics=metrics,
        in_metrics=in_metrics,
        extended_metrics=ext_metrics,
        in_extended_metrics=in_ext_metrics,
    )

    callbacks = None

    if args.routine == "fit":
        callbacks = []
        if trainer_config.log_images_every_n_epochs > 0:
            il = ImageLogger(
                datamodule=XmmDisplayDataModule(dataset_config),
                log_every_n_epochs=trainer_config.log_images_every_n_epochs,
                normalize=datamodule.normalize,
                scaling_normalizers=scaling_normalizers,
                data_range=hr_max,
            )
            callbacks.append(il)
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=f"{trainer_config.checkpoint_root}/checkpoints/"
            f"{wandb_logger.experiment.name}_{wandb_logger.experiment.id}",
            filename=f"epoch:{{epoch:05d}}-val_loss:{{val/loss:.5f}}",
            mode="min",
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

    trainer = Trainer(
        logger=wandb_logger,
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices if args.routine == "fit" else 1,
        max_epochs=trainer_config.epochs,
        strategy=trainer_config.strategy,
        callbacks=callbacks,
        limit_train_batches=10,
        limit_val_batches=10,
    )

    if args.routine == "fit":
        if trainer_config.checkpoint_path is not None:
            logger.warning(
                "You have given a checkpoint_path in the trainer config! If it was on purpose, then "
                "you can ignore this warning."
            )
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=trainer_config.checkpoint_path
        )
    else:
        trainer.test(
            model, datamodule=datamodule, ckpt_path=trainer_config.checkpoint_path
        )
