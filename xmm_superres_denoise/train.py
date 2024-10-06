import tomllib
from argparse import ArgumentParser
from pathlib import Path

import wandb
from config.config import (
    DatasetCfg,
    LossCfg,
    ModelCfg,
    TrainerCfg,
    TrainerStrategy,
    WandbCfg,
)
from loguru import logger
from metrics import get_ext_metrics, get_in_ext_metrics, get_in_metrics, get_metrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from transforms import Normalize
from utils import ImageLogger
from utils.loss_functions import create_loss

from data import XmmDataModule, XmmDisplayDataModule
from models import Model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "routine", choices=["fit", "test"], help="What routine to execute"
    )
    parser.add_argument("run_config", type=Path, help="Path to the run config toml.")
    args = parser.parse_args()

    with open(args.run_config, "rb") as file:
        cfg: dict[str, dict] = tomllib.load(file)

    wandb_config: WandbCfg = WandbCfg(**cfg["wandb"])

    dataset = cfg["dataset"]
    if dataset["hr"]["exp"] == 0:
        dataset["hr"] = None
    dataset_config: DatasetCfg = DatasetCfg(**cfg["dataset"])

    model_config: dict = cfg["model"]
    with open(Path("res") / "configs" / "models.toml", "rb") as file:
        model_config["model"] = tomllib.load(file)[model_config["name"]]

    model_config["optimizer"] = {
        "learning_rate": model_config["model"].pop("learning_rate"),
        "betas": model_config["model"].pop("betas"),
    }
    model_config["batch_size"] = dataset_config.batch_size
    model_config: ModelCfg = ModelCfg(**model_config)

    with open(Path("res") / "configs" / "loss_functions.toml", "rb") as file:
        loss_config: dict = tomllib.load(file)

    sc_dict = None
    if loss_config["loss"].pop("use_scaling"):
        sc_dict = loss_config.pop("scaling")
        sc_dict = sc_dict[dataset_config.scaling]
    loss_config: LossCfg = LossCfg(**loss_config["loss"])

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
    loss = create_loss(sc_dict=sc_dict, loss_config=loss_config)
    del sc_dict, loss_config

    logger.info(f"Created loss function {loss}")

    lr_max = dataset_config.lr.clamp_max
    hr_max = dataset_config.hr.clamp_max
    lr_shape = (dataset_config.lr.res, dataset_config.lr.res)
    hr_shape = (dataset_config.hr.res, dataset_config.hr.res)
    del dataset_config
    scaling_normalizers = [
        Normalize(lr_max=lr_max, hr_max=hr_max, stretch_mode="linear")
    ]

    pre = "val" if args.routine == "fit" else "test"
    metrics = get_metrics(
        dataset_normalizer=datamodule.normalize,
        scaling_normalizers=scaling_normalizers,
        prefix=pre,
    )

    in_metrics = get_in_metrics(
        dataset_normalizer=datamodule.normalize,
        scaling_normalizers=scaling_normalizers,
        prefix=pre,
    )

    ext_metrics = in_ext_metrics = None
    if args.routine == "test":
        ext_metrics = get_ext_metrics(
            dataset_normalizer=datamodule.normalize,
            scaling_normalizers=scaling_normalizers,
            prefix=pre,
        )

        in_ext_metrics = get_in_ext_metrics(
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
    del model_config

    # model.compile()

    callbacks = None

    if args.routine == "fit":
        callbacks = []
        if trainer_config.log_images_every_n_epochs > 0:
            pass
            # TODO ImageLogger has to be fixed
            # il = ImageLogger(
            #     datamodule=XmmDisplayDataModule(dataset_config),
            #     log_every_n_epochs=trainer_config.log_images_every_n_epochs,
            #     normalize=datamodule.normalize,
            #     scaling_normalizers=scaling_normalizers,
            #     data_range=hr_max,
            # )
            # callbacks.append(il)
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=f"{trainer_config.checkpoint_root}/checkpoints/"
            f"{wandb_logger.experiment.name}_{wandb_logger.experiment.id}",
            filename=f"epoch:{{epoch:05d}}-val_loss:{{val/loss:.5f}}",
            mode="min",
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

    strategy = trainer_config.strategy
    if trainer_config.strategy is TrainerStrategy.FSDP:
        strategy = FSDPStrategy(
            auto_wrap_policy=model.auto_wrap_policy,
            activation_checkpointing_policy=model.activation_checkpointing_policy,
        )

    trainer = Trainer(
        logger=wandb_logger,
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices if args.routine == "fit" else 1,
        max_epochs=trainer_config.epochs,
        strategy=strategy,
        callbacks=callbacks,
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

        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    else:
        trainer.test(
            model, datamodule=datamodule, ckpt_path=trainer_config.checkpoint_path
        )
