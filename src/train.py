import os
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


from datasets.xmm_datamodule import XmmDataModule
from datasets.xmm_dispay_datamodule import XmmDisplayDataModule

from transforms.normalize import Normalize

from utils.filehandling import read_yaml
from utils.imagelogger import ImageLogger
from utils.loss_functions import LossFunctionHandler
from utils.metriclogger import MetricLogger
from utils.onnxexporter import OnnxExporter

# Load the run configs
run_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_config.yaml"
)
config = read_yaml(run_config_path)

# Load the dataset configs
dataset_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "dataset",
    config["dataset_name"] + ".yaml",
)
dataset_config = read_yaml(dataset_config_path)

# Add the debug to the dataset configs if enabled
dataset_config["debug"] = config["debug"]
if config["server"]:
    dataset_config["datasets_dir"] = config["server_datasets_dir"]
    config["runs_dir"] = config["server_runs_dir"]
else:
    dataset_config["datasets_dir"] = config["pc_datasets_dir"]
    config["runs_dir"] = config["pc_runs_dir"]

dataset_config["check_files"] = config["check_dataset_files"]
dataset_config["batch_size"] = config["batch_size"]

# Load the model configs
model_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "model",
    config["model_name"] + ".yaml",
)
model_config = read_yaml(model_config_path)

# Add the shape of the dataset to the model configs
model_config["lr_shape"] = (dataset_config["lr_res"], dataset_config["lr_res"])
model_config["hr_shape"] = (dataset_config["hr_res"], dataset_config["hr_res"])

# Add other parts to the config
config["project"] = model_config["base_model"]
config["wandb_log_online"] = config["log_online"]  
# not config['fast_dev_run'] and not config['debug'] # Only log the full runs

config["dataset_config"] = dataset_config
config["model_config"] = model_config

# wandb.login()

if config["end_to_end_test"] or config["fast_dev_run"] or config["debug"]:
    config["project"] = "test"

if config["end_to_end_test"]:
    config["epochs"] = 5
    config["log_images_every_n_epoch"] = 1
    config["log_every_n_steps"] = 10

wandb_mode = "online"
if not config["wandb_log_online"]:
    wandb_mode = "offline"


def main(config):
    if config["sweep"]:
        # We need to set the model parameters again, since sweep does not support nested dictionaries
        config["model_config"]["learning_rate"] = config["learning_rate"]
        config["model_config"]["filters"] = config["filters"]
        config["model_config"]["residual_blocks"] = config["residual_blocks"]
        config["dataset_config"]["batch_size"] = config["batch_size"]

        config["model_config"]["loss_l1"] = config["loss_l1"]
        config["model_config"]["loss_poisson"] = config["loss_poisson"]
        config["model_config"]["loss_psnr"] = config["loss_psnr"]
        config["model_config"]["loss_ssim"] = config["loss_ssim"]
        config["model_config"]["loss_ms_ssim"] = config["loss_ms_ssim"]
        config["dataset_config"]["data_scaling"] = config["data_scaling"]

    # Other settings (need to be set here because the sweep paramters alter them
    # For the display selection, use the same config as for the dataset but change the name
    sim_display_config = config["dataset_config"].copy()
    sim_display_config["dataset_type"] = "sim"
    sim_dataset_name = config["sim_display_name"]
    if not sim_display_config["hr_background"]:
        sim_dataset_name = sim_dataset_name + "_hr_no_back"
    sim_display_config["dataset_name"] = sim_dataset_name
    sim_display_config["datasets_dir"] = os.path.join(
        sim_display_config["datasets_dir"], "display_datasets"
    )
    # Override the lr_exp to always use the same exposure, otherwise the display images would be inconsistent
    sim_display_config["lr_exp"] = config["display_exposure"]

    real_display_config = config["dataset_config"].copy()
    real_display_config["dataset_type"] = "real"
    if "include_hr" not in real_display_config.keys():
        real_display_config["include_hr"] = False

    real_display_config["dataset_name"] = config["real_display_name"]
    real_display_config["datasets_dir"] = os.path.join(
        real_display_config["datasets_dir"], "display_datasets"
    )
    # Override the lr_exp to always use the same exposure, otherwise the display images would be inconsistent
    real_display_config["lr_exp"] = config["display_exposure"]

    # Set the seed for everything such that the results are reproducible
    # pytorch_lightning.utilities.seed.seed_everything(seed=42, workers=True)
    # torch.use_deterministic_algorithms(True)

    split_name_add = ""
    if config["sweep"]:
        split_name_add = "_sweep"

    print("Setting up train dataloader")
    train_datamodule = XmmDataModule(
        config["dataset_config"], split="train" + split_name_add
    )
    train_dataloader = train_datamodule.get_dataloader(shuffle=True)

    print("Setting up val dataloader")
    val_datamodule = XmmDataModule(
        config["dataset_config"], split="val" + split_name_add
    )
    val_dataloader = val_datamodule.get_dataloader(shuffle=False)

    if config["run_test_dataset"]:
        print("Setting up test dataloader")
        test_datamodule = XmmDataModule(
            config["dataset_config"], split="test" + split_name_add
        )
        test_dataloader = test_datamodule.get_dataloader(shuffle=False)
    else:
        # Run the test on the val dataset
        test_datamodule = val_datamodule
        test_dataloader = val_dataloader

    print("Setting simulated display dataloader")
    sim_display_datamodule = XmmDisplayDataModule(sim_display_config)
    sim_display_dataloader = iter(sim_display_datamodule.get_dataloader())

    print("Setting real display dataloader")
    real_display_datamodule = XmmDataModule(real_display_config)
    real_display_dataloader = iter(real_display_datamodule.get_dataloader())

    # Verify file correctness of dataset
    # datamodule.dataset.check_dataset_correctness()

    # TODO: only has to be done once
    # datamodule.dataset.get_mean_std(plot=True, histogram=True)

    sim_display_samples = []
    real_display_samples = []

    for i in range(len(sim_display_dataloader)):
        sim_display_samples.append(
            next(sim_display_dataloader)
        )  # This will return a dict

    for i in range(len(real_display_dataloader)):
        real_display_samples.append(
            next(real_display_dataloader)
        )  # This will return a dict

    # # Sort the display samples to have them always in the same order
    # sim_display_samples.sort()
    # real_display_samples.sort()

    # ------------------------------ Set the loss functions ------------------------------

    lfh = LossFunctionHandler(data_scaling=config["dataset_config"]["data_scaling"])

    # loss_names = config['model_config']['loss_func']
    # # Make sure the loss names are always in a list
    # if type(loss_names) == str:
    #     loss_names = [loss_names]

    criterion = lfh.get_loss_f(
        l1_p=config["model_config"]["loss_l1"],
        poisson_p=config["model_config"]["loss_poisson"],
        psnr_p=config["model_config"]["loss_psnr"],
        ssim_p=config["model_config"]["loss_ssim"],
        ms_ssim_p=config["model_config"]["loss_ms_ssim"],
    )

    # ------------------------------ Setup the model ------------------------------
    callbacks = []

    if config["model_config"]["base_model"] == "esr_gen":
        from models.esrgen_model import LitESRGEN

        model = LitESRGEN(
            lr_shape=config["model_config"]["lr_shape"],
            hr_shape=config["model_config"]["hr_shape"],
            in_channels=config["model_config"]["in_channels"],
            out_channels=config["model_config"]["out_channels"],
            filters=config["model_config"]["filters"],
            residual_blocks=config["model_config"]["residual_blocks"],
            learning_rate=config["model_config"]["learning_rate"],
            b1=config["model_config"]["b1"],
            b2=config["model_config"]["b2"],
            criterion=criterion,
        )
    elif config["model_config"]["base_model"] == "rrdb_denoise":
        from models.rrdb_denoise_model import LitRRDBDenoise

        model = LitRRDBDenoise(
            lr_shape=config["model_config"]["lr_shape"],
            hr_shape=config["model_config"]["hr_shape"],
            in_channels=config["model_config"]["in_channels"],
            out_channels=config["model_config"]["out_channels"],
            filters=config["model_config"]["filters"],
            residual_blocks=config["model_config"]["residual_blocks"],
            learning_rate=config["model_config"]["learning_rate"],
            b1=config["model_config"]["b1"],
            b2=config["model_config"]["b2"],
            criterion=criterion,
        )
    else:
        raise ValueError(
            f"Base model name {model_config['base_model']} is not a valid model name"
        )

    wandb_logger = WandbLogger()

    # Create the model checkpointer
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        filename=config["project"] + "-{epoch:05d}-{val/loss:.5f}",
        save_top_k=3,
        save_last=True,
        verbose=True,
        mode="min",
    )

    # Generate all the scaling_normalizers for the image and metric loggers

    scaling_normalizers = []
    for s_mode in ["linear", "sqrt", "asinh", "log"]:
        scaling_normalizers.append(
            Normalize(
                lr_max=config["dataset_config"]["lr_max"],
                hr_max=config["dataset_config"]["hr_max"],
                stretch_mode=s_mode,
            )
        )

    callbacks += [
        ImageLogger(
            real_samples=real_display_samples,
            sim_samples=sim_display_samples,
            log_every_n_epochs=config["log_images_every_n_epoch"],
            normalize=val_datamodule.normalize,
            scaling_normalizers=scaling_normalizers,
            data_range=config["dataset_config"]["hr_max"],
        ),
        OnnxExporter(
            in_dims=sim_display_samples[0]["lr"].shape, wandb_logger=wandb_logger
        ),
        MetricLogger(
            normalize=val_datamodule.normalize,
            scaling_normalizers=scaling_normalizers,
            data_range=config["dataset_config"]["hr_max"],
        ),
        checkpoint_callback,
    ]  # see Callbacks section

    if config["resume_from_checkpoint"]:
        # Resume from checkpoint
        trainer = Trainer(
            resume_from_checkpoint=config["resume_model_checkpoint_path"],
            logger=wandb_logger,
            # W&B integration
            log_every_n_steps=config["log_every_n_steps"],  # set the logging frequency
            ## Ivan's change
            accelerator="gpu",
            devices=config["gpus"],
            ##gpus=config["gpus"],
            max_epochs=config["epochs"],  # number of epochs
            deterministic=False,  # keep it deterministic
            fast_dev_run=config["fast_dev_run"],
            callbacks=callbacks,
        )

        if config["run_test_from_checkpoint_only"]:
            trainer.test(
                model,
                dataloaders=test_dataloader,
                ckpt_path=config["resume_model_checkpoint_path"],
            )

            wandb.finish()

            return
    else:
        # Create a new trainer
        # callbacks += [
        #             ImageLogger(val_samples=val_samples, test_samples=test_samples,
        #                          log_every_n_epochs=config['log_images_every_n_epoch'], scaling_normalizers=datamodule.normalize),
        #              OnnxExporter(in_dims=val_samples[0]['lr'].shape, wandb_logger=wandb_logger),
        #              MetricLogger(scaling_normalizers=datamodule.normalize),
        #              checkpoint_callback]  # see Callbacks section

        trainer = Trainer(
            logger=wandb_logger,  # W&B integration
            log_every_n_steps=config["log_every_n_steps"],  # set the logging frequency
            ## Ivan's change
            accelerator="gpu",
            devices=config["gpus"],
            ##gpus=config["gpus"],
            max_epochs=config["epochs"],  # number of epochs
            # deterministic=True,  # keep it deterministic
            benchmark=(not config["debug"]) and True,
            fast_dev_run=config["fast_dev_run"],
            callbacks=callbacks,
        )

    # TODO: remove this!!
    # # # evaluate the model on the test set
    # trainer.test(model, dataloaders=val_dataloader)  # uses last-saved model

    # # fit the model
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    # # evaluate the model on the test set
    trainer.test(
        model, dataloaders=test_dataloader, ckpt_path="best"
    )  # uses last-saved model

    print("Best model saved at:", checkpoint_callback.best_model_path)

    wandb.finish()


if __name__ == "__main__":  # This is needed in order to run it on multiple gpu's
    print(f"Starting run with {config}")

    if not config["resume_run"]:
        wandb.init(
            project=config["project"],
            dir=config["runs_dir"],
            entity=config["wandb_entity"],
            mode=wandb_mode,
            config=config,
        )
    else:
        wandb.init(
            project=config["project"],
            dir=config["runs_dir"],
            entity=config["wandb_entity"],
            mode=wandb_mode,
            config=config,
            resume=config["resume_id"],
        )

    # Config parameters are automatically set by Wandb sweep again
    config = wandb.config

    main(config)
