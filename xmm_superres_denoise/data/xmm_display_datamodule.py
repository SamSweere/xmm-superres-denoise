from pathlib import Path

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from xmm_superres_denoise.data import BaseDataModule


class XmmDisplayDataModule(BaseDataModule):
    def __init__(self, config):
        super(XmmDisplayDataModule, self).__init__(config)

        check_files = config["check_files"]
        dataset_lr_res = config["lr"]["res"]
        lr_exps = config["display"]["exposure"]
        det_mask = config["det_mask"]

        # Display xmm_superres_denoise.data
        self.sim_display_dataset = None
        self.real_display_dataset = None
        if config["display"]["sim_display_name"]:
            from xmm_superres_denoise.data import XmmSimDataset

            dataset_dir = (
                Path(config["dir"])
                / "display_datasets"
                / f"{config['display']['sim_display_name']}"
            )
            self.sim_display_dataset = XmmSimDataset(
                dataset_dir=dataset_dir,
                lr_res=config["lr"]["res"],
                hr_res=config["hr"]["res"],
                dataset_lr_res=dataset_lr_res,
                mode=config["mode"],
                lr_exps=lr_exps,
                hr_exp=config["hr"]["exp"],
                lr_agn=False,
                hr_agn=False,
                lr_background=False,
                hr_background=False,
                det_mask=det_mask,
                check_files=check_files,
                transform=self.transform,
                normalize=self.normalize,
            )
        if config["display"]["real_display_name"]:
            from xmm_superres_denoise.data import XmmDataset

            dataset_dir = (
                Path(config["dir"])
                / "display_datasets"
                / f"{config['display']['real_display_name']}"
            )
            self.real_display_dataset = XmmDataset(
                dataset_dir=dataset_dir,
                dataset_lr_res=dataset_lr_res,
                lr_exps=lr_exps,
                hr_exp=None,
                det_mask=det_mask,
                check_files=check_files,
                transform=self.transform,
                normalize=self.normalize,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        raise NotImplementedError(
            "Display xmm_superres_denoise.data are not to be used during training!"
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataloaders = []
        if self.real_display_dataset:
            dataloaders.append(self.get_dataloader(self.real_display_dataset))
        if self.sim_display_dataset:
            dataloaders.append(self.get_dataloader(self.sim_display_dataset))
        return dataloaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()
