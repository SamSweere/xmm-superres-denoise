from pathlib import Path

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from datasets import BaseDataModule


class XmmDisplayDataModule(BaseDataModule):
    def __init__(
            self,
            config
    ):
        super(XmmDisplayDataModule, self).__init__(config)
        # Display datasets
        self.sim_display_dataset = None
        self.real_display_dataset = None
        if config["display"]["sim_display_name"]:
            from datasets import XmmSimDataset
            self.sim_display_dataset = XmmSimDataset(
                dataset_dir=Path(config["dir"]) / "display_datasets" / "xmm_sim_display_selection",
                lr_res=config["lr_res"],
                hr_res=config["hr_res"],
                dataset_lr_res=config["dataset_lr_res"],
                mode=config["mode"],
                lr_exps=config["display"]["exposure"],
                hr_exp=config["hr_exp"],
                agn=False,
                lr_background=False,
                hr_background=False,
                det_mask=config["det_mask"],
                check_files=config["check_files"],
                transform=self.transform,
                normalize=self.normalize
            )
        if config["display"]["real_display_name"]:
            from datasets import XmmDataset
            self.real_display_dataset = XmmDataset(
                dataset_dir=Path(config["dir"]) / "display_datasets" / "xmm_split_display_selection",
                dataset_lr_res=config["dataset_lr_res"],
                lr_exps=config["display"]["exposure"],
                hr_exp=None,
                det_mask=config["det_mask"],
                check_files=config["check_files"],
                transform=self.transform,
                normalize=self.normalize
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        raise NotImplementedError

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataloaders = []
        if self.real_display_dataset:
            dataloaders.append(self.get_dataloader(self.real_display_dataset))
        if self.sim_display_dataset:
            dataloaders.append(self.get_dataloader(self.sim_display_dataset))
        return dataloaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()
