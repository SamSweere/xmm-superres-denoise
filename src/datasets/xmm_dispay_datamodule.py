from torchvision.transforms import transforms

from datasets.base_datamodule import BaseDataModule
from datasets.xmm_sim_dataset import XmmSimDataset
from transforms.crop import Crop
from transforms.normalize import Normalize
from transforms.totensor import ToTensor


class XmmDisplayDataModule(BaseDataModule):
    def __init__(self, config):
        # Prepare the data transforms, note that all these transforms are applied on multiple images
        # thus every transform function needs to accept a list of images
        self.transform = [
            Crop(
                crop_p=config["lr_res"] / config["dataset_lr_res"],
                mode=config["crop_mode"],
            ),
            ToTensor(),
        ]

        self.normalize = Normalize(
            lr_max=config["lr_max"],
            hr_max=config["hr_max"],
            stretch_mode=config["data_scaling"],
        )

        # Use the XmmSimDataset for only loading the images (which already contain the agn's and background)
        self.dataset = XmmSimDataset(
            dataset_name=config["dataset_name"],
            datasets_dir=config["datasets_dir"],
            split="full",
            lr_res=config["lr_res"],
            hr_res=config["hr_res"],
            dataset_lr_res=config["dataset_lr_res"],
            mode=config["mode"],
            lr_exp=config["lr_exp"],
            hr_exp=config["hr_exp"],
            lr_agn=False,
            hr_agn=False,
            lr_background=False,
            hr_background=False,
            det_mask=config["det_mask"],
            exp_channel=config["exp_channel"],
            check_files=config["check_files"],
            transform=self.transform,
            normalize=self.normalize,
        )

        # Set the super class which does
        super().__init__(config, self.dataset)
