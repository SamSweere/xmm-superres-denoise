from torchvision.transforms import transforms

from datasets.base_datamodule import BaseDataModule
from datasets.tng_dataset import TngDataset


class TNGDataModule(BaseDataModule):
    def __init__(self, config):

        # Prepare the data transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.dataset = TngDataset(
            dataset_name=config["dataset_name"],
            datasets_dir=config["datasets_dir"],
            cache_dir=config["cache_dir"],
            dataset_lr_res=config["dataset_lr_res"],
            dataset_hr_res=config["dataset_hr_res"],
            lr_res=config["lr_res"],
            hr_res=config["hr_res"],
            data_scaling=config["data_scaling"],
            transform=self.transform,
        )

        # Set the super class which does
        super().__init__(config, self.dataset)
