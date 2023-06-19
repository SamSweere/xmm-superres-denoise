import torch
from torch.utils.data import Dataset


class BoringDataset(Dataset):
    def __init__(
        self,
        lr_exp: int = 20,
        hr_exp: int = 100,
        hr_res_mult: int = 2,
        dataset_size: int = 10000,
    ):
        super(BoringDataset, self).__init__()
        self.lr_exp = lr_exp
        self.hr_exp = hr_exp
        self.hr_res_mult = hr_res_mult
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return {
            "lr": torch.randn(1, 416, 416),
            "hr": torch.randn(1, 416 * self.hr_res_mult, 416 * self.hr_res_mult),
            "lr_exp": self.lr_exp,
            "hr_exp": self.hr_exp,
            "lr_img_filename": f"boring_dataset_{idx}",
        }
