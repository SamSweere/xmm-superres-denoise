from pathlib import Path
import pickle
from torch.utils.data import Subset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from xmm_superres_denoise.datasets import BaseDataModule
from xmm_superres_denoise.transforms import  Normalize
import pandas as pd


from pytorch_lightning.utilities import rank_zero_warn


class XmmDisplayDataModule(BaseDataModule):
    def __init__(self, config):
        super(XmmDisplayDataModule, self).__init__(config)

        check_files = config["check_files"]
        dataset_lr_res = config["lr"]["res"]
        lr_exps = config["display"]["exposure"]
        hr_exp = config["hr"]["exp"]
        det_mask = config["det_mask"]
        self.divide_dataset = config["divide_dataset"]
        
        if config["normalize"]:
            blended_agn = 'blended_agn' if config['deblend_agn_dir'] else 'no_blended_agn'
            
            self.sim_lr_statistics_path = f'res/statistics/input_statistics_sim_display_lr_{config["lr"]["res"]}pxs_{lr_exps[0]}ks_{blended_agn}.csv'
            self.sim_hr_statistics_path = f'res/statistics/input_statistics_sim_display_hr_{config["hr"]["res"]}pxs_{hr_exp}ks_{blended_agn}.csv'
            self.real_lr_statistics_path = f'res/statistics/input_statistics_real_display_lr_{config["lr"]["res"]}pxs_{lr_exps[0]}ks_{blended_agn}.csv'

            self.sim_normalize = Normalize(
                lr_max=config["lr"]["max"],
                hr_max=config["hr"]["max"],
                config = config,
                lr_statistics= pd.read_csv(self.sim_lr_statistics_path), 
                hr_statistics= pd.read_csv(self.sim_hr_statistics_path), 
                stretch_mode=config["scaling"],
                clamp = config["clamp"],
                sigma_clamp = config["sigma_clamp"],
                quantile_clamp = config["quantile_clamp"],
            )

            self.real_normalize = Normalize(
                lr_max=config["lr"]["max"],
                hr_max=config["hr"]["max"],
                config = config,
                lr_statistics= pd.read_csv(self.real_lr_statistics_path), 
                stretch_mode=config["scaling"],
                clamp = config["clamp"],
                sigma_clamp = config["sigma_clamp"],
                quantile_clamp = config["quantile_clamp"]
            )

        else:
            
            self.sim_normalize = False 
            self.real_normalize = False

        

        # Display xmm_superres_denoise.datasets
        self.sim_display_dataset = None
        self.real_display_dataset = None
        if config["display"]["sim_display_name"]:
            from xmm_superres_denoise.datasets import XmmSimDataset

            dataset_dir = (
                Path(config["dir"])
                / "display_datasets"
                / f"{config['display']['sim_display_name']}"
            )
            
            if config["display"]["comps"]:
                
                rank_zero_warn(
                "You are using the 'composed display image' option, make sure to use the corresponding display dataset."
                )
                
                lr_agn, hr_agn, lr_background, hr_background = config["lr"]["agn"], config["hr"]["agn"], config["lr"]["background"], config["hr"]["background"]
            
            else: 
                lr_agn, hr_agn, lr_background, hr_background = False, False, False, False
            
            self.sim_display_dataset = XmmSimDataset(
                dataset_dir=dataset_dir,
                lr_res=config["lr"]["res"],
                hr_res=config["hr"]["res"],
                dataset_lr_res=dataset_lr_res,
                mode=config["mode"],
                lr_exps=lr_exps,
                hr_exp=hr_exp,
                lr_agn=lr_agn,
                hr_agn=hr_agn,
                lr_background=lr_background,
                hr_background=hr_background,
                det_mask=det_mask,
                constant_img_combs = config["constant_img_combs"],
                check_files=check_files,
                transform=self.transform,
                normalize=self.sim_normalize,
            )
            
            if self.divide_dataset == 'below' or self.divide_dataset == 'above':
                self.sim_subset_str = f"res/splits/sim_display_dataset/no_blended_agn_{config['lr']['res']}px_{config['lr']['exps'][0]}ks_{self.divide_dataset}.p"
                test = 5
        if config["display"]["real_display_name"]:
            from xmm_superres_denoise.datasets import XmmDataset

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
                normalize=self.real_normalize,
            )
            
            if self.divide_dataset == 'below' or self.divide_dataset == 'above':
                self.real_subset_str = f"res/splits/real_display_dataset/no_blended_agn_{config['lr']['res']}px_{config['lr']['exps'][0]}ks_{self.divide_dataset}.p"
                test = 5
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        raise NotImplementedError(
            "Display xmm_superres_denoise.datasets are not to be used during training!"
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataloaders = []
        if self.real_display_dataset:
                if self.divide_dataset != 'all':
                    with open(self.real_subset_str, "rb") as f:
                        real_display_indices = pickle.load(f)
                    self.real_display_sub_dataset = Subset(self.real_display_dataset, real_display_indices)
                    dataloaders.append(self.get_dataloader(self.real_display_sub_dataset))
            
                else:
                    dataloaders.append(self.get_dataloader(self.real_display_dataset))
                
                
        if self.sim_display_dataset:
            if self.divide_dataset != 'all':
                with open(self.sim_subset_str, "rb") as f:
                    sim_display_indices = pickle.load(f)
                self.sim_display_sub_dataset = Subset(self.sim_display_dataset, sim_display_indices)
                dataloaders.append(self.get_dataloader(self.sim_display_sub_dataset))
            else:
                dataloaders.append(self.get_dataloader(self.sim_display_dataset))

        return dataloaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()
