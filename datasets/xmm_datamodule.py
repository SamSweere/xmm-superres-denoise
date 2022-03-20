from datasets.base_datamodule import BaseDataModule
from datasets.xmm_dataset import XmmDataset
from datasets.xmm_sim_dataset import XmmSimDataset
from transforms.normalize import Normalize
from transforms.crop import Crop
from transforms.totensor import ToTensor


class XmmDataModule(BaseDataModule):
    def __init__(self, config, split='full'):  # , dataset_lr_dir, dataset_hr_dir=None):
        # Prepare the data transforms, note that all these transforms are applied on multiple images
        # thus every transform function needs to accept a list of images
        self.transform = [
            Crop(crop_p=config['lr_res'] / config['dataset_lr_res'], mode=config['crop_mode']),
            ToTensor(),
        ]

        self.normalize = Normalize(lr_max=config['lr_max'], hr_max=config['hr_max'],
                                   stretch_mode=config['data_scaling'])

        if config['dataset_type'] == 'real':
            self.dataset = XmmDataset(dataset_name=config['dataset_name'], datasets_dir=config['datasets_dir'],
                                      split=split,
                                      dataset_lr_res=config['dataset_lr_res'],
                                      lr_exp=config['lr_exp'],
                                      hr_exp=config['hr_exp'],
                                      det_mask=config['det_mask'],
                                      exp_channel=config['exp_channel'],
                                      include_hr=config['include_hr'],
                                      check_files=config['check_files'],
                                      transform=self.transform,
                                      normalize=self.normalize)
        elif config['dataset_type'] == 'sim':
            self.dataset = XmmSimDataset(dataset_name=config['dataset_name'], datasets_dir=config['datasets_dir'],
                                         split=split,
                                         lr_res=config['lr_res'],
                                         hr_res=config['hr_res'],
                                         dataset_lr_res=config['dataset_lr_res'],
                                         mode=config['mode'],
                                         lr_exp=config['lr_exp'],
                                         hr_exp=config['hr_exp'],
                                         lr_agn=config['lr_agn'],
                                         hr_agn=config['hr_agn'],
                                         lr_background=config['lr_background'],
                                         hr_background=config['hr_background'],
                                         det_mask=config['det_mask'],
                                         exp_channel=config['exp_channel'],
                                         check_files=config['check_files'],
                                         transform=self.transform,
                                         normalize=self.normalize
                                         )
        else:
            raise ValueError(f"Dataset type {config['dataset_type']} not known, options: 'real', 'sim' ")

        # Set the super class which does
        super().__init__(config, self.dataset)
