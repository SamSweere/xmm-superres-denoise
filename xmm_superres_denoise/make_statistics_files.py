import torch 
import pandas as pd 
from xmm_superres_denoise.datasets.xmm_dataset import XmmDataset
from xmm_superres_denoise.utils.filehandling import read_yaml
from xmm_superres_denoise.datasets import XmmDataModule, XmmDisplayDataModule
from tqdm import tqdm  
import random 

from argparse import ArgumentParser
from pathlib import Path
import pathlib
from torch.utils.data import Dataset


def compute_statistics(
    dataset: Dataset, 
    lr_max: float,
    filename: Path, 
    res: str, 
    fraction: float=1.0
    ):
    """Compute the maxes, means, variances, fractions_over_lr_max and quantile values of the data within the provided dataset.

    Args:
        dataset (XMMDataset): input dataset 
        lr_max (float): clamping threshold for the lr images 
        filename (Path): directory for statistics results
        res (str): resolution of the input images ('lr' and 'hr' for low and high resolution, respectively)
        fraction (float, optional): fraction of the dataset that shall be processed. Defaults to 1.0.
    """
    maxes = []
    means= []
    variances = []
    fractions_over_lr_max = []
    quantiles = []
  
    # Define which quantiles shall be computed
    quantile_values = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999]

    # Determine the number of items to process based on the fraction
    num_items_to_process = int(len(dataset) * fraction)
    i = 0
    
    # Creating the target directory if neccesary 
    if not filename.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)
   
    # Wrap the dataset with tqdm to create a progress bar
    for item in tqdm(dataset, desc="Processing Dataset", unit="item"):
        
        i+= 1
        
        # Extract image with the desired resolution
        item = item[res]
        
        # Compute statistics
        max = torch.max(item)
        mean = torch.mean(item)
        variance = torch.var(item.flatten())
        fraction_over_lr_max = torch.sum(item>lr_max)/item.numel()
        quantile = torch.quantile(item, torch.tensor(quantile_values))

        maxes.append(max.item())
        means.append(mean.item())
        variances.append(variance.item())
        fractions_over_lr_max.append(fraction_over_lr_max.item())
        quantiles.append(quantile)
       
        if i>num_items_to_process:
            break
        
    
    # Convert quantiles to dictionary for easier acceseibility later on 
    quantiles_tensor = torch.stack(quantiles)

    data = {
    'Maxes': maxes,
    'Means': means,
    'Variances': variances,
    'Fractions_over_lr_max': fractions_over_lr_max, 
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Define column titles
    columns = ['quantile_' + str(quantile_value) for quantile_value in quantile_values]
    
    # Convert to data frame
    df_2d = pd.DataFrame(quantiles_tensor.numpy(), columns= columns)
    result_df = pd.concat([df, df_2d], axis=1)
    
   
    result_df.to_csv(filename, index=False)

        
def prepare_statistics(
    dataset_config: dict,
    res: str,
    ):
    """Load the dataset and compute the statistics.

    Args:
        dataset_config (dict): dataset configuration 
        res (str): resolution of the input images ('lr' and 'hr' for low and high resolution, respectively)
    """

    # Load relevant parameters
    input_type = dataset_config["type"]         # datatype (sim or real)
    clamp_th = dataset_config[res]['max']       # clamping threshold in low resolution input 
    img_res = dataset_config[res]["res"]        # image resolution in pixels 
    if res == 'lr':                             # image exposure   #TODO: make sure it also works when more than one lr exposure is given
        exp = dataset_config[res]["exps"][0]        
    elif res == 'hr':
        exp = dataset_config[res]["exp"] 
          
    # Load dataset
    datamodule = XmmDataModule(dataset_config)
    dataset = datamodule.dataset

    filename = Path(f'res/statistics/input_statistics_{input_type}_{res}_{img_res}pxs_{exp}ks.csv')
    print(f'Computing {res} statistics for the {input_type} general dataset')
    compute_statistics(dataset, clamp_th, filename, res, fraction = 1)
    print()



def prepare_display_statistics(dataset_config, res):
    """Load the display dataset and compute the statistics.


    Args:
        dataset_config (dict): dataset configuration 
        res (str): resolution of the input images ('lr' and 'hr' for low and high resolution, respectively)
    """  

    # Load relevant parameters
    input_type = dataset_config["type"]               # datatype (sim or real)
    clamp_th = dataset_config[res]['max']       # clamping threshold in low resolution input 
    img_res = dataset_config[res]["res"]        # image resolution in pixels 
    if res == 'lr':                             # image exposure   #TODO: make sure it also works when more than one lr exposure is given
        exp = dataset_config[res]["exps"][0]        
    elif res == 'hr':
        exp = dataset_config[res]["exp"] 
        
    
    # Load the datasets
    datamodule = XmmDisplayDataModule(dataset_config)
    datasets = []
    dataset_names = ['sim_display', 'real_display']

    if dataset_config["display"]["sim_display_name"]:
        sim_display_dataset = datamodule.sim_display_dataset
        datasets.append(sim_display_dataset)
    
    if dataset_config["display"]["real_display_name"]:
        real_display_dataset = datamodule.real_display_dataset
        datasets.append(real_display_dataset)

    if res == 'lr':
        datasets = datasets
    else:
        # Only using the simulated dataset for the high-resolution case since there are no high resolution images for the real one
        datasets = [sim_display_dataset]

    for i, dataset in enumerate(datasets):
        filename = Path(f'res/statistics/input_statistics_{dataset_names[i]}_{res}_{img_res}pxs_{exp}ks.csv')
        print(f'Computing {res} statistics for the {dataset_names[i]} dataset')
        compute_statistics(dataset, clamp_th, filename, res, fraction = 1)
        print()



if __name__ == "__main__":
    parser = ArgumentParser(prog="", description="")
    parser.add_argument(
        "-p",
        "--config_path",
        type=Path,
        default=pathlib.Path(__file__).parent.parent.resolve() / "res/baseline_config.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=str,
        default='hr',
        help="Wether to consider low resolution (lr) or high resolution (hr) input images",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="gen",
        help="Whether to consider the general ('gen') or display ('dis') dataset",
    )
   
    args = parser.parse_args()

    res = args.resolution
    run_config: dict = read_yaml(args.config_path)
    dataset_config: dict = run_config["dataset"]
    
    # Make sure that no normalization is applied for the statistics and that the entire dataset is considered 
    dataset_config["normalize"]= False
    dataset_config["constant_img_combs"] = True
    # dataset_config["divide_dataset"]= 'all'
     
    if res != 'lr' and res != 'hr':
        raise ValueError("Invalid input for the 'resolution' argument. Valid arguments are 'lr' for low resolution or 'hr' for high resolution")
        
    if args.dataset == 'gen':
        if dataset_config["type"] == 'real':
            print("Warning: There is no high resolution input available for real input data. When computing the statistics of the real input, make sure to adjus the parameters in the config file accordingly")
        prepare_statistics(dataset_config, res)
    elif args.dataset == 'dis':
        prepare_display_statistics(dataset_config, res)
    else: 
        raise ValueError("Invalid input for the 'dataset' argument. Valid arguments are 'gen' for general or 'dis' for display")








 