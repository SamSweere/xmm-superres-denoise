import pandas as pd
import torch
import pickle
import numpy as np
import argparse

def read_csv_data(filename, quantile_value=0.9999):
    """
    Read statistics data from the CSV file.

    Parameters:
    filename (str): Filename of the CSV file.
    quantile_value (float): Quantile value to extract, default is 0.9999.

    Returns:
    tuple: Tensors for means, variances, maxes, quantiles, and fractions over lr max.
    """
    # Load the CSV file into a DataFrame
    statistics = pd.read_csv(filename)

    # Extract data into PyTorch tensors
    means = torch.tensor(statistics['Means'].values, dtype=torch.float32)
    variances = torch.tensor(statistics['Variances'].values, dtype=torch.float32)
    maxes = torch.tensor(statistics['Maxes'].values, dtype=torch.float32)
    quantiles = torch.tensor(statistics[f'quantile_{quantile_value}'].values, dtype=torch.float32)
    fractions_over_lr_max = torch.tensor(statistics['Fractions_over_lr_max'].values, dtype=torch.float32)

    return means, variances, maxes, quantiles, fractions_over_lr_max

def load_subset_indices(stages, input_type, exposure=None):
    """
    Load the indices defining the train, validation and testing subsets.

    Parameters:
    stages (list): List of strings containing the subset names.
    input_type (str): Type of input data ('sim' or 'real').
    exposure (int, optional): Exposure time of input data from statistics file.

    Returns:
    list: List of PyTorch tensors with subset indices.
    """
    subset_indices = []

    # Determine filename based on input type
    filename = 'img.p' if input_type == 'sim' else f'{exposure}ks.p'

    for stage in stages:
        # Define the path to the indices file
        base_indices_dir = f"res/splits/{input_type}_dataset/{stage}/{filename}"
        
        # Load indices and convert to tensor
        with open(base_indices_dir, "rb") as f:
            subset_indices.append(torch.tensor(pickle.load(f)))

    return subset_indices

def make_gen_split_files(data, args, subset_indices, stages):
    """
    Create split files for the general (non-display) dataset.

    Parameters:
    data (tuple): Data containing means, variances, maxes, quantiles, and fractions over lr max.
    args (): Input arguments from terminal
    subset_indices (list): List of PyTorch tensors with subset indices.
    stages (list): List of strings containing the subset names.
    """
    means, variances, maxes, quantiles, fractions_over_lr_max = data

    # Identify indices above and below the fraction threshold
    above_indices = torch.where(fractions_over_lr_max > args.fraction)[0]
    below_indices = torch.where(fractions_over_lr_max <= args.fraction)[0]

    # Define filenames based on input type
    filename_blw = f'below_{args.exp}ks_{args.pixel_res}px_{args.agns}_img.p' if args.input_type == 'sim' else f'{args.agns}_{args.pixel_res}px_{args.exp}ks_below.p'
    filename_abv = f'above_{args.exp}ks_{args.pixel_res}px_{args.agns}_img.p' if args.input_type == 'sim' else f'{args.agns}_{args.pixel_res}px_{args.exp}ks_above.p'

    # Base directory to save the indices
    base_dir = f"res/splits/{args.input_type}_dataset/"
    
    # Initialize lists for below and above fraction indices
    subset_below_indices = []
    subset_above_indices = []

    for i, subset_index in enumerate(subset_indices):
        # Identify indices belonging to each subset
        below_stage_indices = below_indices[torch.isin(below_indices, subset_index)]
        above_stage_indices = above_indices[torch.isin(above_indices, subset_index)]

        # Save the indices in respective lists
        subset_below_indices.append(below_stage_indices)
        subset_above_indices.append(above_stage_indices)

        # Construct file paths
        below_path = base_dir + f'{stages[i]}/{filename_blw}'
        above_path = base_dir + f'{stages[i]}/{filename_abv}'
        
        # Print the test 
        below_test = torch.sum(fractions_over_lr_max[below_stage_indices]>args.fraction)
        above_test = torch.sum(fractions_over_lr_max[above_stage_indices]<=args.fraction)

        print('Stage: {stage}'.format(stage = stages[i]))
        print('Below test: {below_test} inputs with {fraction} of the image over lr_max'.format(below_test = below_test, fraction = args.fraction))
        print('Above test: {above_test} inputs with {fraction} of the image below lr_max'.format(above_test = above_test, fraction = args.fraction))
        print()

     

        # Save the indices to the file
        with open(below_path, "wb") as f:
            pickle.dump(np.asarray(below_stage_indices), f)

        with open(above_path, "wb") as f:
            pickle.dump(np.asarray(above_stage_indices), f)
            
            
            
def make_dis_split_files(data, args):
    """
    Create split files for the display dataset.

    Parameters:
    data (tuple): Data containing means, variances, maxes, quantiles, and fractions over lr max.
    args (): Input arguments from terminal
    """
    means, variances, maxes, quantiles, fractions_over_lr_max = data

    # Identify indices above and below the fraction threshold
    above_indices = torch.where(fractions_over_lr_max > args.fraction)[0]
    below_indices = torch.where(fractions_over_lr_max <= args.fraction)[0]

    # Define filenames based on input type
    filename_blw = f'below_{args.exp}ks_{args.pixel_res}px_{args.agns}_img.p' if args.input_type == 'sim' else f'{args.agns}_{args.pixel_res}px_{args.exp}ks_below.p'
    filename_abv = f'above_{args.exp}ks_{args.pixel_res}px_{args.agns}_img.p' if args.input_type == 'sim' else f'{args.agns}_{args.pixel_res}px_{args.exp}ks_above.p'

    # Base directory to save the indices
    base_dir = f"res/splits/{args.input_type}_dataset/"
    
    # Construct file paths
    below_path = base_dir + f'{filename_blw}'
    above_path = base_dir + f'{filename_abv}'
    
    # Print the test 
    below_test = torch.sum(fractions_over_lr_max[below_indices]>args.fraction)
    above_test = torch.sum(fractions_over_lr_max[above_indices]<=args.fraction)

    print('Below test: {below_test} inputs with {fraction} of the image over lr_max'.format(below_test = below_test, fraction = args.fraction))
    print('Above test: {above_test} inputs with {fraction} of the image below lr_max'.format(above_test = above_test, fraction = args.fraction))
    print()


    # Save the indices to the file
    with open(below_path, "wb") as f:
        pickle.dump(np.asarray(below_indices), f)

    with open(above_path, "wb") as f:
        pickle.dump(np.asarray(above_indices), f)


def main_gen(args):
    
    # Generate the statistics file path
    statistics_file = f'res/statistics/input_statistics_{args.input_type}_{args.res_name}_{args.pixel_res}pxs_{args.exp}ks_{args.agns}.csv'
    
    stages = ['train', 'val', 'test']
    
    # Load the statistics file
    data = read_csv_data(statistics_file)

    # Load the subset indices
    subset_indices = load_subset_indices(stages, args.input_type, exposure=args.exp)

    # Create and save split files based on the fraction threshold
    make_gen_split_files(data, args, subset_indices, stages)
    
def main_dis(args):
   
    # Generate the statistics file path
    statistics_file = f'res/statistics/input_statistics_{args.input_type}_{args.res_name}_{args.pixel_res}pxs_{args.exp}ks_{args.agns}.csv'
    
    # Load the statistics file
    data = read_csv_data(statistics_file)

    # Create and save split files based on the fraction threshold
    make_dis_split_files(data, args)
    

if __name__ == "__main__":
    # Setup argparse for terminal inputs
    parser = argparse.ArgumentParser(description='Process statistics and subset indices.')
    parser.add_argument('--exp', type=int, default=20, help='Exposure value in ks (default: 20).')
    parser.add_argument('--input_type', type=str, default='sim_display', choices=['sim', 'real', 'sim_display', 'real_display'], help='Input type: "sim" or "real" (default: sim).')
    parser.add_argument('--pixel_res', type=int, default=416, help='Pixel resolution (default: 256).')
    parser.add_argument('--res_name', type=str, default='lr', help='Resolution name (default: lr).')
    parser.add_argument('--agns', type=str, default='blended_agn', help='AGN type (default: no_blended_agn).')
    parser.add_argument('--fraction', type=float, default=0.0, help='Fraction threshold for split (default: 0.0).')

    args = parser.parse_args()

    # Call the main function with parsed arguments
    if args.input_type in ['sim', 'real']:
        main_gen(args)
        
    else: 
        main_dis(args)
