
import pandas as pd
import torch 
import matplotlib.pyplot as plt 
from xmm_superres_denoise.utils.filehandling import read_yaml
import numpy as np 
from matplotlib.ticker import FuncFormatter
import argparse

from make_split_files import read_csv_data, load_subset_indices


def apply_to_subsets(data, subset_indices):
    """Extract the data for each subset (train, val and test) from the overall dataset

            Parameters: 
                data (torch.tensor): data from which the subsets shall be extracted 
                subset_indices (torch.tensor): indices defining the subsets from the overall dataset 
    """

    # Create list to collect results 
    results = []
    
    # Extract results for each subset from overall data 
    for subset_index in subset_indices:
        results.append(data[subset_index])

    return results 



def plot_pie_chart(fractions_over_lr_max_all, input_type, subset_indices, stages, fraction, dis = False):

    if dis == False:
        # Compute the fractions_over_lr_max for each subset 
        subset_fractions_over_lr_max = apply_to_subsets(fractions_over_lr_max_all, subset_indices)

        # Add the complete dataset in the frist entry for comparison 
        subset_fractions_over_lr_max.insert(0, fractions_over_lr_max_all)
            
    else: 
        subset_fractions_over_lr_max = [fractions_over_lr_max_all]
    
    fig, ax = plt.subplots(len(subset_fractions_over_lr_max), 1, figsize = (12, 4), squeeze = False)
    plt.subplots_adjust(hspace = 0.9)
   
   
    for i, fractions_over_lr_max in enumerate(subset_fractions_over_lr_max):

        # compute how many inputs contain pixels brighther than lr_max
        num_abv_lr_max = torch.sum(fractions_over_lr_max>fraction)
        num_blw_lr_max = torch.sum(fractions_over_lr_max<=fraction)

        # Data to be represented in the pie chart
        sizes = [num_abv_lr_max, num_blw_lr_max]  # Sizes of each slice
        explode = (0, 0)  # Explode the second slice (B)

        # Create a pie chart
        ax[i,0].pie(sizes, explode=explode, labels=None, autopct='%1.1f%%', startangle=140)

        num_samples = fractions_over_lr_max.numel()
        ax[i,0].text(-1, -1.2, "Total number of samples: {num_samples}".format(num_samples = num_samples))

        # Set aspect ratio to make it a circle, not an ellipse
        ax[i,0].axis('equal')

        # Title for the pie chart
        plt.rcParams['axes.titley'] = 0.85
        ax[i,0].set_title(stages[i])
    
    fig.legend(['Above lr_max', 'Below lr_max'], loc='center', title='Legend', bbox_to_anchor=(0.5, 0.08), bbox_transform=fig.transFigure, ncol = 2)
   
    fig.savefig(f'statistics_plots/{input_type}/pie_charts/pie_chart_fraction_{fraction}_{input_type}_{args.res_name}_{args.pixel_res}px_{args.exp}ks.pdf')
    
    
def make_history(data, args, stages, subset_indices, measure_name, i, dis = False):

    if dis == False:
        # Obtain data for subsets
        subset_data = apply_to_subsets(data, subset_indices)

        # Add the complete dataset in the frist entry for comparison 
        subset_data.insert(0, data)
        
    else: 
        subset_data = [data]
    
    fig, ax = plt.subplots(len(subset_data), 1, figsize = (10, 13), squeeze = False)
    plt.subplots_adjust(hspace = 0.4)

    if i == 2: 
        hline = True 
    else: 
        hline = False

    for i, subset_data in enumerate(subset_data):

        # plot the data 
        ax[i,0].plot(subset_data)
        ax[i,0].set_xlabel("Index [-]")
        ax[i,0].set_ylabel(measure_name + '[-]')
        ax[i,0].set_title(stages[i])

        if hline:
            # plot horizontal line
            ax[i,0].axhline(args.res_name_max, 0, len(data), color = 'r')
            ax[i,0].text(0, args.res_name_max - 0.01, 'clamping threshold', 
           color='r', ha='center', va='top', transform=ax[i,0].get_xaxis_transform())

    fig.savefig(f'statistics_plots/{args.input_type}/histories/history_{measure_name}_{args.input_type}_{args.res_name}_{args.pixel_res}px_{args.exp}ks.pdf')
    plt.close()

def make_histograms(data, subset_indices, input_type, measure_name, stages, num_bins = 20, yscale = 'log', num_digits = 4, dis = False):

   # Compute bin edges for complete dataset for easier comparison between subsets
    _, bin_edges = np.histogram(data, bins=num_bins)
    
    # _, bin_edges = np.geomspace(np.min(data_all), np.max(data_all), num_bins + 1)
   
    # Get the subset data 
    if dis == False:
        # Obtain data for subsets
        subset_data = apply_to_subsets(data, subset_indices)

        # Add the complete dataset in the frist entry for comparison 
        subset_data.insert(0, data)
        
    else: 
        subset_data = [data]
    
    fig, ax = plt.subplots(len(subset_data), 1, figsize = (11, 16), squeeze = False)
    plt.subplots_adjust(hspace = 0.4)


    for i, data in enumerate(subset_data):

        # Create histograms for both groups with the same bins
        hist, _ = np.histogram(data, bins=bin_edges)
        
        # Calculate the bin width
        bin_width = bin_edges[1] - bin_edges[0]

        # Define the x-positions for the bars
        bar_stretch = 3
        x_positions = bar_stretch*bin_edges[:-1]

        # Plot the bars 
        ax[i,0].bar(x_positions, hist, width=bin_width, alpha=0.5, label='above lr_max')
        
        # Compute and set the x-ticks 
        x_ticks = x_positions
        # labels = bin_edges[:-1]
        labels = [f"{edge:.6f}" for edge in bin_edges[:-1]]
        ax[i,0].set_xticks(x_ticks, labels, rotation = 90)

        # Label the x-axis
        ax[i,0].set_xlabel(measure_name + ' [-]')
        ax[i,0].set_ylabel('Frequency [-]')
        ax[i,0].set_yscale(yscale)
        ax[i,0].set_title('{measure_name} {stage}'.format(measure_name = measure_name, stage = stages[i]))


        # Define a custom function for formatting the labels
        def format_labels(x, pos, num_digits=num_digits):
            # Format the labels with the specified number of digits
            formatted_label = f"{x:.{num_digits}f}"
            return formatted_label

        # Create a custom formatter for the x-axis
        num_digits = num_digits  # Change this value to the desired number of digits
        formatter = FuncFormatter(lambda x, pos: format_labels(x, pos, num_digits))
        ax[i,0].xaxis.set_major_formatter(formatter)

    # Save the figure
    fig.savefig(f'statistics_plots/{input_type}/histograms/histogram_{measure_name}_{input_type}__{args.res_name}_{args.pixel_res}px_{args.exp}ks.pdf')
    plt.close()

def make_gen_statistics_plots(data_all, measure_names, args):
    
    stages = ['all', 'train', 'val', 'test']
    

    subset_indices = load_subset_indices(stages[1:], args.input_type, exposure=args.exp)

    for i, data in enumerate(data_all):
        make_history(data, args, stages, subset_indices, measure_names[i], i)
        make_histograms(data, subset_indices, args.input_type, measure_names[i], stages)
    
    # Create a pie chart showing the overall distribution of data 
    for fraction in [0, 0.01, 0.05, 0.1]:
        plot_pie_chart(data_all[-1], args.input_type, subset_indices, stages, fraction)
        
        
def make_dis_statistics_plots(data_all, measure_names, args):
    
    stages = ['all']
    
    subset_indices = [torch.arange(len(data_all[0]))]

    for i, data in enumerate(data_all):
        make_history(data, args, stages, subset_indices, measure_names[i], i, dis = True)
        make_histograms(data, subset_indices, args.input_type, measure_names[i], stages, dis = True)
    
    # Create a pie chart showing the overall distribution of data 
    for fraction in [0, 0.01, 0.05, 0.1]:
        plot_pie_chart(data_all[-1], args.input_type, subset_indices, stages, fraction, dis = True)
    


if __name__ == "__main__":
    # Setup argparse for terminal inputs
    parser = argparse.ArgumentParser(description='Process statistics and subset indices.')
    parser.add_argument('--exp', type=int, default=100, help='Exposure value in ks (default: 20).')
    parser.add_argument('--input_type', type=str, default='real', choices=['sim', 'real', 'sim_display', 'real_display'], help='Input type: "sim" or "real" (default: sim).')
    parser.add_argument('--pixel_res', type=int, default=256, help='Pixel resolution (default: 256).')
    parser.add_argument('--res_name', type=str, default='hr', help='Resolution name (default: lr).')
    parser.add_argument('--agns', type=str, default='no_blended_agn', help='AGN type (default: no_blended_agn).')
    parser.add_argument('--res_name_max', type=float, default='0.0022336', help='Clamping threshold for the chosen resolution (lr_max) or hr_max')
 
    args = parser.parse_args()

    # Generate the statistics file path
    statistics_file = f'res/statistics/input_statistics_{args.input_type}_{args.res_name}_{args.pixel_res}pxs_{args.exp}ks_{args.agns}.csv'

    # Load the data 
    data_all = read_csv_data(statistics_file)
    measure_names = ['Mean', 'Variance', 'Max', '0.9999 Quantile', 'Fraction over lr_max']
    
    if args.input_type in ['sim', 'real']:
        make_gen_statistics_plots(data_all, measure_names, args)
        
    else:
        make_dis_statistics_plots(data_all, measure_names, args)
            

   




