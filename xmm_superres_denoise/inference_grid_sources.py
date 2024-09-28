from pathlib import Path
from xmm_superres_denoise.utils.run_inference_on_file import run_on_file
from shifted_grid_sources.utils.utils import create_directory_if_not_exists
import os 
from pytorch_lightning.utilities import rank_zero_warn
import argparse


def main(clamping_tr_nn, res, exp, base_path, run_config, checkpoint, downscales, br_diffs_scales, shift_amounts):
    
    grid_dirs = [
        f'test_grid_f_1e-13_step_10_mult_{res}_{exp}ks_p_0-4.fits.gz',
        f'test_grid_f_1e-13_step_10_mult_{res}_{exp}ks_p_1-4.fits.gz',
        f'test_grid_f_1e-13_step_10_mult_{res}_{exp}ks_p_2-4.fits.gz',
        f'test_grid_f_1e-13_step_10_mult_{res}_{exp}ks_p_3-4.fits.gz',
        f'test_grid_f_1e-13_step_10_mult_{res}_{exp}ks_p_4-4.fits.gz'
    ]

    grid_dir = grid_dirs[0]

    for ds in downscales:
        print(f'Performing inference for downscale factor {ds}')
        for br_diff in br_diffs_scales:
            for shift_amount in shift_amounts:
                for input_kind in ['original_grids', 'shifted_grids']:
                    if input_kind == 'original_grids':
                        # Define path based on current shift
                        base_input_path = Path(base_path + f'{input_kind}/input/{res}x_{ds:.2f}ds')
                        base_predictions_path = Path(base_path + f'{input_kind}/predictions/{res}x_{ds:.2f}ds')
                        fits_name_end = Path(f'original_{grid_dir}')
                    else:
                        # Define path based on current shift
                        base_input_path = Path(base_path + f'{input_kind}/input/br_diff_sc{br_diff}/{res}x_{ds:.2f}ds/shift{shift_amount}')
                        base_predictions_path = Path(base_path + f'{input_kind}/predictions/br_diff_sc{br_diff}/{res}x_{ds:.2f}ds/shift{shift_amount}')
                        fits_name_end = Path(f'shifted_{grid_dir}')

                    create_directory_if_not_exists(base_predictions_path)

                    fits_path = base_input_path / fits_name_end

                    run_on_file(
                        fits_file=fits_path,
                        checkpoint=checkpoint,
                        out=base_predictions_path,
                        run_config=run_config,
                        plot=True,
                        clamping_tr_nn=clamping_tr_nn,
                        shift_amount=shift_amount
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on FITS files with specified parameters.')

    parser.add_argument('--clamping_tr_nn', type=str, default='original', help='Name of the employed clamping')
    parser.add_argument('--res', type=int, default=1, help='Resolution factor of the input file.')
    parser.add_argument('--exp', type=int, default=20, help='Exposure time of the input file in ks.')
    parser.add_argument('--base_path', type=str, default='/home/xmmsas/mywork/cleanup_new/xmm-superres-denoise/shifted_grid_sources/results/', help='Base directory for results.')
    parser.add_argument('--run_config', type=Path, default=Path('/home/xmmsas/mywork/cleanup_new/xmm-superres-denoise/res/baseline_config.yaml'), help='Path to run configuration YAML file.')
    parser.add_argument('--checkpoint', type=Path, default=Path('/home/xmmsas/mywork/cleanup_new/xmm-superres-denoise/models/original_model_epoch_49_esr_gen_2024-09-23_06-41-32.ckpt'), help='Path to model checkpoint file.')
    parser.add_argument('--downscales', nargs='+', type=float, default=[0.25, 0.5, 0.75, 1], help='List of downscale factors.')
    parser.add_argument('--br_diffs_scales', nargs='+', type=float, default=[1], help='List of brightness difference scales.')
    parser.add_argument('--shift_amounts', nargs='+', type=int, default=[0, 1, 2, 3, 4], help='List of shift amounts.')

    args = parser.parse_args()
    
    rank_zero_warn(
        "This script uses the run_on_file function which uses settings from the loss_functions.yaml file. "
        "Make sure that it contains the desired parameter values"
    )
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

    main(
        args.clamping_tr_nn, 
        args.res, 
        args.exp, 
        args.base_path, 
        args.run_config, 
        args.checkpoint, 
        args.downscales, 
        args.br_diffs_scales, 
        args.shift_amounts
    )
