import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as onnxruntime
import torch
from astropy.io import fits
from tqdm import tqdm

from datasets.xmm_datamodule import XmmDataModule
from utils.filehandling import read_yaml, write_xmm_file_to_fits

inference_model_names = ['fine-paper-55', 'worldly-bird-57', 'worldy-butterfly-58', 'distinctive-firebrand-59']

for model_name in inference_model_names:
    print("Running inference on model:", model_name)

    model_base_path = os.path.join('/home/sam/Documents/ESA/data/models/esr_gen/',  model_name)
    onnx_filepath = os.path.join(model_base_path, 'model_final.onnx')
    model_data_config_path = os.path.join(model_base_path, 'data_config.yaml')
    dataset_real_filepath = os.path.join(model_base_path, 'inference_real')
    simulated_dataset_name = 'xmm_demo_dataset'
    simulated_datasets_dir = '/home/sam/Documents/ESA/data/sim'
    # TODO: make this select based on the model config
    dataset_preselect_sim_filepath = '/home/sam/Documents/ESA/data/sim/xmm_demo_dataset/combined/20ks/img/1x'
    dataset_preselect_sim_ref_filepath = '/home/sam/Documents/ESA/data/sim/xmm_demo_dataset/combined/100ks/img/2x'
    dataset_real = 'TODO'
    dataset_mode = 'preselect_sim'  # sim, real

    # # Load the dataset configs
    dataset_config = read_yaml(model_data_config_path)
    dataset_config['debug'] = True  # Set to True when debugging

    dataset_img_path = None
    dataset_ref_path = None

    if dataset_mode == 'sim':
        dataset_filepath = os.path.join(model_base_path, 'inference_sim')
        if not os.path.exists(dataset_filepath):
            os.makedirs(dataset_filepath)
    elif dataset_mode == 'preselect_sim':
        dataset_filepath = os.path.join(model_base_path, 'inference_select_sim')
        if not os.path.exists(dataset_filepath):
            os.makedirs(dataset_filepath)

        if not os.path.exists(dataset_preselect_sim_filepath):
            raise FileExistsError(f"No folder named {dataset_preselect_sim_filepath}")
        if not os.path.exists(dataset_preselect_sim_ref_filepath):
            raise FileExistsError(f"No folder named {dataset_preselect_sim_ref_filepath}")

    elif dataset_mode == 'real':
        dataset_filepath = os.path.join(model_base_path, 'inference_select_sim')
        if not os.path.exists(dataset_filepath):
            os.makedirs(dataset_filepath)

        if not os.path.exists(dataset_real):
            raise FileExistsError(f"No folder named {dataset_real}")

    else:
        raise ValueError(f"Dataset_mode {dataset_mode} is not known")

    # dataset_img_path = os.path.join(dataset_filepath)


    output_path = os.path.join(dataset_filepath, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # input_transformed_path = os.path.join(dataset_filepath, 'input_transformed')
    # if not os.path.exists(input_transformed_path):
    #     os.makedirs(input_transformed_path)
    #
    # output_path = os.path.join(dataset_filepath, 'output')
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    #
    # reference_path = None
    # if simulated_data:
    #     # Also save the reference images
    #     reference_path = os.path.join(dataset_filepath, 'reference')
    #     if not os.path.exists(reference_path):
    #         os.makedirs(reference_path)

    # Creat the xmm data module


    if dataset_mode == 'real':
        dataset_config['batch_size'] = 1  # For inference we use batch sizes of 1

        datamodule = XmmDataModule(dataset_config, dataset_real)

        dataloader = iter(datamodule.full_dataloader())
    elif dataset_mode == 'preselect_sim':
        dataset_config['batch_size'] = 1  # For inference we use batch sizes of 1

        datamodule = XmmDataModule(dataset_config, dataset_preselect_sim_filepath, dataset_preselect_sim_ref_filepath)

        dataloader = iter(datamodule.full_dataloader())
    elif dataset_mode == 'sim':
        #
        # # TODO: overwritten the batch size to 1
        # config['batch_size'] = 10
        # config['debug'] = True # Makes debugging possible
        #
        # dataset_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'dataset',
        #                                    simulated_dataset_name + '.yaml')
        # dataset_config = read_yaml(dataset_config_path)
        dataset_config['dataset_name'] = simulated_dataset_name
        dataset_config['batch_size'] = 1  # For inference we use batch sizes of 1
        dataset_config['datasets_dir'] = simulated_datasets_dir
        # dataset_config['check_files'] = False

        datamodule = XmmDataModule(dataset_config)

        datamodule.setup()

        dataloader = iter(datamodule.test_dataloader())
    else:
        raise ValueError(f"Dataset_mode {dataset_mode} is not known")

        # test_sample = next(test_dataloader)['lr'].detach().numpy()

    # Start the onnx runtime session
    ort_session = onnxruntime.InferenceSession(onnx_filepath)
    input_name = ort_session.get_inputs()[0].name

    # The output exposure is determined by the trained model
    exposure_out = dataset_config['hr_exp'] * 1000

    # Run for every file in the input folder3
    for input_sample in tqdm(dataloader):

        img_in = input_sample['lr'].detach().numpy().astype(np.float32)
        exposure_in = input_sample['lr_exp'].detach().numpy()[0]

        # The dataloader transformed the header, convert it back to a normal dictS
        lr_header_in = None
        if 'lr_header' in input_sample:
            lr_header_in = input_sample['lr_header']
            for key in lr_header_in.keys():
                if type(lr_header_in[key]) == torch.Tensor:
                    lr_header_in[key] = lr_header_in[key].detach().numpy()[0]
                else:
                    lr_header_in[key] = lr_header_in[key][0]

        hr_header_in = None
        if 'hr_header' in input_sample:
            hr_header_in = input_sample['hr_header']
            for key in hr_header_in.keys():
                if type(hr_header_in[key]) == torch.Tensor:
                    hr_header_in[key] = hr_header_in[key].detach().numpy()[0]
                else:
                    hr_header_in[key] = hr_header_in[key][0]

        img_ref = None
        if dataset_mode == 'sim' or dataset_mode == 'preselect_sim':
            img_ref = input_sample['hr'].detach().numpy().astype(np.float32)

        # TODO:
        # elif dataset_mode == 'preselect_sim':
        #     img_ref

        source_file_name = input_sample['lr_img_file_name'][0]
        source_file_name = source_file_name.replace('.fits.gz', '').replace('.fits', '')

        # # TODO: tmp, remove
        # plt.imshow(img_in[0][0])
        # plt.show()

        # Run the image through the model
        ort_inputs = {input_name: img_in}
        ort_outs = ort_session.run(None, ort_inputs)

        img_out = ort_outs[0][0][0]

        # # TODO: tmp, remove
        # plt.imshow(img_out)
        # plt.show()

        # plt.imshow(ort_session.run(None, {input_name: test_sample.astype(np.float32)})[0][0])
        img_in_denorm = datamodule.normalize.denormalize_lr_image(torch.tensor(img_in[0][0])).detach().numpy()
        img_in_denorm = img_in_denorm * exposure_in

        imgs_out_path = os.path.join(output_path, model_name + "_" + source_file_name[:30])  # Limit the length
        if not os.path.exists(imgs_out_path):
            os.makedirs(imgs_out_path)

        write_xmm_file_to_fits(img=img_in_denorm, output_dir=imgs_out_path, source_file_name=source_file_name,
                               res_mult=1, exposure=exposure_in,
                               comment="Transformed Input Image (input to model)",
                               out_file_name='input', in_header=lr_header_in)

        img_out_denorm = datamodule.normalize.denormalize_hr_image(torch.tensor(img_out)).detach().numpy()
        img_out_denorm = img_out_denorm * exposure_out

        # TODO: put more model information in the comment
        write_xmm_file_to_fits(img=img_out_denorm, output_dir=imgs_out_path, source_file_name=source_file_name,
                               res_mult=2, exposure=exposure_out,
                               comment="Model Prediction",
                               out_file_name='prediction', in_header=lr_header_in)

        if dataset_mode == 'sim' or dataset_mode == 'preselect_sim':
            img_ref_denorm = datamodule.normalize.denormalize_hr_image(torch.tensor(img_ref[0][0])).detach().numpy()
            img_ref_denorm = img_ref_denorm * exposure_out
            write_xmm_file_to_fits(img=img_ref_denorm, output_dir=imgs_out_path, source_file_name=source_file_name,
                                   res_mult=2, exposure=exposure_out,
                                   comment="Simulated Reference Image",
                                   out_file_name="reference", in_header=hr_header_in)

    # TODO: running the training dataloader:
    # #TODO: This dataloader is coppied from the train code
    #

    #
    # # Load the run configs
    # run_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_config.yaml')
    # config = read_yaml(run_config_path)
    #
    # # TODO: overwritten the batch size to 1
    # config['batch_size'] = 10
    # config['debug'] = True # Makes debugging possible
    #
    # dataset_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'dataset', config['dataset_name'] + '.yaml')
    # dataset_config = read_yaml(dataset_config_path)
    #
    # # Add the debug to the dataset configs if enabled
    # dataset_config['debug'] = config['debug']
    # dataset_config['datasets_dir'] = config['datasets_dir']
    # dataset_config['batch_size'] = config['batch_size']
    #
    # config['dataset_config'] = dataset_config

    # # Creat the TNG data module
    # # TODO: make this search for the correct datamodule
    # # tng = TNGDataModule(configs['dataset_config'])
    # # xmm = XmmDataModule(configs['dataset_config'])
    # xmm_sim = XmmSimDataModule(config['dataset_config'])
    #
    # datamodule = xmm_sim
    #
    # datamodule.setup()
    #
    # test_dataloader = iter(datamodule.test_dataloader())
    #
    # test_sample = next(test_dataloader)['lr'].detach().numpy()
