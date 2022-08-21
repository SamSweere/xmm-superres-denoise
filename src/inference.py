import os

import numpy as np
import onnxruntime
import torch
from tqdm import tqdm

from src.datasets.xmm_datamodule import XmmDataModule
from src.utils.filehandling import read_yaml, write_xmm_file_to_fits


def do_inference(
    model_path, model_data_config_path, output_path, check_files=True, debug=False
):
    # Load the dataset configs
    dataset_config = read_yaml(model_data_config_path)

    # Extract the model name from the model path
    model_name = os.path.basename(model_path).replace(".onnx", "")

    # If the output path does not exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # TODO: check if this is needed
    dataset_config["batch_size"] = 1  # For inference, we use batch sizes of 1
    dataset_config["include_hr"] = False
    dataset_config["check_files"] = check_files
    dataset_config["debug"] = debug

    # Load the datamodule
    datamodule = XmmDataModule(dataset_config)
    dataloader = iter(datamodule.get_dataloader())

    ####### TODO: new function

    # Start the onnx runtime session
    ort_session = onnxruntime.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name

    # The output exposure is determined by the trained model
    exposure_out = dataset_config["hr_exp"] * 1000

    # Run for every file in the input folder
    for input_sample in tqdm(dataloader):

        img_in = input_sample["lr"].detach().numpy().astype(np.float32)
        exposure_in = input_sample["lr_exp"].detach().numpy()[0]

        # The dataloader transformed the header, convert it back to a normal dictS
        lr_header_in = None
        if "lr_header" in input_sample:
            lr_header_in = input_sample["lr_header"]
            for key in lr_header_in.keys():
                if type(lr_header_in[key]) == torch.Tensor:
                    lr_header_in[key] = lr_header_in[key].detach().numpy()[0]
                else:
                    lr_header_in[key] = lr_header_in[key][0]

        # TODO: check if you want to keep this in inference.py
        hr_header_in = None
        if "hr_header" in input_sample:
            hr_header_in = input_sample["hr_header"]
            for key in hr_header_in.keys():
                if type(hr_header_in[key]) == torch.Tensor:
                    hr_header_in[key] = hr_header_in[key].detach().numpy()[0]
                else:
                    hr_header_in[key] = hr_header_in[key][0]

    source_file_name = input_sample["lr_img_file_name"][0]
    source_file_name = source_file_name.replace(".fits.gz", "").replace(".fits", "")

    # Run the image through the model
    ort_inputs = {input_name: img_in}
    ort_outs = ort_session.run(None, ort_inputs)

    img_out = ort_outs[0][0][0]

    img_in_denorm = (
        datamodule.normalize.denormalize_lr_image(torch.tensor(img_in[0][0]))
        .detach()
        .numpy()
    )
    img_in_denorm = img_in_denorm * exposure_in

    imgs_out_path = os.path.join(
        output_path, model_name + "_" + source_file_name[:30]
    )  # Limit the length

    # Create the directory to save the transformed input file and predictions in
    if not os.path.exists(imgs_out_path):
        os.makedirs(imgs_out_path)

    write_xmm_file_to_fits(
        img=img_in_denorm,
        output_dir=imgs_out_path,
        source_file_name=source_file_name,
        res_mult=1,
        exposure=exposure_in,
        comment="Transformed Input Image (input to model)",
        out_file_name="input",
        in_header=lr_header_in,
    )

    # For the hr image
    img_out_denorm = (
        datamodule.normalize.denormalize_hr_image(torch.tensor(img_out))
        .detach()
        .numpy()
    )
    img_out_denorm = img_out_denorm * exposure_out

    # TODO: put more model information in the comment
    write_xmm_file_to_fits(
        img=img_out_denorm,
        output_dir=imgs_out_path,
        source_file_name=source_file_name,
        res_mult=2,
        exposure=exposure_out,
        comment="Model Prediction",
        out_file_name="prediction",
        in_header=lr_header_in,
    )

    # if dataset_config['dataset_type'] == 'real':
    #     datamodule = XmmDataModule(dataset_config, dataset_real)
    # elif dataset_config['dataset_type'] == 'sim':
    #     pass
    # else:
    #     raise ValueError(f"key error in dataset_type: {dataset_config['dataset_type']}")
    #
    # datamodule = XmmDataModule(dataset_config, dataset_real)


if __name__ == "__main__":
    model_path = "../models/XMM-SuperRes.onnx"
    model_data_config_path = "../models/XMM-SuperRes_real_data_config.yaml"
    # data_path = "../example_data/sim/20ks"
    output_path = "../data/example_data/output/xmm_superres_preds"

    do_inference(
        model_path=model_path,
        model_data_config_path=model_data_config_path,
        # data_path=data_path,
        output_path=output_path,
        check_files=False,
        debug=True,
    )
