import os

import numpy as np
import onnxruntime as onnxruntime
import torch
from astropy.io import fits

# from datasets.xmm_datamodule import XmmDataModule
from datasets.utils import reshape_img_to_res, load_fits
from transforms import Crop
from transforms import Normalize
from transforms.totensor import ToTensor
from utils.filehandling import write_xmm_file_to_fits_wcs

#
# hardcoded location of the detector mask
#
cwd = os.path.dirname(os.path.abspath(__file__))


# detmask_file = cwd + '/../datasets/detector_mask/1x/pn_mask_500_2000_detxy_1x.ds'
# if not os.path.isfile(detmask_file):
#     print (f'Detector mask file {detmask_file} not found. Cannot continue!')
#     raise FileNotFoundError

# %%
def run_inference_on_file(fits_file, dataset_config, verbose=True):
    '''
    Purpose:
        Run SR or DN inference on an input FITS file with real XMM-Newton image
    Inputs:
        fits_file - str, 
            input FITS file absolute path
        dataset_config - dict
            The configuration for running the inference (this will dictate if it's SR or DN model)
    Outputs:
        input_filename - str,
            The absolute path to the saved input file
        predicted_filename - str,
            The absolute path to the saved predicted file
            
    Notes:
        * The input FITS file must be in detector coordinates **with good WCS** and shape (403,411), 
        ideally made with xmmsas_tools.make_detxy_image.py
    '''
    #
    # some consistency checks
    #
    if not os.path.isfile(fits_file):
        print(f'Input FITS file {fits_file} not found. Cannot continue!')
        return None
    output_path = os.path.dirname(os.path.abspath(fits_file))
    #
    # raise a warning if the exposure (ONTIME) in the inpt fitsfile is outside 20 ks +/- 5ks
    #
    hdr = fits.getheader(fits_file)
    ontime = hdr['EXPOSURE'] / 1000.0  # in ks
    if (ontime >= 25.0 or ontime <= 15.0):
        print(
            f'Warning: the networks were trained on 20 ks exposure images, the exposure time of the input image is {ontime:.2f} ks.')
    else:
        print(f'Info: the exposure time of the input image is {ontime:.2f} ks.')
    #
    loaded = load_fits(fits_file)
    #
    with fits.open(detmask_file) as det_mask_hdu:
        det_mask = det_mask_hdu[0].data.copy()
    #
    # set up transform and normalize functions, these are adapted for real XMM images
    #
    transform = [
        Crop(crop_p=dataset_config["lr_res"] / dataset_config["dataset_lr_res"], mode=dataset_config["crop_mode"]),
        ToTensor()]
    normalize = Normalize(lr_max=dataset_config["lr_max"], hr_max=dataset_config["hr_max"],
                          stretch_mode=dataset_config["data_scaling"])
    #
    # Load the ONNX file with the model
    #
    onnx_filepath = dataset_config['onnx_path']
    ort_session = onnxruntime.InferenceSession(onnx_filepath)
    input_name = ort_session.get_inputs()[0].name
    # The output exposure is determined by the trained model
    exposure_out = dataset_config["hr_exp"] * 1000
    #
    # now do the conversions and prepare for the inference
    #
    lr_img = loaded["img"]
    lr_exp = loaded["exp"]
    lr_img = reshape_img_to_res(dataset_lr_res=416, img=lr_img, res_mult=1)
    # Make a list to save all the images is, this removes the need for a lot of if statements
    images = [lr_img]
    exp_channel = det_mask * (lr_exp / 1000) / 100
    exp_channel = reshape_img_to_res(dataset_lr_res=416, img=exp_channel, res_mult=1)
    images.append(exp_channel)
    #
    do_transform = True
    # # Apply the transformations
    if do_transform:
        for t in transform:
            images = t(images)
    #
    lr_img = images[0]
    exp_channel = images[1]
    #
    do_normalize = True
    #
    if (do_normalize):
        # Apply the normalization
        # lr_img = normalize.normalize_lr_image(lr_img,lr_img.max())
        lr_img = normalize.normalize_lr_image(lr_img)
    # Torch needs the data to have dimensions [1, x, x]
    lr_img = torch.unsqueeze(lr_img, axis=0)
    # Add the exp channel to the lr_img
    exp_channel = torch.unsqueeze(exp_channel, axis=0)
    lr_img = torch.cat((lr_img, exp_channel), 0)
    #
    sample = {
        "lr": lr_img,
        "lr_exp": loaded["exp"],
        "lr_img_file_name": loaded["file_name"],
        "lr_header": loaded["header"],
    }
    #
    # pytorch dataloader
    #
    dataloader = torch.utils.data.DataLoader([sample], batch_size=1)
    #
    # Run the image through the model
    exposure_in = sample["lr_exp"]
    #
    # some leftover from Sam's approach of processing all images in folder
    #
    for x in dataloader:
        img_in = x["lr"].detach().numpy().astype(np.float32)
        ort_inputs = {input_name: img_in}
        ort_outs = ort_session.run(None, ort_inputs)

        # print (ort_outs[0].shape)

        img_out = ort_outs[0][0][0]
        #
        img_in_denorm = normalize.denormalize_lr_image(torch.tensor(img_in[0][0])).detach().numpy()
        img_in_denorm = img_in_denorm * exposure_in
        img_out_denorm = normalize.denormalize_hr_image(torch.tensor(img_out)).detach().numpy()
        img_out_denorm = img_out_denorm * exposure_out
        # we only have one file, so we break after it's done, just in case
        break
    #
    # save input and predicted
    #
    # input
    if ('fits.gz' in os.path.basename(sample["lr_img_file_name"])):
        input_out_name = f'{sample["lr_img_file_name"].replace(".fits.gz", "")}_input_wcs_{dataset_config["model_name"]}'
    elif ('fits' in os.path.basename(sample["lr_img_file_name"])):
        input_out_name = f'{sample["lr_img_file_name"].replace(".fits", "")}_input_wcs_{dataset_config["model_name"]}'
    # predicted
    pred_out_name = input_out_name.replace('_input_', '_predict_')
    # 
    res = 2
    if ('DN' in dataset_config["model_name"]):
        res = 1
    # input, padded
    write_xmm_file_to_fits_wcs(
        img=img_in_denorm,
        output_dir=output_path,
        source_file_name=sample["lr_img_file_name"],
        res_mult=1,
        exposure=exposure_in,
        comment="Input image padded and WCS aligned",
        out_file_name=input_out_name,
        in_header=sample["lr_header"],
    )
    # output
    write_xmm_file_to_fits_wcs(
        img=img_out_denorm,
        output_dir=output_path,
        source_file_name=sample["lr_img_file_name"],
        res_mult=res,
        exposure=exposure_out,
        comment=f"XMM {dataset_config['model_name']} model prediction",
        out_file_name=pred_out_name,
        # out_file_name="prediction",
        in_header=sample["lr_header"],
    )
    return input_out_name, pred_out_name
