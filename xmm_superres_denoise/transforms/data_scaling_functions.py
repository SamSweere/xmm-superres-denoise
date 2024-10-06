import numpy as np
import torch
from torch import Tensor, tensor
import cv2
from pytorch_lightning.utilities import rank_zero_warn


def linear_scale(x, inverse=False):
    # This function does not do anything, but simplifies the architecture
    return x


def sqrt_scale(x, inverse=False):
    if type(x) == Tensor:
        return torch.pow(x, 2) if inverse else torch.sqrt(x)
    else:
        return np.power(x, 2) if inverse else np.sqrt(x)


def asinh_scale(x, a=0.02, inverse=False):
    if type(x) == Tensor:
        a = tensor(a)
        if inverse:
            return a * torch.sinh(x * torch.arcsinh(1.0 / a))
        else:
            return torch.arcsinh(x / a) / torch.arcsinh(1.0 / a)
    else:
        if inverse:
            return a * np.sinh(x * np.arcsinh(1.0 / a))
        else:
            return np.arcsinh(x / a) / np.arcsinh(1.0 / a)


# http://ds9.si.edu/doc/ref/how.html
def log_scale(x, a=1000, inverse=False):
    if type(x) == Tensor:
        a = tensor(a)
        if inverse:
            return (a**x - 1) / a
        else:
            return torch.log(a * x + 1) / torch.log(a)
    else:
        if inverse:
            return (a**x - 1) / a
        else:
            return np.log(a * x + 1) / np.log(a)


# added by yvonne 
#TODO: figure out how to define the inverse 
def hist_eq_scale(x, clipLimit=2.0, tileGridSize=(4,4), inverse = False):
    """ Stretch the input data such that its histogram approaches an even distribution

        Parameters: 
            x (torch.Tensor): input 
            clipLimit (float): input argument for cv2's CLAHE function
            tileGridSize (float): input argument for cv2's CLAHE function 
    """


    if len(x.shape)!=4:
        raise ValueError(f"CLAHE is only implemented for n = 4 input dimensions")
    
    rank_zero_warn(
            "There is no inverse for the CLAHE histogram equalization implemented yet and denorming has no effect!"
            )
    
    #TODO: figure out why there is inputs with first dimensions > 1 (is it just the bactch size?) and adjust loop accordingly 
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    x_out = torch.zeros_like(x)

    for i in range(len(x)):
        cv2_input = np.array(255*x[i].cpu()).astype(np.uint8).reshape(*x.shape[-2:])
        img_clahe = clahe.apply(cv2_input)
        x_out[i] = torch.tensor(img_clahe/255).unsqueeze(0)
    
    return x_out