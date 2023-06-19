import numpy as np
import torch
from torch import Tensor, tensor


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
