import torch
import numpy as np

from pytorch_lightning.utilities import rank_zero_warn
from xmm_superres_denoise.transforms.data_scaling_functions import (
    asinh_scale,
    linear_scale,
    log_scale,
    sqrt_scale,
    hist_eq_scale,
)


class Normalize(object):
    """Normalize and image optionally based on a strectching function. First apply the strectching function then normalize to max = 1.
        The minimum possible value is 0 independent of the image, the image is thus only normalized based on the max value.
        Returns the normalized image and the max value

    Args:
        lr_max (float): Maximum value for lr images
        hr_max (float): Maximum value for hr images
        config (): Dataset configuration parameters 
        stretch_mode (string) (optional) : The stretching function options: linear, sqrt, asinh, log, hist_eq
        clamp (bool): If True, bright regions in the images are clamped according to lr_max/ hr_max


    """

    def __init__(self, lr_max, hr_max,  config, lr_statistics, hr_statistics = None, stretch_mode="linear"):
        assert isinstance(stretch_mode, str)

        # I am now passing all of the configs for the normalization paraemters, so we don't need to pass the individual arguments anymore, but I don't want to mess with this for now 
        self.stretch_mode = stretch_mode
        self.lr_max = lr_max
        self.hr_max = hr_max
        
        self.input_clamp = config["input_clamp"]
        self.target_clamp= config["target_clamp"]
        
        
        if self.input_clamp != self.target_clamp:
            
            rank_zero_warn(
                "You are using different values for clamping input and target images. This is only recommended during testing for evaluation purposes!"
            )
            
        self.target_norm = config["target_norm"]
        self.sigma_clamp = config["sigma_clamp"]
        self.quantile_clamp = config["quantile_clamp"]
        
        self.lr_statistics = lr_statistics
        self.hr_statistics = hr_statistics
        
        
        # Compute the max-value used for normalization based on the clamping technique
        self.lr_max_vals = self.compute_max_vals(lr_statistics, lr_max, self.input_clamp)
        if hr_statistics is not None:
            self.hr_max_vals = self.compute_max_vals(hr_statistics, hr_max, self.target_clamp)
            self.target_norms = self.compute_norm_vals(hr_statistics, self.target_norm, self.hr_max_vals, hr_max)

        else:
            # If the statistics for the hr image are not avaible (there are no hr images for the real display dataset), simply use a quarter of lr_max
            self.hr_max_vals = self.lr_max_vals/4
            self.target_norms = self.lr_max_vals/4
            
        self.stretch_f = None
        if stretch_mode == "linear":
            self.stretch_f = linear_scale
            self.args = ()
        elif stretch_mode == "sqrt":
            self.stretch_f = sqrt_scale
            self.args = ()
        elif stretch_mode == "log":
            self.stretch_f = log_scale
            self.args = (config["log"]["a"],)
        elif stretch_mode == "asinh":
            self.stretch_f = asinh_scale
            self.args = (config["asinh"]["a"],)
        elif stretch_mode == "hist_eq":
            self.stretch_f = hist_eq_scale
            self.args = (config["hist_eq"]["clipLimit"], config["hist_eq"]["tileGridSize"])
        else:
            raise ValueError(f"Stretching function {stretch_mode} is not implemented")
        
    def compute_max_vals(self, statistics, max, clamp):

        if clamp:
            if self.sigma_clamp:

                if self.quantile_clamp:
                    raise ValueError(f"Invalid combination of parameters 'sigma_clamp' ({self.sigma_clamp}) and 'quantile_clamp' ({self.quantile_clamp}). These parameters cannot both be True")

                
                # Compute the x-sigma values 
                means = torch.tensor(statistics['Means'].values, dtype=torch.float32)
                variances = torch.tensor(statistics['Variances'].values, dtype=torch.float32)
                stds = torch.sqrt(variances)

                max_vals = means + self.sigma_clamp*stds 
           

            elif self.quantile_clamp:
                
                # Retrieve the specified qunatiles 
                quantiles =  statistics[f'quantile_{self.quantile_clamp}'].values
                max_vals = torch.tensor(np.array([quantiles]), dtype=torch.float32).reshape(quantiles.shape[-1])
                # max_vals = torch.as_tensor(1)
                # self.plot_histogram(max_vals, 0.002)

            else:

                # Just use the lr_max/ hr_max value from the config file 
                max_vals = torch.as_tensor(max).expand(len(statistics["Means"]))
        else:
            # Check if the combination of parameters is valid
            if self.sigma_clamp and self.input_clamp == self.target_clamp:
                raise ValueError(f"Invalid combination of parameters 'clamp' ({clamp}) and 'sigma_clamp' ({self.sigma_clamp}). 'sigma_clamp' can only be True if 'clamp' is also True)")
            
            elif self.quantile_clamp and self.input_clamp == self.target_clamp: 
                 raise ValueError(f"Invalid combination of parameters 'clamp' ({clamp}) and 'quantile_clamp' ({self.quantile_clamp}). 'quantile_clamp' can only be True if 'clamp' is also True)")
            else:
                # Use the actual max within each image as "clamping" value
                max_vals = torch.tensor(statistics['Maxes'].values, dtype=torch.float32)
                
                
        # Make sure that the max values of empty images dont lead to instability
        max_vals = max_vals.clone()
        max_vals[torch.where(max_vals == 0)] = 1

        return max_vals
    
    def compute_norm_vals(self, statistics, norm, max_vals, max):
        
        if norm == 'same':
            # Target norm is equal to the clamping value 
            norm_val = max_vals
        elif norm == 'equiv':
            # Use the norm that corresponds to the clamping kind of the input image 
            norm_val = self.compute_max_vals(statistics, max, self.input_clamp)
            
        return norm_val 


    def normalize_image(self, image, max_val, norm_val,  clamp = True):
        
        #TODO: check if there isnt a more efficient way to put this 
        # Adjust the dimensions of the max value 
        max_val = max_val.to(image.device).view(-1, *([1] * (image.dim() - 1)))
        norm_val = norm_val.to(image.device).view(-1, *([1] * (image.dim() - 1)))
       
        if clamp: 
            min_val = torch.as_tensor(0).to(image.device)
            image = torch.clamp(image, min=min_val, max=max_val)
            
        # Normalize the image
        image = image / norm_val
                 
        # Apply the stretching function
        image = self.stretch_f(image, *self.args)
       
        # Clip the final image in order to prevent rounding errors
        image = torch.clamp(image, min=0.0, max=1.0)

        return image

    def denormalize_image(self, image, max_val, norm_val, clamp = True):
        
        max_val = max_val.to(image.device).view(-1, *([1] * (image.dim() - 1)))
        norm_val = norm_val.to(image.device).view(-1, *([1] * (image.dim() - 1)))
        
        # De-normalizes the image
        image = self.stretch_f(image, *self.args, inverse=True)
       
        # Denormalize the max val
        image = image * norm_val
        # test = torch.amax(image, dim = (1,2,3))

        if clamp: 
            # The denormalized image cannot be bigger than the maximum value and cannot be negative
            min_val = torch.as_tensor(0).to(image.device)
            image = torch.clamp(image, min=min_val, max=max_val)
               
        return image

    def normalize_lr_image(self, image, idx):
        max_val = self.lr_max_vals.to(image.device)[idx]
        return self.normalize_image(image, max_val = max_val, norm_val = max_val, clamp = self.input_clamp)
       
    def normalize_hr_image(self, image, idx):
        max_val = self.hr_max_vals.to(image.device)[idx]
        return self.normalize_image(image, max_val = max_val, norm_val = self.target_norms.to(image.device)[idx], clamp = self.target_clamp)

    def denormalize_lr_image(self, image, idx):
        max_val = self.lr_max_vals.to(image.device)[idx]
        return self.denormalize_image(image, max_val = max_val, norm_val = max_val, clamp = self.input_clamp)

    def denormalize_hr_image(self, image, idx):
        max_val = self.hr_max_vals.to(image.device)[idx]
        return self.denormalize_image(image, max_val = max_val, norm_val = self.target_norms.to(image.device)[idx], clamp = self.target_clamp)

    

