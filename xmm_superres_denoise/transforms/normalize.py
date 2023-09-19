import torch

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

    def __init__(self, lr_max, hr_max, config, stretch_mode="linear", clamp = True):
        assert isinstance(stretch_mode, str)

        # I am now passing all of the configs for the normalization paraemters, so we don't need to pass the individual arguments anymore, but I don't want to mess with this for now 
        self.stretch_mode = stretch_mode
        self.lr_max = lr_max
        self.hr_max = hr_max
        self.clamp = clamp

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

    def normalize_image(self, image, max_val, clamp = True):

        if clamp: 
            image = torch.clamp(image, min=0.0, max=max_val)

        # Normalize the image
        image = image / max_val

        # Apply the stretching function
        image = self.stretch_f(image, *self.args)

        # Clip the final image in order to prevent rounding errors
        image = torch.clamp(image, min=0.0, max=1.0)

        return image

    def denormalize_image(self, image, max_val, clamp = True):
        # De-normalizes the image
        image = self.stretch_f(image, *self.args, inverse=True)

        # Denormalize the max val
        if clamp: 
            image = image * max_val

        # The denormalized image cannot be bigger than the maximum value and cannot be negative
        image = torch.clamp(image, min=0.0, max=max_val)

        return image

    def normalize_lr_image(self, image):
        return self.normalize_image(image, max_val=self.lr_max, clamp = self.clamp)

    def normalize_hr_image(self, image):
        return self.normalize_image(image, max_val=self.hr_max, clamp = self.clamp)

    def denormalize_lr_image(self, image):
        return self.denormalize_image(image, max_val=self.lr_max, clamp = self.clamp)

    def denormalize_hr_image(self, image):
        return self.denormalize_image(image, max_val=self.hr_max, clamp = self.clamp)

    def __call__(self, image):
        # Returns the transformed image if one image, a list of transformed images if a list of images
        if type(image) == list:
            assert len(image) == 2
            lr_img = image[0]
            hr_img = image[1]

            return [self.normalize_lr_image(lr_img), self.normalize_hr_image(hr_img)]
        else:
            # return self.normalize_image(image)
            raise NotImplementedError(
                "Normalize call not implemented for single images, call normalize directly"
            )


if __name__ == "__main__":

    from xmm_superres_denoise.utils.filehandling import read_yaml

    dataset_config: dict = read_yaml("/home/xmmsas/mywork/xmm-superres-denoise/res/baseline_config.yaml")["dataset"]
  
    lr_max = 0.0022336
    hr_max = 0.0005584

    norm_linear = Normalize(lr_max, hr_max, dataset_config, "linear")
    norm_sqrt = Normalize(lr_max, hr_max, dataset_config, "sqrt")
    norm_asinh = Normalize(lr_max, hr_max, dataset_config, "asinh")
    norm_log = Normalize(lr_max, hr_max, dataset_config, "log")
    norm_hist_eq = Normalize(lr_max, hr_max, dataset_config, "hist_eq")

    # input = torch.linspace(0.0, hr_max, 10)
    input = torch.rand(1, 1, 10, 10)
    

    input_lin_norm = norm_linear.normalize_hr_image(input)
    input_sqrt_norm = norm_sqrt.normalize_hr_image(input)
    input_asinh_norm = norm_asinh.normalize_hr_image(input)
    input_log_norm = norm_log.normalize_hr_image(input)
    input_hist_eq_norm = norm_hist_eq.normalize_hr_image(input)

    output_lin_corr = norm_linear.denormalize_hr_image(input_lin_norm)
    output_sqrt_corr = norm_sqrt.denormalize_hr_image(input_sqrt_norm)
    output_asinh_corr = norm_asinh.denormalize_hr_image(input_asinh_norm)
    output_log_corr = norm_log.denormalize_hr_image(input_log_norm)
    output_hist_eq_corr = norm_hist_eq.denormalize_hr_image(input_hist_eq_norm)

    def torch_round(arr, n_digits):
        return torch.round(arr * (10**n_digits)) / (10**n_digits)

    print("input:", input)
    print("lin norm and denormed input:", output_lin_corr)
    print("sqrt norm and denormed input:", output_sqrt_corr)
    print("asinh norm and denormed input:", output_asinh_corr)
    print("log norm and denormed input:", output_log_corr)
    print("hist_eq norm and denormed input:", output_hist_eq_corr)

    # if torch.equal(torch_round(input, 5), torch_round(output_lin_corr, 5)):
    #     print("WARNING NOT THE SAME")
    #     print("input:",input)
    #     print("norm and denormed input:",output_lin_corr)
