import torch

from transforms.data_scaling_functions import linear_scale, sqrt_scale, log_scale, asinh_scale


class Normalize(object):
    """Normalize and image optionally based on a strectching function. First apply the strectching function then normalize to max = 1.
        The minimum possible value is 0 independent of the image, the image is thus only normalized based on the max value.
        Returns the normalized image and the max value

    Args:
        lr_max (float): Maximum value for lr images
        hr_max (float): Maximum value for hr images
        stretch_mode (string) (optional) : The stretching function options: linear, sqrt, asinh, log

    """

    def __init__(self, lr_max, hr_max, stretch_mode="linear"):
        assert isinstance(stretch_mode, str)

        self.stretch_mode = stretch_mode
        self.lr_max = lr_max
        self.hr_max = hr_max

        self.stretch_f = None
        if stretch_mode == "linear":
            self.stretch_f = linear_scale
        elif stretch_mode == "sqrt":
            self.stretch_f = sqrt_scale
        elif stretch_mode == "log":
            self.stretch_f = log_scale
        elif stretch_mode == "asinh":
            self.stretch_f = asinh_scale
        else:
            raise ValueError(f"Stretching function {stretch_mode} is not implemented")

    def normalize_image(self, image, max_val):
        image = torch.clamp(image, min=0.0, max=max_val)

        # Normalize the image
        image = image / max_val

        # Apply the stretching function
        image = self.stretch_f(image)

        # Clip the final image in order to prevent rounding errors
        image = torch.clamp(image, min=0.0, max=1.0)

        return image

    def denormalize_image(self, image, max_val):
        # De-normalizes the image
        image = self.stretch_f(image, inverse=True)

        # Denormalize the max val
        image = image * max_val

        # The denormalized image cannot be bigger than the maximum value and cannot be negative
        image = torch.clamp(image, min=0.0, max=max_val)

        return image

    def normalize_lr_image(self, image):
        return self.normalize_image(image, max_val=self.lr_max)

    def normalize_hr_image(self, image):
        return self.normalize_image(image, max_val=self.hr_max)

    def denormalize_lr_image(self, image):
        return self.denormalize_image(image, max_val=self.lr_max)

    def denormalize_hr_image(self, image):
        return self.denormalize_image(image, max_val=self.hr_max)

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
    lr_max = 0.0022336
    hr_max = 0.0005584

    norm_linear = Normalize(lr_max, hr_max, "linear")
    norm_sqrt = Normalize(lr_max, hr_max, "sqrt")
    norm_asinh = Normalize(lr_max, hr_max, "asinh")
    norm_log = Normalize(lr_max, hr_max, "log")

    input = torch.linspace(0.0, hr_max, 10)

    input_lin_norm = norm_linear.normalize_hr_image(input)
    input_sqrt_norm = norm_sqrt.normalize_hr_image(input)
    input_asinh_norm = norm_asinh.normalize_hr_image(input)
    input_log_norm = norm_log.normalize_hr_image(input)

    output_lin_corr = norm_linear.denormalize_hr_image(input_lin_norm)
    output_sqrt_corr = norm_sqrt.denormalize_hr_image(input_sqrt_norm)
    output_asinh_corr = norm_asinh.denormalize_hr_image(input_asinh_norm)
    output_log_corr = norm_log.denormalize_hr_image(input_log_norm)

    def torch_round(arr, n_digits):
        return torch.round(arr * (10**n_digits)) / (10**n_digits)

    print("input:", input)
    print("lin norm and denormed input:", output_lin_corr)
    print("sqrt norm and denormed input:", output_sqrt_corr)
    print("asinh norm and denormed input:", output_asinh_corr)
    print("log norm and denormed input:", output_log_corr)

    # if torch.equal(torch_round(input, 5), torch_round(output_lin_corr, 5)):
    #     print("WARNING NOT THE SAME")
    #     print("input:",input)
    #     print("norm and denormed input:",output_lin_corr)
