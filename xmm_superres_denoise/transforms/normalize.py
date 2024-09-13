import torch


def _asinh(x: torch.Tensor):
    a = torch.tensor(0.02)

    x = torch.asinh(x / a)
    a = torch.asinh(1.0 / a)

    return x / a


def _asinh_inv(x: torch.Tensor):
    a = torch.tensor(0.02)

    x = x * torch.asinh(1.0 / a)
    x = torch.sinh(x)

    return a * x


# http://ds9.si.edu/doc/ref/how.html
def _log(x: torch.Tensor):
    a = torch.tensor(1000)

    return torch.log(a * x + 1) / torch.log(a)


def _log_inv(x: torch.Tensor):
    a = torch.tensor(1000)

    return (torch.pow(a, x) - 1) / a


class Normalize:
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
        self.lr_max: torch.Tensor = torch.tensor(lr_max)
        self.hr_max: torch.Tensor = torch.tensor(hr_max)

        self.norm = self.denorm = None
        if stretch_mode == "linear":
            self.norm = self.denorm = lambda x: x
        elif stretch_mode == "sqrt":
            self.norm, self.denorm = torch.sqrt, torch.square
        elif stretch_mode == "log":
            self.norm, self.denorm = _log, _log_inv
        elif stretch_mode == "asinh":
            self.norm, self.denorm = _asinh, _asinh_inv
        else:
            raise ValueError(f"Stretching function {stretch_mode} is not implemented")

    def normalize_image(
        self, image: torch.Tensor, max_val: torch.Tensor
    ) -> torch.Tensor:
        if max_val > 0:
            torch.clamp_(image, min=torch.tensor(0.0), max=max_val)
            image = image / max_val
        else:
            max_val = torch.max(image)
            image = image / max_val

        # Apply the stretching function
        image = self.norm(image)

        # Clip the final image in order to prevent rounding errors
        torch.clamp_(image, min=0.0, max=1.0)

        return image

    def denormalize_image(self, image: torch.Tensor, max_val: torch.Tensor):
        # De-normalizes the image
        image = self.denorm(image)

        image = max_val[:, None, None, None] * image
        torch.clamp_min_(image, min=torch.tensor(0.0))
        torch.clamp_max_(image, max=max_val[:, None, None, None])

        return image

    def normalize_lr_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.normalize_image(image, max_val=self.lr_max)

    def normalize_hr_image(self, image: torch.Tensor | None):
        if image is None:
            return None

        return self.normalize_image(image, max_val=self.hr_max)

    def denormalize_lr_image(self, image: torch.Tensor):
        return self.denormalize_image(image, max_val=self.lr_max)

    def denormalize_hr_image(self, image):
        return self.denormalize_image(image, max_val=self.hr_max)
