import torch
from data.config import StretchingFunction


def asinh(x: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(0.02, device=x.device)

    x = torch.asinh(x / a)
    a = torch.asinh(1.0 / a)

    return x / a


def asinh_(x: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(0.02, device=x.device)

    x.div_(a).asinh_()
    a.pow_(-1).asinh_()

    x.div_(a)
    return


def asinh_inv(x: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(0.02, device=x.device)

    x = x * torch.asinh(1.0 / a)
    x = torch.sinh(x)

    return a * x


def asinh_inv_(x: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(0.02, device=x.device)

    x.mul_(torch.asinh(1.0 / a))
    x.sinh_()
    x.mul_(a)

    return x


# http://ds9.si.edu/doc/ref/how.html
def log(x: torch.Tensor):
    a = torch.tensor(1000, device=x.device)

    return torch.log(a * x + 1) / torch.log(a)


def log_(x: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(1000).to(device=x.device)

    x.mul_(a).add_(1).log_()
    a.log_()
    x.div_(a)

    return x


def log_inv(x: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(1000, device=x.device)

    return (torch.pow(a, x) - 1) / a


def log_inv_(x: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(1000).to(device=x.device)

    torch.pow(a, x, out=x)
    x.sub_(1).div_(a)

    return x


norm_map = {
    StretchingFunction.LINEAR: lambda x: x,
    StretchingFunction.SQRT: torch.sqrt,
    StretchingFunction.ASINH: asinh,
    StretchingFunction.LOG: log,
}


norm_map_ = {
    StretchingFunction.LINEAR: lambda x: x,
    StretchingFunction.SQRT: torch.sqrt_,
    StretchingFunction.ASINH: asinh_,
    StretchingFunction.LOG: log_,
}


denorm_map = {
    StretchingFunction.LINEAR: lambda x: x,
    StretchingFunction.SQRT: torch.square,
    StretchingFunction.ASINH: asinh_inv,
    StretchingFunction.LOG: log_inv,
}


denorm_map_ = {
    StretchingFunction.LINEAR: lambda x: x,
    StretchingFunction.SQRT: torch.square_,
    StretchingFunction.ASINH: asinh_inv_,
    StretchingFunction.LOG: log_inv_,
}


def normalize_image(
    stretch_mode: StretchingFunction,
    image: torch.Tensor,
    max_val: float,
    inplace: bool = False,
) -> torch.Tensor:
    clamp_func = torch.clamp_ if inplace else torch.clamp

    if max_val > 0:
        image = clamp_func(image, min=0.0, max=max_val)
    else:
        max_val = torch.max(image)

    image = image.div_(max_val) if inplace else image.div(max_val)

    stretch_func = norm_map_[stretch_mode] if inplace else norm_map[stretch_mode]
    image = stretch_func(image)

    # Clip the final image in order to prevent rounding errors
    image = clamp_func(image, min=0.0, max=1.0)

    return image


def denormalize_image(
    stretch_mode: StretchingFunction,
    image: torch.Tensor,
    max_val: float,
    inplace: bool = False,
) -> torch.Tensor:
    clamp_func = torch.clamp_ if inplace else torch.clamp

    stretch_func = denorm_map_[stretch_mode] if inplace else denorm_map[stretch_mode]
    image = stretch_func(image)

    if max_val > 0:
        image = image.mul_(max_val) if inplace else image.mul(max_val)
    image = clamp_func(image, min=0.0, max=max_val)

    return image
