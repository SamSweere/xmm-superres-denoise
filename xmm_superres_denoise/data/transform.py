import torch
from torch.nn.functional import interpolate


def upsample_image(
    x: torch.Tensor,
    scale_factor: int,
    inplace: bool = False,
) -> torch.Tensor:
    single_image = len(x.shape) < 4

    if single_image:
        # Put the input into a batch
        x = x.unsqueeze_(dim=0) if inplace else x.unsqueeze(dim=0)

    x = interpolate(x, scale_factor=scale_factor, mode="nearest")

    # Fix the upsample brightness by dividing by the scale factor squared
    x = x.div_(scale_factor**2) if inplace else x.div(scale_factor**2)

    if single_image:
        # Pull the upsampled input out of the batch
        x = x.squeeze_(dim=0) if inplace else x.squeeze(dim=0)

    return x
