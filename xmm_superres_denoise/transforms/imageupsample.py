import torch
from torch import nn


class ImageUpsample:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")

    def __call__(self, x):
        single_image = len(x.shape) < 4

        if single_image:
            # Put the input into a batch
            x = torch.unsqueeze(x, axis=0)

        x = self.upsample(x)

        # Fix the upsample brightness by dividing by the scale factor squared
        x = x / (self.scale_factor**2)

        if single_image:
            # Pull the upsampled input out of the batch
            x = x[0]

        return x
