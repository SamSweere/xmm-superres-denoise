# Altered from torchvision.transforms.functional.normalize
# (https://pytorch.org/vision/stable/_modules/torchvision/transforms/functional.html)
import torch
from torch import Tensor


class ZScaleNormalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std

        if mean is None:
            raise ValueError("mean is None")

        if std is None:
            raise ValueError("std is None")

        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Input tensor should be a torch tensor. Got {type(tensor)}."
            )

        if not tensor.is_floating_point():
            raise TypeError(
                "Input tensor should be a float tensor. Got {}.".format(tensor.dtype)
            )

        if tensor.ndim != 3:
            raise ValueError(
                "Expected tensor to be a tensor image of size (..., C, H, W) (three-dimensional). Got "
                "tensor.size() = "
                "{}.".format(tensor.size())
            )

        if not self.inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(
                "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                    dtype
                )
            )

        tensor.sub_(mean).div_(std)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )
