# Based on: https://pytorch.org/vision/0.10/_modules/torchvision/transforms/transforms.html#ToTensor
import torch
import torchvision.transforms.functional as F


class ToTensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/master/references/segmentation
    """

    def __call__(self, pic):
        """
        Args:
            pic (single or list numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """

        # print('To tens pic:', pic)

        if type(pic) == list:
            ret = []
            for img in pic:
                ret.append(torch.from_numpy(img).float())

            return ret
        else:
            return torch.from_numpy(pic).float()

    def __repr__(self):
        return self.__class__.__name__ + "()"
