import torch
import torchvision.transforms.functional as F


class RandomFlip(torch.nn.Module):
    """Horizontally and vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, lr_img, hr_img):
        """
        Args:
            lr_img/hr_img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        # Horizontal flip
        if torch.rand(1) < self.p:
            lr_img = F.hflip(lr_img)
            hr_img = F.hflip(hr_img)

        # Vertical flip
        if torch.rand(1) < self.p:
            lr_img = F.vflip(lr_img)
            hr_img = F.vflip(hr_img)

        return lr_img, hr_img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)
