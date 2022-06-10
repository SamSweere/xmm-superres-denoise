import torch
from torch import nn
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        # Since our images are black and white clone them three times
        rgb_image = torch.cat(3*[img], axis=1)

        return self.vgg19_54(rgb_image)