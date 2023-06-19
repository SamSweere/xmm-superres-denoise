# Based on: https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py

from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def _apply_layer(in_tensors: List[torch.Tensor], layer) -> torch.Tensor:
    concat = cp(torch.cat, tensors=in_tensors, dim=1, use_reentrant=False)
    return layer(concat)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True, memory_efficient: bool = False):
        super(ResidualDenseBlock_5C, self).__init__()
        self.mem_efficient = memory_efficient
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x: torch.Tensor):
        x1 = self.lrelu(self.conv1(x))
        if self.mem_efficient:
            x2 = self.lrelu(cp(_apply_layer, [x, x1], self.conv2, use_reentrant=False))
            x3 = self.lrelu(
                cp(_apply_layer, [x, x1, x2], self.conv3, use_reentrant=False)
            )
            x4 = self.lrelu(
                cp(_apply_layer, [x, x1, x2, x3], self.conv4, use_reentrant=False)
            )
            x5 = cp(_apply_layer, [x, x1, x2, x3, x4], self.conv5, use_reentrant=False)
        else:
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32, memory_efficient: bool = False):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, memory_efficient=memory_efficient)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc, memory_efficient=memory_efficient)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc, memory_efficient=memory_efficient)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
