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


def _cat(in_tensors: List[torch.Tensor], out, dummy_tensor):
    torch.cat(in_tensors, 1, out=out)


def _apply_layer(in_tensors: torch.Tensor, layer, dummy_tensor) -> torch.Tensor:
    return layer(in_tensors)


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
            out1 = torch.zeros_like(x)
            out2 = torch.zeros_like(x)

            cp(_cat, [x, x1], out1, torch.ones(1, requires_grad=True))
            x2 = cp(_apply_layer, out1, self.conv2, torch.ones(1, requires_grad=True))
            self.lrelu(x2)

            cp(_cat, [out1, x2], out2, torch.ones(1, requires_grad=True))
            del x2
            x3 = cp(_apply_layer, out2, self.conv3, torch.ones(1, requires_grad=True))
            self.lrelu(x3)

            cp(_cat, [out2, x3], out1, torch.ones(1, requires_grad=True))
            del x3
            x4 = cp(_apply_layer, out1, self.conv4, torch.ones(1, requires_grad=True))
            self.lrelu(x4)

            cp(_cat, [out1, x4], out2, torch.ones(1, requires_grad=True))
            del x4
            del out1
            x5 = cp(_apply_layer, out2, self.conv5, torch.ones(1, requires_grad=True))
            del out2
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
