import functools
import math

import torch
from torch import nn

from xmm_superres_denoise.models.modules import RRDB, make_layer

from xmm_superres_denoise.models.modules.rrdb_blocks import get_conv2d_output_size


class _GeneratorRRDB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_res_blocks: int,
        H_in: int,
        W_in: int,
        memory_efficient: bool = False,
        normalization_layer: bool = False
    ):
        super(_GeneratorRRDB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        # self.H_in = H_in, 
        # self.W_in = W_in,
        self.memory_efficient = memory_efficient
        # self.normalization_layer = normalization_layer
        
        # swapped the order so I can compute the image output dimensions after the first convolution in case the choice of convolutional parameters changes it (Yvonne)
        self.conv_first = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # compute output size of first convolutional layer (not sure if neccessary. If not H/W_first_out = H/W_in) (Yvonne)
        self.H_first_out, self.W_first_out = get_conv2d_output_size((H_in, W_in), self.conv_first)

        rrdb = functools.partial(
            RRDB,
            H_in = self.H_first_out, 
            W_in = self.W_first_out,
            nf=self.num_filters,
            gc=num_filters,
            memory_efficient=self.memory_efficient,
            layer_normalization = normalization_layer
        )
        
        self.rrdb = make_layer(rrdb, self.num_res_blocks)
        self.trunk_conv = nn.Conv2d(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_last = nn.Conv2d(
            in_channels=self.num_filters,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Bias the weights to be positive, since we need to clamp in the end,
        # based on the default init:
        # https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2
        positive_offset_std = 0.01
        stdv = 1.0 / math.sqrt(self.conv_last.weight.size(1))

        self.conv_last.weight.data.uniform_(-stdv, stdv + positive_offset_std * stdv)
        if self.conv_last.bias is not None:
            self.conv_last.bias.data.uniform_(-stdv, stdv + positive_offset_std * stdv)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.rrdb(fea))
        return fea + trunk


class GeneratorRRDB_SR(_GeneratorRRDB):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_res_blocks: int,
        H_in: int, 
        W_in: int,
        num_upsample: int = 2,
        memory_efficient: bool = False,
        normalization_layer: bool = False,
    ):
        super(GeneratorRRDB_SR, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            num_res_blocks=num_res_blocks,
            H_in = H_in, 
            W_in = W_in,
            memory_efficient=memory_efficient,
            normalization_layer=normalization_layer,
        )
        self.num_upsample = num_upsample

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(num_filters, num_filters * 4, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        self.HRconv = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = super().forward(x)

        fea = self.upsampling(fea)
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        out = torch.clamp(out, min=0.0, max=1.0)

        return out


class GeneratorRRDB_DN(_GeneratorRRDB):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_filters,
        num_res_blocks,
        H_in, 
        W_in,
        memory_efficient=False,
        normalization_layer = False,
    ):
        super(GeneratorRRDB_DN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            num_res_blocks=num_res_blocks,
            H_in = H_in, 
            W_in = W_in,
            memory_efficient=memory_efficient,
            normalization_layer=normalization_layer,
        )

    def forward(self, x):
        fea = super().forward(x)
        out = self.conv_last(fea)

        out = out + x
        out = torch.clamp(out, min=0.0, max=1.0)

        return out
