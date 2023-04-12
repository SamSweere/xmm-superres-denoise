import functools
import math

import torch
from torch import nn

from models.modules import RRDB, make_layer


class GeneratorRRDB_SR(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_filters, num_res_blocks, num_upsample=2
    ):
        # We set the grow count equal to the number of num_filters
        grow_count_filters = num_filters

        super(GeneratorRRDB_SR, self).__init__()
        self.num_upsample = num_upsample

        RRDB_block_f = functools.partial(RRDB, nf=num_filters, gc=grow_count_filters)

        self.conv_first = nn.Conv2d(in_channels, num_filters, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, num_res_blocks)
        self.trunk_conv = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)
        #### upsampling
        # # Upsampling layers
        # self.upsample_layers = []
        # for _ in range(num_upsample):
        #     self.upsample_layers.append(nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True))

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(num_filters, num_filters * 4, 3, 1, 1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        # self.upconv1 = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_filters, out_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Bias the weights to be positive, since we need to clamp in the end, based on the default init: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2
        positive_offset_std = 0.01
        stdv = 1.0 / math.sqrt(self.conv_last.weight.size(1))

        self.conv_last.weight.data.uniform_(-stdv, stdv + positive_offset_std * stdv)
        if self.conv_last.bias is not None:
            self.conv_last.bias.data.uniform_(-stdv, stdv + positive_offset_std * stdv)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        # for upsample_layer in self.upsample_layers:
        #     fea = self.lrelu(upsample_layer(F.interpolate(fea, scale_factor=2, mode='nearest')))

        fea = self.upsampling(fea)

        # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        out = torch.clamp(out, min=0.0, max=1.0)

        return out


class GeneratorRRDB_DN(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_filters, num_res_blocks, num_upsample=2
    ):
        # We set the grow count equal to the number of num_filters
        grow_count_filters = num_filters

        self.in_channels = in_channels

        super(GeneratorRRDB_DN, self).__init__()
        self.num_upsample = num_upsample

        RRDB_block_f = functools.partial(RRDB, nf=num_filters, gc=grow_count_filters)

        self.conv_first = nn.Conv2d(in_channels, num_filters, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, num_res_blocks)
        self.trunk_conv = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)

        # self.HRconv = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_filters, out_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Bias the weights to be positive, since we need to clamp in the end, based on the default init: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2
        positive_offset_std = 0.01
        stdv = 1.0 / math.sqrt(self.conv_last.weight.size(1))

        self.conv_last.weight.data.uniform_(-stdv, stdv + positive_offset_std * stdv)
        if self.conv_last.bias is not None:
            self.conv_last.bias.data.uniform_(-stdv, stdv + positive_offset_std * stdv)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        out = self.conv_last(fea)

        if self.in_channels > 1:
            x = x[:, 0, :, :].reshape(x.shape[0], 1, x.shape[2], x.shape[3])

        out = out + x

        out = torch.clamp(out, min=0.0, max=1.0)

        return out


# class GeneratorRRDB_DN(nn.Module):
#     def __init__(self, in_channels, out_channels, filters=64, num_res_blocks=16):
#         super(GeneratorRRDB, self).__init__()
#         self.in_channels = in_channels
#
#         # First layer
#         self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)
#         # Residual blocks
#         self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
#         # Second conv layer post residual blocks
#         self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
#         # # Upsampling layers
#         # upsample_layers = []
#         # for _ in range(num_upsample):
#         #     upsample_layers += [
#         #         nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, stride=1, padding=1),
#         #         nn.LeakyReLU(),
#         #         nn.PixelShuffle(upscale_factor=2),
#         #     ]
#         # self.upsampling = nn.Sequential(*upsample_layers)
#         # Final output block
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding=1),
#         )
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out = self.res_blocks(out1)
#         out2 = self.conv2(out)
#         out = torch.add(out1, out2)
#         # out = self.upsampling(out)
#         out = self.conv3(out)
#         if self.in_channels > 1:
#             x = x[:, 0, :, :].reshape(x.shape[0], 1, x.shape[2], x.shape[3])
#         out = torch.add(x, out)
#         out = torch.clamp(out, min=0.0, max=1.0)
#         return out

# class GeneratorRRDB(nn.Module):
#     def __init__(self, in_channels, out_channels, filters=64, num_res_blocks=16, num_upsample=2):
#         super(GeneratorRRDB, self).__init__()
#
#         # First layer
#         self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)
#
#         # torch.nn.init.xavier_uniform(self.conv1.weight)
#         # Residual blocks
#         self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
#         # Second conv layer post residual blocks
#         self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
#         # Upsampling layers
#         upsample_layers = []
#         for _ in range(num_upsample):
#             upsample_layers += [
#                 nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(),
#                 nn.PixelShuffle(upscale_factor=2),
#             ]
#         self.upsampling = nn.Sequential(*upsample_layers)
#         # Final output block
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding=1),
#         )
#
#         # Bias the weights to be positive, since we need to clamp in the end, based on the default init: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2
#         positive_offset_std = 0.01
#         for m in self.conv3:
#             if isinstance(m, nn.Conv2d):
#                 stdv = 1. / math.sqrt(m.weight.size(1))
#
#                 m.weight.data.uniform_(-stdv, stdv + positive_offset_std * stdv)
#                 if m.bias is not None:
#                     m.bias.data.uniform_(-stdv, stdv + positive_offset_std * stdv)
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out = self.res_blocks(out1)
#         out2 = self.conv2(out)
#         out = torch.add(out1, out2)
#         out = self.upsampling(out)
#         out = self.conv3(out)
#         out = torch.clamp(out, min=0.0, max=1.0)
#         return out
