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


def get_conv2d_output_size(input_size, conv_layer):
    """
    Calculate the output size of a 2D convolutional layer. (added by Yvonne)

    Parameters:
        input_size (tuple): The size of the input image as a tuple (H_in, W_in).
        conv_layer (nn.Conv2d): The convolutional layer for which to calculate the output size.

    Returns:
        tuple: The output size of the convolutional layer as a tuple (H_out, W_out).
    """
    
    # Extract convolutional layer parameters
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding

    # Calculate output size
    H_in, W_in = input_size
    H_out = int(((H_in - kernel_size[0] + 2 * padding[0]) / stride[0]) + 1)
    W_out = int(((W_in - kernel_size[1] + 2 * padding[1]) / stride[1]) + 1)

    return (H_out, W_out)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, H_in, W_in, nf=64, gc=32, bias=True, memory_efficient: bool = False, layer_normalization: bool = False):
        super(ResidualDenseBlock_5C, self).__init__()
        self.mem_efficient = memory_efficient
        self.layer_normalization = layer_normalization
        
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # layer normalization after every convolutional layer (added by Yvonne)
        if self.layer_normalization: 
            # compute the output image size of each convolutional layer
            # note that this is only needed if the convolution changes the image size which is not the case for the current choice of parameters
            H1_out, W1_out = get_conv2d_output_size((H_in, W_in), self.conv1)
            H2_out, W2_out = get_conv2d_output_size((H1_out, W1_out), self.conv2)
            H3_out, W3_out = get_conv2d_output_size((H2_out, W2_out), self.conv3)
            H4_out, W4_out = get_conv2d_output_size((H3_out, W3_out), self.conv4)
            
            # define layer normalization layers 
            self.ln1 = nn.LayerNorm([gc, H_in, W_in])
            self.ln2 = nn.LayerNorm([gc, H1_out, W1_out])
            self.ln3 = nn.LayerNorm([gc, H2_out, W2_out])
            self.ln4 = nn.LayerNorm([gc, H3_out, W3_out])
            self.ln5 = nn.LayerNorm([nf, H4_out, W4_out])
            
        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    #TODO: implement layer normalization for memory efficient 
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
            if self.layer_normalization:
                x1 = self.ln1(x1)
                x2 = self.lrelu(self.ln2(self.conv2(torch.cat((x, x1), 1))))
                x3 = self.lrelu(self.ln3(self.conv3(torch.cat((x, x1, x2), 1))))
                x4 = self.lrelu(self.ln4(self.conv4(torch.cat((x, x1, x2, x3), 1))))
                x5 = self.ln5(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))

            else:
                x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
                x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
                x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
                x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, H_in, W_in, nf, gc=32, memory_efficient: bool = False, layer_normalization: bool = False):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(H_in, W_in, nf, gc, memory_efficient=memory_efficient, layer_normalization=layer_normalization)
        self.RDB2 = ResidualDenseBlock_5C(H_in, W_in, nf, gc, memory_efficient=memory_efficient, layer_normalization=layer_normalization)
        self.RDB3 = ResidualDenseBlock_5C(H_in, W_in, nf, gc, memory_efficient=memory_efficient, layer_normalization=layer_normalization)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
