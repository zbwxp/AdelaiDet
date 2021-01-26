import torch
from torch import nn

from detectron2.layers import Conv2d
from adet.modeling.backbone.se_module import *
import fvcore.nn.weight_init as weight_init

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class DR1conv(nn.Module):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.

    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py


    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            dilation=1,
            bias=False,
            padding=None
    ):
        super(DR1conv, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation,
            bias=False
        )
        # weight_init.c2_msra_fill(self.conv)

        self.se_a = ALayer_A_v2_2(in_channels, stride, reduction=16)
        # for m in self.se_a.se_a:
        #     if isinstance(m,nn.Conv2d):
        #         weight_init.c2_msra_fill(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()




    def forward(self, x):
        out = self.se_a(x, self.conv.weight)
        return out


class ADR1conv(nn.Module):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.

    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py


    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            dilation=1,
            bias=False,
            padding=None
    ):
        super(ADR1conv, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation,
            bias=False
        )
        # weight_init.c2_msra_fill(self.conv)

        self.se_a = ALayer_ADR1(in_channels, stride, reduction=16)
        # for m in self.se_a.se_a:
        #     if isinstance(m, nn.Conv2d):
        #         weight_init.c2_msra_fill(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        out = self.se_a(x, self.conv.weight)
        return out

class DR1_v3conv(nn.Module):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.

    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py


    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            dilation=1,
            bias=False,
            padding=None
    ):
        super(DR1_v3conv, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation,
            bias=False
        )
        # weight_init.c2_msra_fill(self.conv)

        self.se_a = ALayer_DR1_v3(in_channels, stride, reduction=16)
        # for m in self.se_a.se_a:
        #     if isinstance(m, nn.Conv2d):
        #         weight_init.c2_msra_fill(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        self.se_b = SEBLayer(in_channels, reduction=16)


    def forward(self, x):
        out = self.se_a(x, self.conv.weight)
        out = self.se_b(x, out)
        return out

class SEconv(nn.Module):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.

    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py


    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            dilation=1,
            bias=False,
            padding=None
    ):
        super(SEconv, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation,
            bias=False
        )
        # weight_init.c2_msra_fill(self.conv)

        # self.se_a = ALayer_DR1_v3(in_channels, stride, reduction=16)
        # for m in self.se_a.se_a:
        #     if isinstance(m, nn.Conv2d):
        #         weight_init.c2_msra_fill(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        self.se_b = SEBLayer(in_channels, reduction=16)


    def forward(self, x):
        # out = self.se_a(x, self.conv.weight)
        out = self.se_b(x, x)
        return out