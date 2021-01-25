from torch import nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3_rand(in_planes, out_planes, stride=1):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    nn.init.kaiming_normal_(conv.weight)
    conv.weight.requires_grad = False
    return conv

def conv3x3_rand_binary(in_planes, out_planes, stride=1):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    conv.weight.requires_grad = False
    conv.weight[conv.weight > 0.05] = 1
    conv.weight[conv.weight < -0.05] = -1
    mask = (conv.weight == 1) | (conv.weight == -1)
    conv.weight[~mask] = 0
    return conv

class SEBLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        print("SE_B_Layer")

    def forward(self, x_in, x_out):
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x_out * y.expand_as(x_out)

class SEBLayer_v1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBLayer_v1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())

        print("SE_B_Layer_v1")

    def forward(self, x_in, x_out):
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y_a = self.se_a(x_in)
        return x_out * y.expand_as(x_out) * y_a.expand_as(x_out)

class ALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        print("A_Layer")

    def forward(self, x_in, x_out):
        y = self.se_a(x_in)
        return x_out * y.expand_as(x_out)


class A_relu_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(A_relu_Layer, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.act = nn.ReLU(inplace=True)
        print("A_relu_Layer")

    def forward(self, x_in, x_out):
        A = self.se_a(x_in)
        y = x_out * A.expand_as(x_out)
        y = self.act(y)
        return y

class A_tanh_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(A_tanh_Layer, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.act = nn.Tanh()
        print("A_tanh_Layer")

    def forward(self, x_in, x_out):
        A = self.se_a(x_in)
        y = x_out * A.expand_as(x_out)
        y = self.act(y)
        return y



class ALayer_v1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_v1, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())

        print("A_Layer_v1")

    def forward(self, x_in, x_out):
        # A generated from x_in
        A = self.se_a(x_in)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        fold = nn.Fold(x_in.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x_out)
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape)
        out = fold(out_unfold)

        return out

class ALayer_v1_1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_v1_1, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        # self.weights = torch.
        print("A_Layer_v1.1")

    def forward(self, x, weights):
        # A generated from x_in
        A = self.se_a(x)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        fold = nn.Fold(x.size()[2:], **fold_params)

        # input_ones = torch.ones(x.shape, dtype=x.dtype, device='cuda')
        # divisor = fold(unfold(input_ones))

        inp = torch.randn(128, 64, 8, 8)
        w = torch.randn(64, 64, 3, 3)

        # apply A to x_out
        out_unfold = unfold(x)
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape)
        out = fold(out_unfold)

        return out


class ALayer_v2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_v2, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        print("A_Layer_v2")

    def forward(self, x_in, x_out):
        # A generated from x_in
        A = self.se_a(x_in)
        # B
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        fold = nn.Fold(x_in.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x_out)
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape)
        out = fold(out_unfold)

        return out

class ALayer_DR1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_DR1, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        print("A_Layer_DR1")

    def forward(self, x_in, x_out):
        y = self.se_a(x_in)
        return x_out * y

class ALayer_DR1_v1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_DR1_v1, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        print("A_Layer_DR1")

    def forward(self, x, weight):

        A = self.se_a(x)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        # fold = nn.Fold(x.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x)
        out_unfold = out_unfold * A.flatten(2, -1).repeat(1, 9, 1)  # hardcoded 3x3
        weight = weight.flatten(1, -1)
        conv_result = out_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        out = conv_result.view(x.shape)

        # inp = torch.randn(128, 32, 8, 8)
        # w = torch.randn(72, 32, 3, 3)
        # inp_unf = unfold(inp)
        # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        return out

class ALayer_ADR1(nn.Module):
    def __init__(self, channel, stride,reduction=16):
        super(ALayer_ADR1, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, channel//reduction, kernel_size=3, padding=1, stride=stride, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel//reduction, 9, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        self.stride = stride
        self.channels = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        print("A_Layer_ADR1, reduction=",reduction)

    def forward(self, x, weight):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)


        A = self.se_a(x)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=self.stride)
        unfold = nn.Unfold(**fold_params)
        # fold = nn.Fold(x.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x)
        # HxW to x
        out_unfold = out_unfold * A.flatten(2, -1).repeat(1,self.channels,1) * y.repeat(1,1,9,1).flatten(1,2).expand_as(out_unfold)
        weight = weight.flatten(1, -1)
        conv_result = out_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        out = conv_result.view(b, c, h//self.stride, w//self.stride)

        # inp = torch.randn(128, 32, 8, 8)
        # w = torch.randn(72, 32, 3, 3)
        # inp_unf = unfold(inp)
        # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        return out


class ALayer_DR1_v3(nn.Module):
    def __init__(self, channel, stride,reduction=16):
        super(ALayer_DR1_v3, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, channel//reduction, kernel_size=3, padding=1, stride=stride, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel//reduction, 9*channel, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        self.stride = stride
        self.channels = channel
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        print("A_Layer_DR1_v3, reduction=", reduction)

    def forward(self, x, weight):
        b, c, h, w = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)


        A = self.se_a(x)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=self.stride)
        unfold = nn.Unfold(**fold_params)
        # fold = nn.Fold(x.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x)
        # HxW to x
        out_unfold = out_unfold * A.flatten(2, -1)
        weight = weight.flatten(1, -1)
        conv_result = out_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        out = conv_result.view(b, c, h//self.stride, w//self.stride)

        # inp = torch.randn(128, 32, 8, 8)
        # w = torch.randn(72, 32, 3, 3)
        # inp_unf = unfold(inp)
        # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        return out

class ALayer_A_v2_2(nn.Module):
    def __init__(self, channel, stride, reduction=16):
        super(ALayer_A_v2_2, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, channel//reduction, kernel_size=3, padding=1, stride=stride, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel//reduction, 9, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        self.channels = channel
        self.stride = stride
        print("A_Layer_A_v2.2, reduction=",reduction)

    def forward(self, x, weight):
        b, c, h, w = x.size()
        A = self.se_a(x)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=self.stride)
        unfold = nn.Unfold(**fold_params)
        # fold = nn.Fold(x.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x)
        # HxW to x
        out_unfold = out_unfold * A.flatten(2, -1).repeat(1,self.channels,1)
        weight = weight.flatten(1, -1)
        conv_result = out_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        out = conv_result.view([b,c,h//self.stride, w//self.stride])

        # inp = torch.randn(128, 32, 8, 8)
        # w = torch.randn(72, 32, 3, 3)
        # inp_unf = unfold(inp)
        # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        return out


class ALayer_DR1_v1_light(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_DR1_v1_light, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        kernel_size = 3
        self.A_w = Parameter(torch.Tensor(
                1, channel, kernel_size, kernel_size))
        torch.nn.init.constant_(self.A_w, 1)


        print("A_Layer_DR1_light")

    def forward(self, x, weight):
        # 3x3xCin to w
        A_w = self.A_w.expand_as(weight)
        weight = weight * A_w

        A = self.se_a(x)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        # fold = nn.Fold(x.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x)
        # HxW to x
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape)
        weight = weight.flatten(1, -1)
        conv_result = out_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        out = conv_result.view(x.shape)

        # inp = torch.randn(128, 32, 8, 8)
        # w = torch.randn(72, 32, 3, 3)
        # inp_unf = unfold(inp)
        # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        return out


class ALayer_DR1_v1_light_v1(nn.Module):
    def __init__(self, channel, stride, reduction=16):
        super(ALayer_DR1_v1_light_v1, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, channel // reduction, kernel_size=3, padding=1, stride=stride, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel // reduction, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        self.stride = stride
        self.kernel_size = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, self.kernel_size * self.kernel_size * channel, bias=False),
            nn.Sigmoid()
        )

        print("A_Layer_DR1_light_v1")

    def forward(self, x, weight):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c * self.kernel_size * self.kernel_size,1)

        A = self.se_a(x)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=self.stride)
        unfold = nn.Unfold(**fold_params)
        # fold = nn.Fold(x.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x)
        # HxW to x
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape) * y.expand(out_unfold.shape)
        weight = weight.flatten(1, -1)
        conv_result = out_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        out = conv_result.view([b,c,h//self.stride, w//self.stride])

        # inp = torch.randn(128, 32, 8, 8)
        # w = torch.randn(72, 32, 3, 3)
        # inp_unf = unfold(inp)
        # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        return out

class ALayer_DR1_wh_light_v1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_DR1_wh_light_v1, self).__init__()
        # self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
        #                           nn.ReLU(inplace=True),
        #                           nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
        #                           nn.Sigmoid())
        input_w = 8
        self.conv2_wh = nn.Sequential(
            nn.Linear(input_w * channel, input_w * channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_w * channel // 16, input_w, bias=False),
            nn.Sigmoid()
        )
        self.kernel_size = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, self.kernel_size * self.kernel_size * channel, bias=False),
            nn.Sigmoid()
        )

        print("A_Layer_DR1_wh_light_v1")

    def forward(self, x, weight):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c * self.kernel_size * self.kernel_size,1)

        # A = self.se_a(x)

        x_w = torch.mean(x, dim=2).unsqueeze(2)
        x_h = torch.mean(x, dim=3).unsqueeze(3)

        # compress channels
        x_w = x_w.view(x_w.size(0), -1)
        x_h = x_h.view(x_h.size(0), -1)

        A_w = self.conv2_wh(x_w)
        A_h = self.conv2_wh(x_h)

        A = A_w.unsqueeze(1).unsqueeze(2) * A_h.unsqueeze(1).unsqueeze(3)

        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        # fold = nn.Fold(x.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x)
        # HxW to x
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape) * y.expand(out_unfold.shape)
        weight = weight.flatten(1, -1)
        conv_result = out_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        out = conv_result.view(x.shape)

        # inp = torch.randn(128, 32, 8, 8)
        # w = torch.randn(72, 32, 3, 3)
        # inp_unf = unfold(inp)
        # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        return out


class ALayer_DR1_v1_light_vx(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_DR1_v1_light_vx, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())

        print("A_Layer_DR1_light")

    def forward(self, x, weight, A_w):
        # 3x3xCin to w
        A_w = A_w.expand_as(weight)
        weight = weight * A_w

        A = self.se_a(x)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        # fold = nn.Fold(x.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x)
        # HxW to x
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape)
        weight = weight.flatten(1, -1)
        conv_result = out_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        out = conv_result.view(x.shape)

        # inp = torch.randn(128, 32, 8, 8)
        # w = torch.randn(72, 32, 3, 3)
        # inp_unf = unfold(inp)
        # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        return out


class ALayer_wh(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_wh, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        input_w = 8

        self.conv2_wh = nn.Sequential(
            nn.Linear(input_w * channel, input_w * channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_w * channel // 16, input_w, bias=False),
            nn.Sigmoid()
        )
        print("A_Layer_wh-share fc")

    def forward(self, x_in, x_out):
        x_w = torch.mean(x_in, dim=2).unsqueeze(2)
        x_h = torch.mean(x_in, dim=3).unsqueeze(3)

        # compress channels
        x_w = x_w.view(x_w.size(0), -1)
        x_h = x_h.view(x_h.size(0), -1)

        A_w = self.conv2_wh(x_w)
        A_h = self.conv2_wh(x_h)

        A = A_w.unsqueeze(1).unsqueeze(2) * A_h.unsqueeze(1).unsqueeze(3)

        return x_out * A.expand_as(x_out)


class GCN(nn.Module):
    def __init__(self, c, out_c, k=3):  # out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k, 1), padding=((k - 1) / 2, 0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k), padding=(0, (k - 1) / 2))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k), padding=((k - 1) / 2, 0))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k, 1), padding=(0, (k - 1) / 2))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x

class AB_conv3x3(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_conv3x3, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels // groups, kernel_size, kernel_size))

        self.B = Parameter(torch.Tensor(
                out_channels, 1, 1, 1))
        # initialize to 1
        torch.nn.init.constant_(self.A, 1)
        torch.nn.init.constant_(self.B, 1)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        weight = self.conv.weight * AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out

class rand_conv3x3(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(rand_conv3x3, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3(c_in, c_out)
        self.randmask = torch.zeros_like(self.conv.weight, device='cuda', requires_grad=False)
        nn.init.kaiming_normal_(self.randmask)

        print("rand_conv")



    def forward(self, x):
        AB = self.randmask
        print("AB=",AB[1,0,0,0].item())
        weight = self.conv.weight * AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out

class AB_as_conv3x3(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_as_conv3x3, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels // groups, kernel_size, kernel_size))

        self.B = Parameter(torch.Tensor(
                out_channels, 1, 1, 1))
        # initialize to 1
        torch.nn.init.kaiming_normal_(self.A)
        torch.nn.init.kaiming_normal_(self.B)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        weight = AB # weight directly = AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        return out

class AB_split_kernel_as_conv3x3(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_split_kernel_as_conv3x3, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                out_channels, in_channels, kernel_size, 1))

        self.B = Parameter(torch.Tensor(
                out_channels, in_channels, 1, kernel_size))
        # initialize to 1
        torch.nn.init.kaiming_normal_(self.A)
        torch.nn.init.kaiming_normal_(self.B)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        weight = AB # weight directly = AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        return out

class AB_split_channel_as_conv3x3(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_split_channel_as_conv3x3, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels, kernel_size, kernel_size))

        self.B = Parameter(torch.Tensor(
                out_channels, 1, kernel_size, kernel_size))
        # initialize to 1
        torch.nn.init.kaiming_normal_(self.A)
        torch.nn.init.kaiming_normal_(self.B)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        weight = AB # weight directly = AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        return out

class AB_split_channel_kernel_as_conv3x3(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_split_channel_kernel_as_conv3x3, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels, 1, kernel_size))

        self.B = Parameter(torch.Tensor(
                out_channels, 1, kernel_size, 1))
        # initialize to 1
        torch.nn.init.kaiming_normal_(self.A)
        torch.nn.init.kaiming_normal_(self.B)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        weight = AB # weight directly = AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        return out

class AB_as_conv3x3_res(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_as_conv3x3_res, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels // groups, kernel_size, kernel_size))

        self.B = Parameter(torch.Tensor(
                out_channels, 1, 1, 1))
        # initialize to 1
        torch.nn.init.kaiming_normal_(self.A)
        torch.nn.init.kaiming_normal_(self.B)


    def forward(self, x):
        residual = x

        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        weight = AB # weight directly = AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        out += residual
        return out


class AB_conv3x3_rand(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_conv3x3_rand, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3_rand(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels // groups, kernel_size, kernel_size))

        self.B = Parameter(torch.Tensor(
                out_channels, 1, 1, 1))
        # initialize to 1
        torch.nn.init.constant_(self.A, 1)
        torch.nn.init.constant_(self.B, 1)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        weight = self.conv.weight * AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out

class AB_split_kernel_conv3x3_rand(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_split_kernel_conv3x3_rand, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3_rand(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                out_channels, in_channels, 1, kernel_size))

        self.B = Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, 1))

        torch.nn.init.kaiming_normal_(self.A)
        torch.nn.init.kaiming_normal_(self.B)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        weight = self.conv.weight * AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out

class AB_split_channel_conv3x3_rand(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_split_channel_conv3x3_rand, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3_rand(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels, kernel_size, kernel_size))

        self.B = Parameter(torch.Tensor(
            out_channels, 1, kernel_size, kernel_size))

        torch.nn.init.kaiming_normal_(self.A)
        torch.nn.init.kaiming_normal_(self.B)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        weight = self.conv.weight * AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out

class AB_split_channel_kernel_conv3x3_rand(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_split_channel_kernel_conv3x3_rand, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3_rand(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels, 1, kernel_size))

        self.B = Parameter(torch.Tensor(
            out_channels, 1, kernel_size, 1))

        torch.nn.init.kaiming_normal_(self.A)
        torch.nn.init.kaiming_normal_(self.B)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        weight = self.conv.weight * AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out


class AB_conv3x3_rand_res(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_conv3x3_rand_res, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3_rand(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels // groups, kernel_size, kernel_size))

        self.B = Parameter(torch.Tensor(
                out_channels, 1, 1, 1))
        # initialize to 1
        torch.nn.init.constant_(self.A, 1)
        torch.nn.init.constant_(self.B, 1)


    def forward(self, x):
        residual = x

        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        weight = self.conv.weight * AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)
        out += residual

        return out


class AB_conv3x3_rand_binary(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(AB_conv3x3_rand_binary, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3_rand_binary(c_in, c_out)
        self.A = Parameter(torch.Tensor(
                1, in_channels // groups, kernel_size, kernel_size))

        self.B = Parameter(torch.Tensor(
                out_channels, 1, 1, 1))
        # initialize to 1
        torch.nn.init.constant_(self.A, 1)
        torch.nn.init.constant_(self.B, 1)


    def forward(self, x):
        A = self.A.expand_as(self.conv.weight)
        B = self.B.expand_as(A)
        AB = A * B
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        weight = self.conv.weight * AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out

class dynam_AB_conv3x3_rand(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel_size=3, groups=1):
        super(dynam_AB_conv3x3_rand, self).__init__()
        in_channels = c_in
        out_channels = c_out
        self.conv = conv3x3_rand(c_in, c_out)


    def forward(self, x, a, b):
        A = a.expand_as(self.conv.weight)
        B = b.expand_as(A)
        AB = A * B
        # print("B = ", self.B[1,0,0,0])
        # print("weight=", self.conv.weight[1,1,1])
        weight = self.conv.weight * AB
        out = F.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out