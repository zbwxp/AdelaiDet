import torch
from torch import nn


class DR1Conv2d(nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support dynamic rank1 weights
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        bias = kwargs.pop("bias", False)
        assert not bias, "Bias is not supported in DR1Conv."
        stand_alone = kwargs.pop("stand_alone", False)
        assert stand_alone == "DR1" or stand_alone == "DR1_stand_alone", f"DR1 supports 'DR1' and 'DR1_stand_alone', '{stand_alone}' is not supported"
        stand_alone = True if stand_alone == "DR1_stand_alone" else False

        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation
        self.stand_alone = stand_alone

        if self.stand_alone:
            self.in_channel = args[0]
            self.out_channel = args[1]
            kernel_size = args[2]
            self.conv2_alpha_gamma = nn.Conv2d(
                self.in_channel,
                self.in_channel+self.out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
                dilation=1,
            )

            nn.init.constant_(self.conv2_alpha_gamma.bias, 0)
            nn.init.normal_(self.conv2_alpha_gamma.weight, std=0.01)

    def forward(self, x, alpha=None, gamma=None):
        if self.stand_alone:
            assert not alpha, "stand alone DR1conv should not have alpha."
            alpha_gamma = self.conv2_alpha_gamma(x)
            alpha, gamma = torch.split(alpha_gamma, [self.in_channel, self.out_channel], dim=1)

        x = super().forward(x * alpha)
        if gamma is not None:
            x = x * gamma
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DR1CoordConv2d(DR1Conv2d):

    def forward(self, x, coords, alpha, gamma=None):
        x = torch.cat((x, coords), dim=1)
        return super().forward(x, alpha, gamma)