import torch
import torch.nn.functional as F
from torch import nn

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    if bias is None:
        return F.leaky_relu(input, negative_slope=negative_slope) * scale
    else:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)