"""Checkpoint-loadable parallel classifier: immutable legacy logits + trainable |."""
from copy import deepcopy

import torch
from torch import nn


class AppendOnlyBarHead(nn.Module):
    def __init__(self, legacy):
        super().__init__()
        self.legacy=deepcopy(legacy)
        self.bar=deepcopy(legacy)
        previous=self.bar[-1]
        assert isinstance(previous,nn.Conv2d) and previous.out_channels==67
        final=nn.Conv2d(previous.in_channels,1,previous.kernel_size,previous.stride,previous.padding,
                        dilation=previous.dilation,groups=previous.groups,bias=previous.bias is not None)
        with torch.no_grad():
            final.weight.copy_(previous.weight[37:38])
            if previous.bias is not None:final.bias.copy_(previous.bias[37:38])
        self.bar[-1]=final
        for p in self.legacy.parameters():p.requires_grad_(False)

    def forward(self,x):
        return torch.cat((self.legacy(x),self.bar(x)),dim=1)

    def train(self,mode=True):
        super().train(mode)
        self.legacy.eval()
        return self
