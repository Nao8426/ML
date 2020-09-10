import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# 自作モジュール
from layer import ConvModuleG, ConvModuleD, MiniBatchStd


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # conv modules & toRGBs
        scale = 1
        inchs  = np.array([512,256,128,64,32,16], dtype=np.uint32)*scale
        outchs = np.array([256,128, 64,32,16, 8], dtype=np.uint32)*scale
        sizes = np.array([4,8,16,32,64,128], dtype=np.uint32)
        firsts = np.array([True, False, False, False, False, False], dtype=np.bool)
        blocks, toRGBs = [], []
        for s, inch, outch, first in zip(sizes, inchs, outchs, firsts):
            blocks.append(ConvModuleG(s, inch, outch, first))
            toRGBs.append(nn.Conv2d(outch, 3, 1, padding=0))

        self.blocks = nn.ModuleList(blocks)
        self.toRGBs = nn.ModuleList(toRGBs)

    def forward(self, x, res, eps=1e-7):
        # to image
        n, c = x.shape
        x = x.reshape(n,c//16,4,4)

        # for the highest resolution
        res = min(res, len(self.blocks))

        # get integer by floor
        nlayer = max(int(res-eps), 0)
        for i in range(nlayer):
            x = self.blocks[i](x)

        # high resolution
        x_big = self.blocks[nlayer](x)
        dst_big = self.toRGBs[nlayer](x_big)

        if nlayer==0:
            x = dst_big
        else:
            # low resolution
            x_sml = F.interpolate(x, x_big.shape[2:4], mode='nearest')
            dst_sml = self.toRGBs[nlayer-1](x_sml)
            alpha = res - int(res-eps)
            x = (1-alpha)*dst_sml + alpha*dst_big

        #return x, n, res
        return torch.sigmoid(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.minbatch_std = MiniBatchStd()

        # conv modules & toRGBs
        scale = 1
        inchs = np.array([256,128, 64,32,16, 8], dtype=np.uint32)*scale
        outchs  = np.array([512,256,128,64,32,16], dtype=np.uint32)*scale
        sizes = np.array([1,4,8,16,32,64], dtype=np.uint32)
        finals = np.array([True, False, False, False, False, False], dtype=np.bool)
        blocks, fromRGBs = [], []
        for s, inch, outch, final in zip(sizes, inchs, outchs, finals):
            fromRGBs.append(nn.Conv2d(3, inch, 1, padding=0))
            blocks.append(ConvModuleD(s, inch, outch, final=final))

        self.fromRGBs = nn.ModuleList(fromRGBs)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, res):
        # for the highest resolution
        res = min(res, len(self.blocks))

        # get integer by floor
        eps = 1e-7
        n = max(int(res-eps), 0)

        # high resolution
        x_big = self.fromRGBs[n](x)
        x_big = self.blocks[n](x_big)

        if n==0:
            x = x_big
        else:
            # low resolution
            x_sml = F.adaptive_avg_pool2d(x, x_big.shape[2:4])
            x_sml = self.fromRGBs[n-1](x_sml)
            alpha = res - int(res-eps)
            x = (1-alpha)*x_sml + alpha*x_big

        for i in range(n):
            x = self.blocks[n-1-i](x)

        return x
