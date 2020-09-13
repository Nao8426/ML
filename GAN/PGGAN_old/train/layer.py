import torch
from torch import nn


class PixelNorm(nn.Module):
    def forward(self, x):
        eps = 1e-7
        mean = torch.mean(x**2, dim=1, keepdims=True)
        return x / (torch.sqrt(mean)+eps)


class WeightScale(nn.Module):
    def forward(self, x, gain=2):
        scale = (gain/x.shape[1])**0.5
        return x * scale


class MiniBatchStd(nn.Module):
    def forward(self, x):
        std = torch.std(x, dim=0, keepdim=True)
        mean = torch.mean(std, dim=(1,2,3), keepdim=True)
        n, c, h, w = x.shape
        mean = torch.ones(n,1,h,w, dtype=x.dtype, device=x.device)*mean
        return torch.cat((x,mean), dim=1)


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            WeightScale(),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=0),
            PixelNorm(),
            )
        nn.init.kaiming_normal_(self.layers[2].weight)

    def forward(self, x):
        return self.layers(x)


class ConvModuleG(nn.Module):
    '''
    Args:
        out_size: (int), Ex.: 16 (resolution)
        inch: (int),  Ex.: 256
        outch: (int), Ex.: 128
    '''
    def __init__(self, out_size, inch, outch, first=False):
        super().__init__()

        if first:
            layers = [
                Conv2d(in_ch=inch, out_ch=outch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(in_ch=outch, out_ch=outch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
            ]

        else:
            layers = [
                nn.Upsample((out_size[0], out_size[1]), mode='nearest'),
                Conv2d(in_ch=inch, out_ch=outch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(in_ch=outch, out_ch=outch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvModuleD(nn.Module):
    '''
    Args:
        out_size: (int), Ex.: 16 (resolution)
        inch: (int),  Ex.: 256
        outch: (int), Ex.: 128
    '''
    def __init__(self, out_size, inch, outch, final=False):
        super().__init__()

        if final:
            layers = [
                MiniBatchStd(), # final block only
                Conv2d(in_ch=inch+1, out_ch=outch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(in_ch=outch, out_ch=outch, kernel_size=4, padding=0), 
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(outch, 1, 1, padding=0), 
            ]
        else:
            layers = [
                Conv2d(in_ch=inch, out_ch=outch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(in_ch=outch, out_ch=outch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                nn.AdaptiveAvgPool2d((out_size[0], out_size[1])),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
