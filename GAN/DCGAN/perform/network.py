# ネットワーク構造
from torch import nn


# ジェネレータの構造
class Generator(nn.Module):
    def __init__(self, nz, width, height, channel):
        self.L1_C = 256
        self.L2_C = 128
        self.L3_C = 64
        self.L4_C = 32

        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=self.L1_C, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.L1_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L1_C, out_channels=self.L2_C, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.L2_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L2_C, out_channels=self.L3_C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.L3_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L3_C, out_channels=self.L4_C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.L4_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L4_C, out_channels=channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)
