# 各モデルのネットワーク構造
from torch import nn


# 各モデルで使用するためのResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def __call__(self, x):
        h = self.main(x)
        h = self.activation(h + x)
        return self.conv(h)


# ジェネレータの構造
class Generator(nn.Module):
    # 各層のチャンネル数
    L1_C = 512
    L2_C = 256
    L2_midC = L1_C // 4
    L3_C = 128
    L3_midC = L2_C // 4
    L4_C = 64
    L4_midC = L3_C // 4
    L5_C = 32
    L5_midC = L4_C // 4

    def __init__(self, nz, width, height, channel):
        # 全結合層のチャンネル数を計算
        self.W = width // (2**4)
        self.H = height // (2**4)
        self.L_fc = self.W * self.H * self.L1_C

        # ネットワーク構造
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=nz, out_features=self.L_fc),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.res = nn.Sequential(
            ResBlock(self.L1_C, self.L2_C, self.L2_midC),
            nn.ConvTranspose2d(in_channels=self.L2_C, out_channels=self.L2_C, kernel_size=2, stride=2, padding=0, bias=False),
            ResBlock(self.L2_C, self.L3_C, self.L3_midC),
            nn.ConvTranspose2d(in_channels=self.L3_C, out_channels=self.L3_C, kernel_size=2, stride=2, padding=0, bias=False),
            ResBlock(self.L3_C, self.L4_C, self.L4_midC),
            nn.ConvTranspose2d(in_channels=self.L4_C, out_channels=self.L4_C, kernel_size=2, stride=2, padding=0, bias=False),
            ResBlock(self.L4_C, self.L5_C, self.L5_midC),
            nn.ConvTranspose2d(in_channels=self.L5_C, out_channels=self.L5_C, kernel_size=2, stride=2, padding=0, bias=False)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.L5_C, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    # 順伝播
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.L1_C, self.H, self.W)
        x = self.res(x)
        return self.conv(x)


# ディスクリミネータの構造
class Discriminator(nn.Module):
    L1_C = 16
    L2_C = 32
    L2_midC = L1_C // 4
    L3_C = 64
    L3_midC = L2_C // 4
    L4_C = 128
    L4_midC = L3_C // 4
    L5_C = 256
    L5_midC = L4_C // 4

    def __init__(self, width, height, channel):
        self.W = width // (2**4)
        self.H = height // (2**4)
        self.L_fc = self.W * self.H * self.L5_C

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.L1_C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )

        self.res = nn.Sequential(
            ResBlock(self.L1_C, self.L2_C, self.L2_midC),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.L2_C, self.L3_C, self.L3_midC),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.L3_C, self.L4_C, self.L4_midC),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.L4_C, self.L5_C, self.L5_midC)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.L_fc, out_features=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.res(x)
        return self.fc(x).squeeze()


# エンコーダの構造
class Encoder(nn.Module):
    L1_C = 32
    L2_C = 64
    L2_midC = L1_C // 4
    L3_C = 128
    L3_midC = L2_C // 4
    L4_C = 256
    L4_midC = L3_C // 4
    L5_C = 512
    L5_midC = L4_C // 4

    def __init__(self, nz, width, height, channel):
        self.W = width // (2**4)
        self.H = height // (2**4)
        self.L_fc = self.W * self.H * self.L5_C

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.L1_C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )

        self.res = nn.Sequential(
            ResBlock(self.L1_C, self.L2_C, self.L2_midC),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.L2_C, self.L3_C, self.L3_midC),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.L3_C, self.L4_C, self.L4_midC),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.L4_C, self.L5_C, self.L5_midC)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.L_fc, out_features=nz),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.res(x)
        return self.fc(x)


# コードディスクリミネータの構造
class CodeDiscriminator(nn.Module):
    def __init__(self, nz):    
        C = nz // 2
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=nz, out_features=C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=C, out_features=C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=C, out_features=C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=C, out_features=1)
        )

    def forward(self, x):
        return self.main(x).squeeze()
