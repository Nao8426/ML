# 各モデルのネットワーク構造
from torch import nn


# 各ネットワークで使用するためのResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, bn=True, act_fn=nn.ReLU(inplace=True)):
        self.bn = bn

        super().__init__()

        self.main_bn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            act_fn,

            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            act_fn,

            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
            act_fn,

            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            act_fn,

            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.activation = act_fn

    def __call__(self, x):
        if self.bn == True:
            h = self.main_bn(x)
        else:
            h = self.main(x)
        x = self.shortcut(x)
        return self.activation(h + x)


# ジェネレータの構造
class Generator(nn.Module):
    # 各層のチャンネル数
    inRes_C = 512
    Res1_C = 256
    Res1_midC = inRes_C // 4
    Res2_C = 128
    Res2_midC = Res1_C // 4
    Res3_C = 64
    Res3_midC = Res2_C // 4
    Res4_C = 32
    Res4_midC = Res3_C // 4

    def __init__(self, nz, width, height, channel):
        # 全結合層のチャンネル数を計算
        self.W = width // (2**4)
        self.H = height // (2**4)
        outFC_C = self.W * self.H * self.inRes_C

        # ネットワーク構造
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=nz, out_features=outFC_C),
            nn.BatchNorm1d(outFC_C),
            nn.ReLU(inplace=True)
        )

        self.res = nn.Sequential(
            ResBlock(self.inRes_C, self.Res1_C, self.Res1_midC, bn=True, act_fn=nn.ReLU(inplace=True)),
            nn.ConvTranspose2d(in_channels=self.Res1_C, out_channels=self.Res1_C, kernel_size=2, stride=2, padding=0),
            ResBlock(self.Res1_C, self.Res2_C, self.Res2_midC, bn=True, act_fn=nn.ReLU(inplace=True)),
            nn.ConvTranspose2d(in_channels=self.Res2_C, out_channels=self.Res2_C, kernel_size=2, stride=2, padding=0),
            ResBlock(self.Res2_C, self.Res3_C, self.Res3_midC, bn=True, act_fn=nn.ReLU(inplace=True)),
            nn.ConvTranspose2d(in_channels=self.Res3_C, out_channels=self.Res3_C, kernel_size=2, stride=2, padding=0),
            ResBlock(self.Res3_C, self.Res4_C, self.Res4_midC, bn=True, act_fn=nn.ReLU(inplace=True)),
            nn.ConvTranspose2d(in_channels=self.Res4_C, out_channels=self.Res4_C, kernel_size=2, stride=2, padding=0)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.Res4_C, out_channels=channel, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    # 順伝播
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.inRes_C, self.H, self.W)
        x = self.res(x)
        return self.conv(x)


# ディスクリミネータの構造
class Discriminator(nn.Module):
    Conv_C = 16
    Res1_C = 32
    Res1_midC = Conv_C // 4
    Res2_C = 64
    Res2_midC = Res1_C // 4
    Res3_C = 128
    Res3_midC = Res2_C // 4
    Res4_C = 256
    Res4_midC = Res3_C // 4

    def __init__(self, width, height, channel):
        W = width // (2**4)
        H = height // (2**4)
        inFC_C = W * H * self.Res4_C

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.Conv_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )

        self.res = nn.Sequential(
            ResBlock(self.Conv_C, self.Res1_C, self.Res1_midC, bn=False, act_fn=nn.LeakyReLU(0.2, inplace=True)),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.Res1_C, self.Res2_C, self.Res2_midC, bn=False, act_fn=nn.LeakyReLU(0.2, inplace=True)),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.Res2_C, self.Res3_C, self.Res3_midC, bn=False, act_fn=nn.LeakyReLU(0.2, inplace=True)),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.Res3_C, self.Res4_C, self.Res4_midC, bn=False, act_fn=nn.LeakyReLU(0.2, inplace=True))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=inFC_C, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.res(x)
        return self.fc(x).squeeze()


# エンコーダの構造
class Encoder(nn.Module):
    Conv_C = 32
    Res1_C = 64
    Res1_midC = Conv_C // 4
    Res2_C = 128
    Res2_midC = Res1_C // 4
    Res3_C = 256
    Res3_midC = Res2_C // 4
    Res4_C = 512
    Res4_midC = Res3_C // 4

    def __init__(self, nz, width, height, channel):
        W = width // (2**4)
        H = height // (2**4)
        inFC_C = W * H * self.Res4_C

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.Conv_C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.Conv_C),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )

        self.res = nn.Sequential(
            ResBlock(self.Conv_C, self.Res1_C, self.Res1_midC, bn=True, act_fn=nn.ReLU(inplace=True)),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.Res1_C, self.Res2_C, self.Res2_midC, bn=True, act_fn=nn.ReLU(inplace=True)),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.Res2_C, self.Res3_C, self.Res3_midC, bn=True, act_fn=nn.ReLU(inplace=True)),
            nn.AvgPool2d(kernel_size=2),
            ResBlock(self.Res3_C, self.Res4_C, self.Res4_midC, bn=True, act_fn=nn.ReLU(inplace=True))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=inFC_C, out_features=nz)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.res(x)
        return self.fc(x)


# コードディスクリミネータの構造
class CodeDiscriminator(nn.Module):
    def __init__(self, nz):    
        C = 1024
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=nz, out_features=C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=C, out_features=C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=C, out_features=C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(in_features=C, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).squeeze()
