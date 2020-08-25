# ジェネレータとディスクリミネータのネットワーク構造
from torch import nn


# ジェネレータの構造
class Generator(nn.Module):
    # 各層のチャンネル数
    L_C = 512
    L1_C = 256
    L2_C = 128
    L3_C = 64

    def __init__(self, nz, width, height, channel):
        # 全結合層のチャンネル数を計算
        self.W = width // (2**4)
        self.H = height // (2**4)
        self.L_fc = self.W * self.H * self.L_C

        # ネットワーク構造
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=nz, out_features=self.L_fc),
            nn.ReLU(inplace=True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.L_C, out_channels=self.L1_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.L1_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L1_C, out_channels=self.L2_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.L2_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L2_C, out_channels=self.L3_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.L3_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L3_C, out_channels=channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    # 順伝播
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.L_C, self.H, self.W)
        return self.main(x)


# ディスクリミネータの構造
class Discriminator(nn.Module):
    L1_C = 32
    L2_C = 64
    L3_C = 128
    L4_C = 256

    def __init__(self, width, height, channel):
        self.W = width // (2**4)
        self.H = height // (2**4)
        self.L_fc = self.W * self.H * self.L4_C

        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.L1_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.L1_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.L1_C, out_channels=self.L2_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.L2_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.L2_C, out_channels=self.L3_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.L3_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.L3_C, out_channels=self.L4_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.L4_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=self.L_fc, out_features=1)
        )

    def forward(self, x):
        return self.main(x).squeeze()
