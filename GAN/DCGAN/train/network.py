# ジェネレータとディスクリミネータのネットワーク構造
from torch import nn


# ジェネレータの構造
class Generator(nn.Module):
    # 各層のチャンネル数
    inConv_C = 512
    Conv1_C = 256
    Conv2_C = 128
    Conv3_C = 64

    def __init__(self, nz, width, height, channel):
        # 全結合層のチャンネル数を計算
        self.W = width // (2**4)
        self.H = height // (2**4)
        outFC_C = self.W * self.H * self.inConv_C

        # ネットワーク構造
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=nz, out_features=outFC_C),
            nn.ReLU(inplace=True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.inConv_C, out_channels=self.Conv1_C, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.Conv1_C, out_channels=self.Conv2_C, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.Conv2_C, out_channels=self.Conv3_C, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.Conv3_C, out_channels=channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    # 順伝播
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.inConv_C, self.H, self.W)
        return self.main(x)


# ディスクリミネータの構造
class Discriminator(nn.Module):
    Conv1_C = 32
    Conv2_C = 64
    Conv3_C = 128
    Conv4_C = 256

    def __init__(self, width, height, channel):
        W = width // (2**4)
        H = height // (2**4)
        inFC_C = W * H * self.Conv4_C

        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.Conv1_C, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.Conv1_C, out_channels=self.Conv2_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv2_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.Conv2_C, out_channels=self.Conv3_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv3_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.Conv3_C, out_channels=self.Conv4_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv4_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=inFC_C, out_features=1)
        )

    def forward(self, x):
        return self.main(x).squeeze()
