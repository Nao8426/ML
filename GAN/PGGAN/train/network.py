# ジェネレータとディスクリミネータのネットワーク構造
from torch import nn


# ジェネレータの構造
class Generator(nn.Module):
    # 各層のチャンネル数
    Conv1_C = 512
    Conv2_C = 256
    Conv3_C = 128
    Conv4_C = 64
    Conv5_C = 32
    Conv6_C = 16

    # 各解像度における終了エポック
    border = [100, 200, 300, 400, 500]

    # 層追加の際の残差処理を行うエポックの長さ
    tl = 50

    # ネットワーク構造
    def __init__(self, nz, width, height, channel):
        super().__init__()
        self.level1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=self.Conv1_C, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.Conv1_C, out_channels=self.Conv1_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.level2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=self.Conv1_C, out_channels=self.Conv2_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.Conv2_C, out_channels=self.Conv2_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.level3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=self.Conv2_C, out_channels=self.Conv3_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.Conv3_C, out_channels=self.Conv3_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.level4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=self.Conv3_C, out_channels=self.Conv4_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.Conv4_C, out_channels=self.Conv4_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.level5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=self.Conv4_C, out_channels=self.Conv5_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.Conv5_C, out_channels=self.Conv5_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.level6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=self.Conv5_C, out_channels=self.Conv6_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.Conv6_C, out_channels=self.Conv6_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(in_channels=self.Conv1_C, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.Conv2_C, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=self.Conv3_C, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=self.Conv4_C, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=self.Conv5_C, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=self.Conv6_C, out_channels=channel, kernel_size=1, stride=1, padding=0)

    # 順伝播
    def forward(self, x, epoch, alpha):
        x = self.level1(x)
        h = self.level1(x)
        if epoch <= self.border[0]:
            x = self.conv1(x)
        elif epoch > self.border[0] and epoch <= self.border[1]:
            x = self.conv2(x)
        elif epoch > self.border[1] and epoch <= self.border[2]:
            x = self.conv3(x)
        elif epoch > self.border[2] and epoch <= self.border[3]:
            x = self.conv4(x)
        elif epoch > self.border[3] and epoch <= self.border[4]:
            x = self.conv5(x)
        else:
            x = self.conv6(x)

        if epoch in self.border == True:
            alpha = 0
        return torch.sigmoid(x), alpha


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
            nn.BatchNorm2d(self.Conv1_C),
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

    def forward(self, x, epoch):
        return self.main(x).squeeze()
