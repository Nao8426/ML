# エンコーダとデコーダのネットワーク構造
from torch import nn


# エンコーダの構造
class Encoder(nn.Module):
    L1_C = 32
    L2_C = 64
    L3_C = 128
    L4_C = 256

    def __init__(self, nz, width, height, channel):
        self.W = width // (2**4)
        self.H = height // (2**4)
        self.in_FC = self.W * self.H * self.L3_C

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.L1_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=self.L1_C, out_channels=self.L2_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=self.L2_C, out_channels=self.L3_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=self.L3_C, out_channels=self.L4_C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.in_FC, out_features=nz),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


# エンコーダの構造
class Decoder(nn.Module):
    in_C = 16
    L1_C = 32
    L1_C = 64
    L1_C = 128
    L1_C = 256

    def __init__(self, nz, width, height, channel):
        self.W = width // (2**4)
        self.H = height // (2**4)
        self.L_fc = self.W * self.H * self.in_C

        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=nz, out_features=self.L_fc),
            nn.LeakyReLU(0.2)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.in_C, out_channels=self.L1_C, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=self.L1_C, out_channels=self.L2_C, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=self.L2_C, out_channels=self.L3_C, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=self.L3_C, out_channels=self.L4_C, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.in_C, self.H, self.W)
        return self.conv(x)
