# ジェネレータのネットワーク構造
from torch import nn


# ジェネレータの構造
class Generator(nn.Module):
    L_C = 256
    L1_C = 128
    L2_C = 64
    L3_C = 32

    def __init__(self, nz, width, height, channel):
        self.W = width // (2**4)
        self.H = height // (2**4)
        self.L_fc = self.W * self.H * self.L_C

        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=nz, out_features=self.L_fc),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.L_C, out_channels=self.L1_C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.L1_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=self.L1_C, out_channels=self.L2_C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.L2_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=self.L2_C, out_channels=self.L3_C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.L3_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=self.L3_C, out_channels=channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.L_C, self.H, self.W)
        return self.main(x)
