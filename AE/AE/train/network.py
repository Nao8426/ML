# エンコーダとデコーダのネットワーク構造
from torch import nn


# エンコーダの構造
class Encoder(nn.Module):
    L1_C = 128
    L2_C = 64
    L3_C = 32
    L4_C = 16

    def __init__(self, width, height, channel):
        self.in_C = width * height * channel

        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.in_C, out_features=self.L1_C),
            nn.BatchNorm2d(self.L1_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.L1_C, out_features=self.L2_C),
            nn.BatchNorm2d(self.L2_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.L2_C, out_features=self.L3_C),
            nn.BatchNorm2d(self.L3_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.L3_C, out_features=self.L4_C)
        )

    def forward(self, x):
        return self.main(x)


# デコーダの構造
class Decoder(nn.Module):
    in_C = 16
    L1_C = 32
    L2_C = 64
    L3_C = 128

    def __init__(self, width, height, channel):
        self.L4_C = width * height * channel

        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=self.in_C, out_features=self.L1_C),
            nn.BatchNorm2d(self.L1_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.L1_C, out_features=self.L2_C),
            nn.BatchNorm2d(self.L2_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.L2_C, out_features=self.L3_C),
            nn.BatchNorm2d(self.L3_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.L3_C, out_features=self.L4_C),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# オートエンコーダの構造
class AutoEncoder(nn.Module):
    def __init__(self, width, height, channel):
        super().__init__()
        self.enc = Encoder(width, height, channel)
        self.dec = Decoder(width, height, channel)

    def forward(self, x):
        x = self.enc(x)
        return self.dec(x)
