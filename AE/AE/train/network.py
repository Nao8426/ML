# エンコーダとデコーダのネットワーク構造
from torch import nn


# エンコーダの構造
class Encoder(nn.Module):
    FC1_C = 128
    FC2_C = 64
    FC3_C = 32
    FC4_C = 16

    def __init__(self, width, height, channel):
        inFC_C = width * height * channel

        super().__init__()

        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=inFC_C, out_features=self.FC1_C),
            nn.BatchNorm2d(self.FC1_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.FC1_C, out_features=self.FC2_C),
            nn.BatchNorm2d(self.FC2_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.FC2_C, out_features=self.FC3_C),
            nn.BatchNorm2d(self.FC3_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.FC3_C, out_features=self.FC4_C)
        )

    def forward(self, x):
        return self.main(x)


# デコーダの構造
class Decoder(nn.Module):
    inFC_C = 16
    FC1_C = 32
    FC2_C = 64
    FC3_C = 128

    def __init__(self, width, height, channel):
        FC4_C = width * height * channel

        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(in_features=self.inFC_C, out_features=self.FC1_C),
            nn.BatchNorm2d(self.FC1_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.FC1_C, out_features=self.FC2_C),
            nn.BatchNorm2d(self.FC2_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.FC2_C, out_features=self.FC3_C),
            nn.BatchNorm2d(self.FC3_C),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.FC3_C, out_features=FC4_C),
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
