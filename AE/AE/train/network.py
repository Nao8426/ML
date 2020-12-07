# エンコーダとデコーダのネットワーク構造
from torch import nn


# エンコーダの構造
class Encoder(nn.Module):
    Conv1_C = 64
    Conv2_C = 128
    Conv3_C = 256
    Conv4_C = 512
    FC_C = 64

    def __init__(self, width, height, channel):
        W = width // (2**4)
        H = height // (2**4)
        inFC_C = W * H * self.Conv4_C

        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.Conv1_C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv1_C),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=self.Conv1_C, out_channels=self.Conv2_C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv2_C),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=self.Conv2_C, out_channels=self.Conv3_C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv3_C),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=self.Conv3_C, out_channels=self.Conv4_C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv4_C),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=inFC_C, out_features=self.FC_C)
        )

    def forward(self, x):
        return self.main(x)


# デコーダの構造
class Decoder(nn.Module):
    inFC_C = 64
    inConv_C = 512
    Conv1_C = 256
    Conv2_C = 128
    Conv3_C = 64

    def __init__(self, width, height, channel):
        self.W = width // (2**4)
        self.H = height // (2**4)
        outFC_C = self.W * self.H * self.inConv_C

        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.inFC_C, out_features=outFC_C),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.inConv_C, out_channels=self.Conv1_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv1_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.Conv1_C, out_channels=self.Conv2_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv2_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.Conv2_C, out_channels=self.Conv3_C, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Conv3_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.Conv3_C, out_channels=channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.inConv_C, self.H, self.W)
        return self.conv(x)


# オートエンコーダの構造
class AutoEncoder(nn.Module):
    def __init__(self, width, height, channel):
        super().__init__()
        self.enc = Encoder(width, height, channel)
        self.dec = Decoder(width, height, channel)

    def forward(self, x):
        x = self.enc(x)
        return self.dec(x)
