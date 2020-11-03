# ジェネレータとディスクリミネータのネットワーク構造
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channel)
        )

    # 順伝播
    def forward(self, x):
        return x + self.block(x)


# ジェネレータの構造
class Generator(nn.Module):
    # 各層のチャンネル数
    Conv1_C = 64
    Conv2_C = 128
    Conv3_C = 256
    Conv4_C = 128
    Conv5_C = 64

    # resblockの深さ
    res_dep = 6

    def __init__(self, channel):
        # ネットワーク構造
        super().__init__()
        
        self.encode_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=channel, out_channels=self.Conv1_C, kernel_size=7, stride=1),
            nn.InstanceNorm2d(self.Conv1_C),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=self.Conv1_C, out_channels=self.Conv2_C, kernel_size=3,stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(self.Conv2_C),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=self.Conv2_C, out_channels=self.Conv3_C, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(self.Conv3_C),
            nn.ReLU(inplace=True),
        )

        res_blocks = [ResidualBlock(self.Conv3_C) for _ in range(self.res_dep)]

        self.res_block = nn.Sequential(
            *res_blocks
        )

        self.decode_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.Conv3_C, out_channels=self.Conv4_C, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(self.Conv4_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.Conv4_C, out_channels=self.Conv5_C, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(self.Conv5_C),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=self.Conv5_C, out_channels=channel, kernel_size=7, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encode_block(x)
        x = self.res_block(x)
        return self.decode_block(x)


# ディスクリミネータの構造
class Discriminator(nn.Module):
    Conv1_C = 64
    Conv2_C = 128
    Conv3_C = 256
    Conv4_C = 512

    def __init__(self, channel):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.Conv1_C, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.Conv1_C, out_channels=self.Conv2_C, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(self.Conv2_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.Conv2_C, out_channels=self.Conv3_C, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(self.Conv3_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.Conv3_C, out_channels=self.Conv4_C, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(self.Conv4_C),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.Conv4_C, out_channels=1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.block(x).squeeze()
