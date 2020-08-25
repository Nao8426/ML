# CNNのネットワーク構造
from torch import nn


# CNNの構造
class CNN(nn.Module):
    L_C = 32
    L_FC1 = 128
    L_FC2 = 10
    CHANNEL = 13*13*L_C

    def __init__(self, channel):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.L_C, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Dropout(0.25),
            nn.Flatten(),

            nn.Linear(in_features=self.CHANNEL, out_features=self.L_FC1),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=self.L_FC1, out_features=self.L_FC2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.main(x)
