# データセットの読み込みとその際に実行する処理の設定
import numpy as np
import torch
import torchvision
from PIL import Image


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, df, root, transform=None):
        self.img_id = df.values.tolist()
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, i):
        image = Image.open('{}/{}'.format(self.root, self.img_id[i][0]))
        if self.transform:
            image = self.transform(image)

        return image


# データセットに対する処理（正規化など）
class Trans():
    def __init__(self):
        self.norm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

    def __call__(self, image):
        image = image.convert('L')
        return self.norm(image)
