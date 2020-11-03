# データセットの読み込みとその際に実行する処理の設定
import numpy as np
import torch
import torchvision
from PIL import Image


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, df, root, transform=None):
        self.img_id = df['Path'].values.tolist()
        self.img_label = df['Label'].values.tolist()
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, i):
        image = Image.open('{}/{}'.format(self.root, self.img_id[i]))
        if self.transform:
            image = self.transform(image)     
        label = self.img_label[i]

        return image, label


# データセットに対する処理（正規化など）
class Trans():
    def __init__(self):
        self.norm = torchvision.transforms.ToTensor()

    def __call__(self, image):
        image = image.convert('L')
        return self.norm(image)
