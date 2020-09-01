# データセットの読み込みとその際に実行する処理の設定
import numpy as np
import torch
from PIL import Image


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, df, root, transform=None):
        self.transform = transform
        self.img_id = df['Path'].values.tolist()
        self.img_label = df['Label'].values.tolist()
        self.root = root

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, i):
        image = Image.open('{}/{}'.format(self.root, self.img_id[i]))
        image = image.convert('L')
        if self.transform:
            image = self.transform(image)     
        label = self.img_label[i]

        return image, label
