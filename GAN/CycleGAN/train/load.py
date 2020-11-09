# データセットの読み込みとその際に実行する処理の設定
import numpy as np
import torch
from PIL import Image


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, df_A, df_B, root, transform=None):
        self.img_id_A = df_A.values.tolist()
        self.img_id_B = df_B.values.tolist()
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.img_id_A)

    def __getitem__(self, i):
        image_A = Image.open('{}/{}'.format(self.root, self.img_id_A[i][0]))
        if self.transform:
            image_A = self.transform(image_A)
        image_B = Image.open('{}/{}'.format(self.root, self.img_id_B[i][0]))
        if self.transform:
            image_B = self.transform(image_B)

        return image_A, image_B


# データセットに対する処理（正規化など）
class Trans():
    def __init__(self):
        self.norm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

    def __call__(self, image):
        image = image.convert('L')
        return self.norm(image)
