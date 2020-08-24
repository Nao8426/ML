# データセットの読み込みとその際に実行する処理の設定
import numpy as np
import torch
from PIL import Image


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, df, root):
        self.img_id = df.values.tolist()
        self.root = root

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, i):
        image = Image.open('{}/{}'.format(self.root, self.img_id[i][0]))
        image = image.convert('L')
        image = np.array(image)

        return image
