# オートエンコーダによる画像生成
import numpy as np
import os
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
# 自作モジュール
from network import AutoEncoder


def generate(savedir, model_path, _list, root):
    width = 28
    height = 28
    channel = 1

    device = 'cuda'

    # モデル設定
    model = AutoEncoder(width, height, channel)
    model = nn.DataParallel(model)
    model.module.load_state_dict(torch.load(model_path))
    model.eval()    # 推論モードへ切り替え（Dropoutなどの挙動に影響）

    # 保存先のファイルを作成
    if os.path.exists(savedir):
        n = 1
        while 1:
            if os.path.exists('{}({})'.format(savedir, n)):
                n += 1
            else:
                savedir = '{}({})'.format(savedir, n)
                break
    os.makedirs(savedir, exist_ok=True)

    df = pd.read_csv(_list, usecols=['Path'])
    img_id = df.values.tolist()

    for i, img in enumerate(img_id):
        image = Image.open('{}/{}'.format(root, img[0]))
        image = image.convert('L')
        image = np.array(image)

        gene_img = model(image)

        # オートエンコーダの出力画像を保存
        torchvision.utils.save_image(gene_img, "{}/{:04}.png".format(savedir, i+1))
