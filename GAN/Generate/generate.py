import cv2
import os
import sys
import torch
import torchvision
from torch import nn
from tqdm import tqdm
# 自作モジュール
from network import Generator


def generate(savedir, root, num, batch_size, nz, width, height, channel):
    device = 'cuda'

    model = Generator(nz, width, height, channel)
    model = nn.DataParallel(model)
    model = model.to(device)

    model.eval()    # 推論モードへ切り替え（Dropoutなどの挙動に影響）

    # 保存先のファイルを作成
    if os.path.exists('{}/output.mp4'.format(savedir)):
        while 1:
            check = input('ファイルがすでに存在します．上書きしますか？（y/n）: ')
            if check == 'y':
                break
            elif check == 'n':
                sys.exit()
            else:
                print('不正な入力です．もう一度入力してください．')
    os.makedirs(savedir, exist_ok=True)

    # エンコード（for mp4）
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # アウトプットの動画を用意
    video = cv2.VideoWriter('{}/output.mp4'.format(savedir), fourcc, 20.0, (262, 2066)) # バッチサイズ: 16
    # video = cv2.VideoWriter('{}/output.mp4'.format(savedir), fourcc, 20.0, (522, 2066)) # バッチサイズ: 32
    
    # 入力
    _input = torch.randn(batch_size, nz, 1, 1)
    _input = _input.to(device)

    for i in tqdm(range(num)):
        # モデルの読み込み
        model.module.load_state_dict(torch.load('{}/G_model_{}.pth'.format(root, (i+1)*10)))

        gene_img = model(_input)

        # ジェネレータの出力画像を保存
        output = torchvision.utils.make_grid(gene_img[:batch_size])

        video.write(output)

    video.release()
