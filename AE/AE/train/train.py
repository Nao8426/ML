# AE(Auto Encoder)の学習
import numpy as np
import os
import pandas as pd
import statistics
import torch
import torchvision
from PIL import Image
from torch import nn
from tqdm import tqdm
# 自作モジュール
from load import LoadDataset, Trans
from network import AutoEncoder
from util import plot, output_env


# ロス
class MyLoss():
    def __init__(self):
        self.loss_MSE = nn.MSELoss()

    def loss(self, x, y):
        return self.loss_MSE(x, y)


def train(savedir, _list, root, epochs, batch_size):
    # Adam設定(default: lr=0.001, betas=(0.9, 0.999), weight_decay=0) 
    opt_para = {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0}

    device = 'cuda'

    # 保存先のファイルを作成
    if os.path.exists(savedir):
        num = 1
        while 1:
            if os.path.exists('{}({})'.format(savedir, num)):
                num += 1
            else:
                savedir = '{}({})'.format(savedir, num)
                break
    os.makedirs(savedir, exist_ok=True)
    os.makedirs('{}/generating_image'.format(savedir), exist_ok=True)
    os.makedirs('{}/model'.format(savedir), exist_ok=True)
    os.makedirs('{}/loss'.format(savedir), exist_ok=True)

    df = pd.read_csv(_list, usecols=['Path'])
    img_id = df.values.tolist()

    check_img = Image.open('{}/{}'.format(root, img_id[0][0]))
    check_img = check_img.convert('L')
    width, height = check_img.size

    myloss = MyLoss()

    # モデルの読み込み
    ae_model = AutoEncoder(width, height, 1)
    ae_model = nn.DataParallel(ae_model)
    ae_model = ae_model.to(device)

    # 最適化アルゴリズムの設定
    para = torch.optim.Adam(ae_model.parameters(), lr=opt_para['lr'], betas=opt_para['betas'], weight_decay=opt_para['weight_decay'])

    # ロスの推移
    result = []

    dataset = LoadDataset(df, root, transform=Trans())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    output_env('{}/env.txt'.format(savedir), batch_size, opt_para, ae_model)

    for epoch in range(epochs):
        print('########## epoch : {}/{} ##########'.format(epoch+1, epochs))

        log_loss = []

        for real_img in tqdm(train_loader):
            # GPU用の変数に変換
            real_img = real_img.to(device)

            # 真正画像をエンコーダに入力し，特徴ベクトルを取得
            output = ae_model(real_img)

            # デコーダの出力を保存
            output_tensor = output.view(batch_size, 1, height, width).detach()

            # ロス計算
            real_img = real_img.view(-1, width*height)
            loss = myloss.loss(real_img, output)
            log_loss.append(loss.item())

            # 重み更新
            para.zero_grad()
            loss.backward()
            para.step()

        result.append(statistics.mean(log_loss))
        print('loss = {}'.format(result[-1]))

        # ロスのログを保存
        with open('{}/loss/log.txt'.format(savedir), mode='a') as f:
            f.write('Epoch {:03}: {}\n'.format(epoch+1, result[-1]))
        
        # 定めた保存周期ごとにモデル，出力画像を保存する
        if (epoch+1) % 10 == 0:
            # モデルの保存
            torch.save(ae_model.module.state_dict(), '{}/model/model_{}.pth'.format(savedir, epoch+1))

            # オートエンコーダの出力画像を保存
            torchvision.utils.save_image(output_tensor[:batch_size], "{}/generating_image/epoch_{:03}.png".format(savedir, epoch+1))

        # 定めた保存周期ごとにロスを保存する
        if (epoch+1) % 50 == 0:
            x = np.linspace(1, epoch+1, epoch+1, dtype='int')
            plot(result, x, savedir)

    # 最後のエポックが保存周期でない場合に，保存する
    if (epoch+1)%10 != 0 and epoch+1 == epochs:
        torch.save(ae_model.module.state_dict(), '{}/model/model_{}.pth'.format(savedir, epoch+1))

        torchvision.utils.save_image(output_tensor[:batch_size], "{}/generating_image/epoch_{:03}.png".format(savedir, epoch+1))

        x = np.linspace(1, epoch+1, epoch+1, dtype='int')
        plot(result, x, savedir)
