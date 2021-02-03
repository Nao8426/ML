# CNN(Convolutional Neural Network)の学習
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
from network import CNN
from util import evaluate, plot, output_env


# ロスの計算
class MyLoss():
    def __init__(self):
        self.CE_loss = nn.CrossEntropyLoss()

    def loss(self, x, y):
        return self.CE_loss(x, y)


# 学習用関数
def train(savedir, train_list, test_list, root, epochs, batch_size):
    # 入力画像のチャンネル数
    width = 28
    height = 28
    channel = 1

    # Adam設定(default: lr=0.001, betas=(0.9, 0.999), weight_decay=0) 
    opt_para = {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0}

    device = 'cuda'

    myloss = MyLoss()

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
    os.makedirs('{}/model'.format(savedir), exist_ok=True)
    os.makedirs('{}/loss'.format(savedir), exist_ok=True)

    # トレーニングデータとテストデータのリストを読み込み
    df_train = pd.read_csv(train_list)
    df_test = pd.read_csv(test_list)

    # モデルの読み込み
    model = CNN(width, height, channel)
    model = nn.DataParallel(model)
    model = model.to(device)

    # 最適化アルゴリズムの設定
    para = torch.optim.Adam(model.parameters(), lr=opt_para['lr'], betas=opt_para['betas'], weight_decay=opt_para['weight_decay'])

    # ロスの推移を保存するためのリストを確保
    result = []

    # データセットのローダーを作成
    train_dataset = LoadDataset(df_train, root, transform=Trans())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = LoadDataset(df_test, root, transform=Trans())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # パラメータ設定，ネットワーク構造などの環境をテキストファイルで保存．
    output_env('{}/env.txt'.format(savedir), batch_size, opt_para, model)

    for epoch in range(epochs):
        print('#################### epoch: {}/{} ####################'.format(epoch+1, epochs))

        log_loss = []

        for img, label in tqdm(train_loader):
            # 画像とラベルをGPU用変数に設定
            img = img.to(device)
            label = label.to(device)

            # モデルに画像を入力
            output = model(img)

            # ロスを計算
            loss = myloss.loss(output, label)
            log_loss.append(loss.item())

            # 微分計算，重み更新
            para.zero_grad()
            loss.backward()
            para.step()

        # ロスのログを保存し，各エポック終わりにロスを表示．
        result.append(statistics.mean(log_loss))
        print('loss = {}'.format(result[-1]))

        # ロスのログを保存
        with open('{}/loss/log.txt'.format(savedir), mode='a') as f:
            f.write('Epoch {:03}: {}\n'.format(epoch+1, result[-1]))
        
        # 定めた保存周期ごとにモデル，ロスを保存．
        if (epoch+1)%10 == 0:
            # モデルの保存
            torch.save(model.module.state_dict(), '{}/model/model_{}.pth'.format(savedir, epoch+1))

        if (epoch+1)%50 == 0:
            # ロス（画像）の保存
            x = np.linspace(1, epoch+1, epoch+1, dtype='int')
            plot(result, x, savedir)

        # 各エポック終わりに，テストデータに対する精度を計算．
        evaluate(model, test_loader)

    # 最後のエポックが保存周期でない場合に，保存．
    if epoch+1 == epochs and (epoch+1)%10 != 0:
        torch.save(model.module.state_dict(), '{}/model/model_{}.pth'.format(savedir, epoch+1))

        x = np.linspace(1, epoch+1, epoch+1, dtype='int')
        plot(result, x, savedir)
