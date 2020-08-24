# CNN(Convolutional Neural Network)の学習
import numpy as np
import os
import pandas as pd
import pickle
import statistics
import torch
import torchvision
from PIL import Image
from torch import nn
from tqdm import tqdm
# 自作モジュール
from load import LoadDataset
from network import CNN
from util import evaluate, plot, output_env


# ロスの計算
class MyLoss():
    def __init__(self):
        self.loss_CEL = nn.CrossEntropyLoss()

    def loss(self, x, y):
        return self.loss_CEL(x, y)


# 学習用関数
def train(savedir, train_list, test_list, root, epochs, batch_size):
    # CNNのAdam設定
    lr = 0.001  # default: 0.001
    betas = (0.9, 0.999)    # default: (0.9, 0.999)
    weight_decay = 0    # default: 0

    # 入力画像のチャンネル数
    channel = 1

    # モデル等の保存周期
    rotate = 10

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
    os.makedirs('{}/logs'.format(savedir), exist_ok=True)

    myloss = MyLoss()

    device = 'cuda'

    # モデルの読み込み
    model = CNN(channel)
    model = nn.DataParallel(model)
    model = model.to(device)

    # 最適化アルゴリズムの設定
    para = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # ロスの推移を保存するためのリストを確保
    result = {}
    result['log_loss'] = []

    # トレーニングデータとテストデータを読み込み（ローダーを作成）
    df_train = pd.read_csv(train_list)
    train_loader = LoadDataset(df_train, root)
    train_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=True)
    df_test = pd.read_csv(test_list)
    test_loader = LoadDataset(df_test, root)
    test_dataset = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True, drop_last=True)

    # パラメータ設定，ネットワーク構造などの環境をテキストファイルで保存．
    output_env('{}/env.txt'.format(savedir), batch_size, lr, betas, weight_decay, model)

    for epoch in range(epochs):
        print('#################### epoch: {}/{} ####################'.format(epoch+1, epochs))

        log_loss = []

        for img, label in tqdm(train_dataset):
            img = img.float()
            # 0~1に正規化
            img = img / 255
            # 3次元テンソルを4次元テンソルに変換（1チャネルの情報を追加）
            batch, height, width = img.shape
            img = torch.reshape(img, (batch_size, 1, height, width))

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
        result['log_loss'].append(statistics.mean(log_loss))
        print('loss =', result['log_loss'][-1])
        
        # 定めた保存周期ごとにモデル，ロス，ログを保存．
        if (epoch+1) % rotate == 0:
            # モデルの保存
            torch.save(model.module.state_dict(), '{}/model/model_{}.pth'.format(savedir, epoch+1))

            # ロス（画像）の保存
            x = np.linspace(1, epoch+1, epoch+1, dtype='int')
            plot(result['log_loss'], x, savedir)
    
            # ログの保存
            with open('{}/logs/logs_{}.pkl'.format(savedir, epoch+1), 'wb') as fp:
                pickle.dump(result, fp)

        # 各エポック終わりに，テストデータに対する精度を計算．
        evaluate(model, root, test_dataset, batch_size)

    # 最後のエポックが保存周期でない場合に，保存．
    if epoch+1 == epochs and (epoch+1)%rotate != 0:
        torch.save(model.module.state_dict(), '{}/model/model_{}.pth'.format(savedir, epoch+1))
        x = np.linspace(1, epoch+1, epoch+1, dtype='int')
        plot(result['log_loss'], x, savedir)
        with open('{}/logs/logs_{}.pkl'.format(savedir, epoch+1), 'wb') as fp:
            pickle.dump(result, fp)
