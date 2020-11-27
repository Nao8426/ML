# DCGAN(Deep Convolutional GAN)の学習
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
from network import Generator, Discriminator
from util import plot, output_env


# ジェネレータ，ディスクリミネータのロス
class MyLoss():
    def __init__(self):
        self.BCE_loss = nn.BCELoss()

    def G_loss(self, x, ones):
        return self.BCE_loss(x, ones)

    def D_loss(self, p, ones, r, zeros):
        return self.BCE_loss(p, ones) + self.BCE_loss(r, zeros)


def train(savedir, _list, root, epochs, batch_size, nz):
    # 画像のチャンネル数
    channel = 1

    # ジェネレータのAdam設定(default: lr=0.001, betas=(0.9, 0.999), weight_decay=0) 
    G_opt_para = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}
    # ディスクリミネータのAdam設定
    D_opt_para = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}

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
    os.makedirs('{}/generating_image'.format(savedir), exist_ok=True)
    os.makedirs('{}/model'.format(savedir), exist_ok=True)
    os.makedirs('{}/loss'.format(savedir), exist_ok=True)

    df = pd.read_csv(_list, usecols=['Path'])
    img_id = df.values.tolist()

    check_img = Image.open('{}/{}'.format(root, img_id[0][0]))
    check_img = check_img.convert('L')
    width, height = check_img.size

    G_model, D_model = Generator(nz, width, height, channel), Discriminator(width, height, channel)
    G_model, D_model = nn.DataParallel(G_model), nn.DataParallel(D_model)
    G_model, D_model = G_model.to(device), D_model.to(device)

    # 最適化アルゴリズムの設定
    G_para = torch.optim.Adam(G_model.parameters(), lr=G_opt_para['lr'], betas=G_opt_para['betas'], weight_decay=G_opt_para['weight_decay'])
    D_para = torch.optim.Adam(D_model.parameters(), lr=D_opt_para['lr'], betas=D_opt_para['betas'], weight_decay=D_opt_para['weight_decay'])

    # ロスの推移を保存するためのリストを確保
    result = {}
    result['G_log_loss'] = []
    result['D_log_loss'] = []

    dataset = LoadDataset(df, root, transform=Trans())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    output_env('{}/env.txt'.format(savedir), batch_size, nz, G_opt_para, D_opt_para, G_model, D_model)

    # テスト用の一定乱数
    z0 = torch.randn(batch_size, nz, 1, 1)

    for epoch in range(epochs):
        print('########## epoch : {}/{} ##########'.format(epoch+1, epochs))

        G_log_loss, D_log_loss = [], []

        for real_img in tqdm(train_loader):
            # 入力の乱数を作成
            z = torch.randn(batch_size, nz, 1, 1)

            # GPU用の変数に変換
            real_img = real_img.to(device)
            z = z.to(device)

            # ジェネレータに入力
            fake_img = G_model(z)

            # ディスクリミネータに真正画像と生成画像を入力
            real_out = D_model(real_img)
            fake_out = D_model(fake_img)

            # ジェネレータのロス計算
            G_loss = myloss.G_loss(fake_out, torch.tensor(1.0).expand_as(fake_out).to(device))
            G_log_loss.append(G_loss.item())
            # ディスクリミネータのロス計算
            D_loss = myloss.D_loss(real_out, torch.tensor(1.0).expand_as(real_out).to(device), fake_out, torch.tensor(0.0).expand_as(fake_out).to(device))
            D_log_loss.append(D_loss.item())

            # ジェネレータの重み更新
            G_para.zero_grad()
            G_loss.backward(retain_graph=True)
            G_para.step()
            # ディスクリミネータの重み更新
            D_para.zero_grad()
            D_loss.backward()
            D_para.step()

        result['G_log_loss'].append(statistics.mean(G_log_loss))
        result['D_log_loss'].append(statistics.mean(D_log_loss))
        print('G_loss = {} , D_loss = {}'.format(result['G_log_loss'][-1], result['D_log_loss'][-1]))

        # ロスのログを保存
        with open('{}/loss/log.txt'.format(savedir), mode='a') as f:
            f.write('##### Epoch {:03} #####\n'.format(epoch+1))
            f.write('G: {}, D: {}\n'.format(result['G_log_loss'][-1], result['D_log_loss'][-1]))
        
        # 定めた保存周期ごとにモデル，出力画像を保存する
        if (epoch+1)%10 == 0:
            # モデルの保存
            torch.save(G_model.module.state_dict(), '{}/model/G_model_{}.pth'.format(savedir, epoch+1))
            torch.save(D_model.module.state_dict(), '{}/model/D_model_{}.pth'.format(savedir, epoch+1))

            G_model.eval()
            
            # メモリ節約のためパラメータの保存は止める（テスト時にパラメータの保存は不要）
            with torch.no_grad():
                fake_img_test = G_model(z0)

            # ジェネレータの出力画像を保存
            torchvision.utils.save_image(fake_img_test[:batch_size], "{}/generating_image/epoch_{:03}.png".format(savedir, epoch+1))

            G_model.train()

        # 定めた保存周期ごとにロスを保存する
        if (epoch+1)%50 == 0:
            x = np.linspace(1, epoch+1, epoch+1, dtype='int')
            plot(result['G_log_loss'], result['D_log_loss'], x, savedir)

    # 最後のエポックが保存周期でない場合に，保存する
    if (epoch+1)%10 != 0 and epoch+1 == epochs:
        torch.save(G_model.module.state_dict(), '{}/model/G_model_{}.pth'.format(savedir, epoch+1))
        torch.save(D_model.module.state_dict(), '{}/model/D_model_{}.pth'.format(savedir, epoch+1))

        G_model.eval()
        
        with torch.no_grad():
            fake_img_test = G_model(z0)

        torchvision.utils.save_image(fake_img_test[:batch_size], "{}/generating_image/epoch_{:03}.png".format(savedir, epoch+1))

        x = np.linspace(1, epoch+1, epoch+1, dtype='int')
        plot(result['G_log_loss'], result['D_log_loss'], x, savedir)
