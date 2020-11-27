# alpha-GAN(alpha Generative Adversarial Net)の学習
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
from network import Generator, Discriminator, Encoder, CodeDiscriminator
from util import plot, output_env


# 各モデルのロス
class MyLoss():
    def __init__(self):
        self.BCE_loss = nn.BCELoss()
        self.L1_loss = nn.L1Loss()

    def G_loss(self, real_img, fake_img, fake_y, rnd_y, ones, alpha=1.0):
        return self.L1_loss(real_img, fake_img) + alpha*(self.BCE_loss(fake_y, ones)/2 + self.BCE_loss(rnd_y, ones)/2)

    def D_loss(self, real_y, ones, fake_y, rnd_y, zeros):
        return self.BCE_loss(real_y, ones) + self.BCE_loss(fake_y, zeros)/2 + self.BCE_loss(rnd_y, zeros)/2

    def E_loss(self, real_img, fake_img, real_cy, ones, alpha=1.0):
        return self.L1_loss(real_img, fake_img) + alpha*self.BCE_loss(real_cy, ones)

    def CD_loss(self, real_cy, zeros, rnd_cy, ones):
        return self.BCE_loss(real_cy, zeros) + self.BCE_loss(rnd_cy, ones)


def train(savedir, _list, root, epochs, batch_size, nz):
    # 画像のチャンネル数
    channel = 1

    # ジェネレータのAdam設定(default: lr=0.001, betas=(0.9, 0.999), weight_decay=0) 
    G_opt_para = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}
    # ディスクリミネータのAdam設定
    D_opt_para = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}
    # エンコーダのAdam設定
    E_opt_para = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}
    # コードディスクリミネータのAdam設定
    CD_opt_para = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}

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
    os.makedirs('{}/generating_image_rnd'.format(savedir), exist_ok=True)
    os.makedirs('{}/model'.format(savedir), exist_ok=True)
    os.makedirs('{}/loss'.format(savedir), exist_ok=True)

    myloss = MyLoss()

    df = pd.read_csv(_list, usecols=['Path'])
    img_id = df.values.tolist()

    check_img = Image.open('{}/{}'.format(root, img_id[0][0]))
    check_img = check_img.convert('L')
    width, height = check_img.size

    G_model, D_model, E_model, CD_model = Generator(nz, width, height, channel), Discriminator(width, height, channel), Encoder(nz, width, height, channel), CodeDiscriminator(nz)
    G_model, D_model, E_model, CD_model = nn.DataParallel(G_model), nn.DataParallel(D_model), nn.DataParallel(E_model), nn.DataParallel(CD_model)
    G_model, D_model, E_model, CD_model = G_model.to(device), D_model.to(device), E_model.to(device), CD_model.to(device)

    # 最適化アルゴリズムの設定
    G_para = torch.optim.Adam(G_model.parameters(), lr=G_opt_para['lr'], betas=G_opt_para['betas'], weight_decay=G_opt_para['weight_decay'])
    D_para = torch.optim.Adam(D_model.parameters(), lr=D_opt_para['lr'], betas=D_opt_para['betas'], weight_decay=D_opt_para['weight_decay'])
    E_para = torch.optim.Adam(E_model.parameters(), lr=E_opt_para['lr'], betas=E_opt_para['betas'], weight_decay=E_opt_para['weight_decay'])
    CD_para = torch.optim.Adam(CD_model.parameters(), lr=CD_opt_para['lr'], betas=CD_opt_para['betas'], weight_decay=CD_opt_para['weight_decay'])

    # ロスの推移を保存するためのリストを確保
    result = {}
    result['G_log_loss'] = []
    result['D_log_loss'] = []
    result['E_log_loss'] = []
    result['CD_log_loss'] = []

    dataset = LoadDataset(df, root, transform=Trans())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    output_env('{}/env.txt'.format(savedir), batch_size, nz, G_opt_para, D_opt_para, E_opt_para, CD_opt_para, G_model, D_model, E_model, CD_model)

    # テスト用の一定乱数
    z0 = torch.randn(batch_size, nz, 1, 1)

    for epoch in range(epochs):
        print('########## epoch : {}/{} ##########'.format(epoch+1, epochs))

        G_log_loss, D_log_loss, E_log_loss, CD_log_loss = [], [], [], []

        for real_img in tqdm(train_loader):
            # 入力の乱数を作成
            rnd_z = torch.randn(batch_size, nz, 1, 1)

            # GPU用の変数に変換
            real_img = real_img.to(device)
            rnd_z = rnd_z.to(device)

            # 真正画像をエンコーダに入力し，特徴ベクトルを取得
            real_z = E_model(real_img)

            # 特徴ベクトルをジェネレータに入力し，画像を生成
            fake_img = G_model(real_z)
            rnd_img = G_model(rnd_z)

            # 特徴ベクトルをコードディスクリミネータに入力し，判定結果を取得
            real_cy = CD_model(real_z)
            rnd_cy = CD_model(rnd_z)

            # 真正画像および生成画像をディスクリミネータに入力し，判定結果を取得
            real_y = D_model(real_img)
            fake_y = D_model(fake_img)
            rnd_y = D_model(rnd_img)

            # エンコーダのロス計算
            E_loss = myloss.E_loss(real_img, fake_img, real_cy, torch.tensor(1.0).expand_as(real_cy).to(device), 1.0)
            E_log_loss.append(E_loss.item())
            # ジェネレータのロス計算
            G_loss = myloss.G_loss(real_img, fake_img, fake_y, rnd_y, torch.tensor(1.0).expand_as(fake_y).to(device), 1.0)
            G_log_loss.append(G_loss.item())
            # コードディスクリミネータのロス計算
            CD_loss = myloss.CD_loss(real_cy, torch.tensor(0.0).expand_as(real_cy).to(device), rnd_cy, torch.tensor(1.0).expand_as(rnd_cy).to(device))
            CD_log_loss.append(CD_loss.item())
            # ディスクリミネータのロス計算
            D_loss = myloss.D_loss(real_y, torch.tensor(1.0).expand_as(real_y).to(device), fake_y, rnd_y, torch.tensor(0.0).expand_as(fake_y).to(device))
            D_log_loss.append(D_loss.item())

            # エンコーダの重み更新
            E_para.zero_grad()
            E_loss.backward(retain_graph=True)
            E_para.step()
            # ジェネレータの重み更新
            G_para.zero_grad()
            G_loss.backward(retain_graph=True)
            G_para.step()
            # コードディスクリミネータの重み更新
            CD_para.zero_grad()
            CD_loss.backward(retain_graph=True)
            CD_para.step()
            # ディスクリミネータの重み更新
            D_para.zero_grad()
            D_loss.backward()
            D_para.step()

        result['G_log_loss'].append(statistics.mean(G_log_loss))
        result['D_log_loss'].append(statistics.mean(D_log_loss))
        result['E_log_loss'].append(statistics.mean(E_log_loss))
        result['CD_log_loss'].append(statistics.mean(CD_log_loss))
        print('G_loss = {} , D_loss = {} , E_loss = {} , CD_loss = {}'.format(result['G_log_loss'][-1], result['D_log_loss'][-1], result['E_log_loss'][-1], result['CD_log_loss'][-1]))

        # ロスのログを保存
        with open('{}/loss/log.txt'.format(savedir), mode='a') as f:
            f.write('##### Epoch {:03} #####\n'.format(epoch+1))
            f.write('G: {}, D: {}, E: {}, CD: {}\n'.format(result['G_log_loss'][-1], result['D_log_loss'][-1], result['E_log_loss'][-1], result['CD_log_loss'][-1]))
        
        # 定めた保存周期ごとにモデル，出力画像を保存する
        if (epoch+1)%10 == 0:
            # モデルの保存
            torch.save(G_model.module.state_dict(), '{}/model/G_model_{}.pth'.format(savedir, epoch+1))
            torch.save(D_model.module.state_dict(), '{}/model/D_model_{}.pth'.format(savedir, epoch+1))
            torch.save(E_model.module.state_dict(), '{}/model/E_model_{}.pth'.format(savedir, epoch+1))
            torch.save(CD_model.module.state_dict(), '{}/model/CD_model_{}.pth'.format(savedir, epoch+1))

            G_model.eval()
            
            # メモリ節約のためパラメータの保存は止める（テスト時にパラメータの保存は不要）
            with torch.no_grad():
                rnd_img_test = G_model(z0)

            # ジェネレータの出力画像を保存
            torchvision.utils.save_image(fake_img[:batch_size], "{}/generating_image/epoch_{:03}.png".format(savedir, epoch+1))
            torchvision.utils.save_image(rnd_img_test[:batch_size], "{}/generating_image_rnd/epoch_{:03}.png".format(savedir, epoch+1))

            G_model.train()

        # 定めた保存周期ごとにロスを保存する
        if (epoch+1)%50 == 0:
            x = np.linspace(1, epoch+1, epoch+1, dtype='int')
            plot(result['G_log_loss'], result['D_log_loss'], result['E_log_loss'], result['CD_log_loss'], x, savedir)

    # 最後のエポックが保存周期でない場合に，保存する
    if (epoch+1)%10 != 0 and epoch+1 == epochs:
        torch.save(G_model.module.state_dict(), '{}/model/G_model_{}.pth'.format(savedir, epoch+1))
        torch.save(D_model.module.state_dict(), '{}/model/D_model_{}.pth'.format(savedir, epoch+1))
        torch.save(E_model.module.state_dict(), '{}/model/E_model_{}.pth'.format(savedir, epoch+1))
        torch.save(CD_model.module.state_dict(), '{}/model/CD_model_{}.pth'.format(savedir, epoch+1))

        G_model.eval()
        
        with torch.no_grad():
            rnd_img_test = G_model(z0)

        torchvision.utils.save_image(fake_img[:batch_size], "{}/generating_image/epoch_{:03}.png".format(savedir, epoch+1))
        torchvision.utils.save_image(rnd_img_test[:batch_size], "{}/generating_image_rnd/epoch_{:03}.png".format(savedir, epoch+1))

        x = np.linspace(1, epoch+1, epoch+1, dtype='int')
        plot(result['G_log_loss'], result['D_log_loss'], result['E_log_loss'], result['CD_log_loss'], x, savedir)
