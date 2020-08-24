# DCGAN(Deep Convolutional GAN)の学習
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
from network import Generator, Discriminator, Encoder, CodeDiscriminator
from util import plot, output_env


# 各モデルのロス
class MyLoss():
    def __init__(self):
        self.loss_BCE = nn.BCEWithLogitsLoss()
        self.loss_L1 = nn.L1Loss()

    def gen_loss(self, x, y, p, q, r, alpha=1.0):
        return self.loss_L1(x, y) + alpha*(self.loss_BCE(p, r)/2 + self.loss_BCE(q, r)/2)

    def dis_loss(self, x, y, p, q, r):
        return self.loss_BCE(x, y) + self.loss_BCE(p, r)/2 + self.loss_BCE(q, r)/2

    def enc_loss(self, x, y, p, q, alpha=1.0):
        return self.loss_L1(x, y) + alpha*self.loss_BCE(p, q)

    def cdis_loss(self, x, y, p, q):
        return self.loss_BCE(x, y) + self.loss_BCE(p, q)


def train(savedir, _list, root, epochs, batch_size, nz):
    # ジェネレータのAdam設定(default: lr=0.001, betas=(0.9, 0.999), weight_decay=0) 
    para_G = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}
    # ディスクリミネータのAdam設定
    para_D = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}
    # エンコーダのAdam設定
    para_E = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}
    # コードディスクリミネータのAdam設定
    para_CD = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}

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
    os.makedirs('{}/logs'.format(savedir), exist_ok=True)

    device = 'cuda'

    myloss = MyLoss()

    df = pd.read_csv(_list, usecols=['Path'])
    img_id = df.values.tolist()

    check_img = Image.open('{}/{}'.format(root, img_id[0][0]))
    check_img = check_img.convert('L')
    check_img = np.array(check_img)
    height, width = check_img.shape

    gen_model, dis_model, enc_model, cdis_model = Generator(nz, width, height, 1), Discriminator(width, height, 1), Encoder(nz, width, height, 1), CodeDiscriminator(nz)
    gen_model, dis_model, enc_model, cdis_model = nn.DataParallel(gen_model), nn.DataParallel(dis_model), nn.DataParallel(enc_model), nn.DataParallel(cdis_model)
    gen_model, dis_model, enc_model, cdis_model = gen_model.to(device), dis_model.to(device), enc_model.to(device), cdis_model.to(device)

    # 最適化アルゴリズムの設定
    gen_para = torch.optim.Adam(gen_model.parameters(), lr=para_G['lr'], betas=para_G['betas'], weight_decay=para_G['weight_decay'])
    dis_para = torch.optim.Adam(dis_model.parameters(), lr=para_D['lr'], betas=para_D['betas'], weight_decay=para_D['weight_decay'])
    enc_para = torch.optim.Adam(enc_model.parameters(), lr=para_E['lr'], betas=para_E['betas'], weight_decay=para_E['weight_decay'])
    cdis_para = torch.optim.Adam(cdis_model.parameters(), lr=para_CD['lr'], betas=para_CD['betas'], weight_decay=para_CD['weight_decay'])

    # ロスを計算するためのラベル変数
    ones = torch.ones(512).to(device)
    zeros = torch.zeros(512).to(device)

    # ロスの推移
    result = {}
    result['log_loss_G'] = []
    result['log_loss_D'] = []
    result['log_loss_E'] = []
    result['log_loss_CD'] = []

    imgs = LoadDataset(df, root)
    train_img = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True)

    output_env('{}/env.txt'.format(savedir), batch_size, nz, para_G, para_D, para_E, para_CD, gen_model, dis_model, enc_model, cdis_model)

    for epoch in range(epochs):
        print('########## epoch : {}/{} ##########'.format(epoch+1, epochs))

        log_loss_G, log_loss_D, log_loss_E, log_loss_CD = [], [], [], []

        for real_img in tqdm(train_img):
            # 入力の乱数を作成
            rnd_z = torch.randn(batch_size, nz, 1, 1)

            # 画像の輝度値を正規化
            real_img = real_img.float()
            real_img = real_img / 255
            # 3次元テンソルを4次元テンソルに変換（1チャネルの情報を追加）
            batch, height, width = real_img.shape
            real_img = torch.reshape(real_img, (batch_size, 1, height, width))

            # GPU用の変数に変換
            real_img = real_img.to(device)
            rnd_z = rnd_z.to(device)

            # 真正画像をエンコーダに入力し，特徴ベクトルを取得
            real_z = enc_model(real_img)

            # 特徴ベクトルをジェネレータに入力し，画像を生成
            fake_img = gen_model(real_z)
            rnd_img = gen_model(rnd_z)
            # ジェネレータの出力を保存
            fake_img_tensor = fake_img.detach()
            rnd_img_tensor = rnd_img.detach()

            # 特徴ベクトルをコードディスクリミネータに入力し，判定結果を取得
            real_cy = cdis_model(real_z)
            rnd_cy = cdis_model(rnd_z)

            # 真正画像および生成画像をディスクリミネータに入力し，判定結果を取得
            real_y = dis_model(real_img)
            fake_y = dis_model(fake_img)
            rnd_y = dis_model(rnd_img)

            # エンコーダのロス計算
            loss_E = myloss.enc_loss(real_img, fake_img, real_cy, ones[:batch_size], 1.0)
            log_loss_E.append(loss_E.item())
            # ジェネレータのロス計算
            loss_G = myloss.gen_loss(real_img, fake_img, fake_y, rnd_y, ones[:batch_size], 1.0)
            log_loss_G.append(loss_G.item())
            # コードディスクリミネータのロス計算
            loss_CD = myloss.cdis_loss(real_cy, zeros[:batch_size], rnd_cy, ones[:batch_size])
            log_loss_CD.append(loss_CD.item())
            # ディスクリミネータのロス計算
            loss_D = myloss.gen_loss(real_y, ones[:batch_size], fake_y, rnd_y, zeros[:batch_size])
            log_loss_D.append(loss_D.item())

            # エンコーダの重み更新
            enc_para.zero_grad()
            loss_E.backward(retain_graph=True)
            enc_para.step()
            # ジェネレータの重み更新
            gen_para.zero_grad()
            loss_G.backward(retain_graph=True)
            gen_para.step()
            # コードディスクリミネータの重み更新
            cdis_para.zero_grad()
            loss_CD.backward(retain_graph=True)
            cdis_para.step()
            # ディスクリミネータの重み更新
            dis_para.zero_grad()
            loss_D.backward()
            dis_para.step()

        result['log_loss_G'].append(statistics.mean(log_loss_G))
        result['log_loss_D'].append(statistics.mean(log_loss_D))
        result['log_loss_E'].append(statistics.mean(log_loss_E))
        result['log_loss_CD'].append(statistics.mean(log_loss_CD))
        print('loss_G = {} , loss_D = {} , loss_E = {} , loss_CD = {}'.format(result['log_loss_G'][-1], result['log_loss_D'][-1], result['log_loss_E'][-1], result['log_loss_CD'][-1]))
        
        # 定めた保存周期ごとにモデル，出力画像，ログを保存する
        if (epoch+1)%10 == 0:
            # モデルの保存
            torch.save(gen_model.module.state_dict(), '{}/model/G_model_{}.pth'.format(savedir, epoch+1))
            torch.save(dis_model.module.state_dict(), '{}/model/D_model_{}.pth'.format(savedir, epoch+1))
            torch.save(enc_model.module.state_dict(), '{}/model/E_model_{}.pth'.format(savedir, epoch+1))
            torch.save(cdis_model.module.state_dict(), '{}/model/CD_model_{}.pth'.format(savedir, epoch+1))

            # ジェネレータの出力画像を保存
            torchvision.utils.save_image(fake_img_tensor[:batch_size], "{}/generating_image/epoch_{:03}.png".format(savedir, epoch+1))
            torchvision.utils.save_image(rnd_img_tensor[:batch_size], "{}/generating_image_rnd/epoch_{:03}.png".format(savedir, epoch+1))
    
            # ログの保存
            with open('{}/logs/logs_{}.pkl'.format(savedir, epoch+1), 'wb') as fp:
                pickle.dump(result, fp)

        # 定めた保存周期ごとにロスを保存する
        if (epoch+1)%50 == 0:
            x = np.linspace(1, epoch+1, epoch+1, dtype='int')
            plot(result['log_loss_G'], result['log_loss_D'], result['log_loss_E'], result['log_loss_CD'], x, savedir)

    # 最後のエポックが保存周期でない場合に，保存する
    if (epoch+1)%10 != 0 and epoch+1 == epochs:
        torch.save(gen_model.module.state_dict(), '{}/model/G_model_{}.pth'.format(savedir, epoch+1))
        torch.save(dis_model.module.state_dict(), '{}/model/D_model_{}.pth'.format(savedir, epoch+1))
        torch.save(enc_model.module.state_dict(), '{}/model/E_model_{}.pth'.format(savedir, epoch+1))
        torch.save(cdis_model.module.state_dict(), '{}/model/CD_model_{}.pth'.format(savedir, epoch+1))

        x = np.linspace(1, epoch+1, epoch+1, dtype='int')
        plot(result['log_loss_G'], result['log_loss_D'], result['log_loss_E'], result['log_loss_CD'], x, savedir)
        
        with open('{}/logs/logs_{}.pkl'.format(savedir, epoch+1), 'wb') as fp:
            pickle.dump(result, fp)
