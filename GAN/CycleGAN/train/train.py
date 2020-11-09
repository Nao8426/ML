# DCGAN(Deep Convolutional GAN)の学習
import numpy as np
import os
import pandas as pd
import statistics
import torch
import torchvision
from itertools import chain
from PIL import Image
from torch import nn
from tqdm import tqdm
# 自作モジュール
from load import LoadDataset
from network import Generator, Discriminator
from util import plot, output_env


# ジェネレータ，ディスクリミネータのロス
class MyLoss():
    def __init__(self):
        self.MSE_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()

    def G_loss(self, fake_pred_A, fake_pred_B, ones, img_A, img_B, rec_img_A, rec_img_B, iden_img_A, iden_img_B, alpha=10.0, beta=0):
        G_A2B_loss = self.MSE_loss(fake_pred_B, ones)
        G_B2A_loss = self.MSE_loss(fake_pred_A, ones)
        cycle_A_loss = self.L1_loss(img_A, rec_img_A)
        cycle_B_loss = self.L1_loss(img_B, rec_img_B)
        if beta == 0:
            return G_A2B_loss + G_B2A_loss + alpha*(cycle_A_loss + cycle_B_loss)
        else:
            iden_A_loss = self.L1_loss(img_A, iden_img_A)
            iden_B_loss = self.L1_loss(img_B, iden_img_B)
            return G_A2B_loss + G_B2A_loss + alpha*(cycle_A_loss + cycle_B_loss) + beta*(iden_A_loss + iden_B_loss)

    def D_A_loss(self, real_pred_A, ones, fake_pred_A, zeros):
        return self.MSE_loss(real_pred_A, ones) + self.MSE_loss(fake_pred_A, zeros)

    def D_B_loss(self, real_pred_B, ones, fake_pred_B, zeros):
        return self.MSE_loss(real_pred_B, ones) + self.MSE_loss(fake_pred_B, zeros)


# データセットに対する処理（正規化など）
class Trans():
    def __init__(self):
        self.norm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

    def __call__(self, image):
        return self.norm(image)


def train(savedir, train_list_A, train_list_B, test_list_A, test_list_B, root, epochs, batch_size):
    # 画像のチャンネル数
    channel = 1
    
    # LossにおけるCycle Lossの割合を決めるパラメータ（Cycle Lossにかかる係数）
    cycle_rate = 10.0
    
    # LossにおけるIdentity Lossの割合を決めるパラメータ（Identity Lossにかかる係数）
    iden_rate = 0

    # ジェネレータのAdam設定(default: lr=0.001, betas=(0.9, 0.999), weight_decay=0) 
    G_opt_para = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}
    # ディスクリミネータのAdam設定
    D_A_opt_para = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}
    D_B_opt_para = {'lr': 0.0002, 'betas': (0.5, 0.9), 'weight_decay': 0}

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

    G_A2B_model, G_B2A_model, D_A_model, D_B_model = Generator(channel), Generator(channel), Discriminator(channel), Discriminator(channel)
    G_A2B_model, G_B2A_model, D_A_model, D_B_model = nn.DataParallel(G_A2B_model), nn.DataParallel(G_B2A_model), nn.DataParallel(D_A_model), nn.DataParallel(D_B_model)
    G_A2B_model, G_B2A_model, D_A_model, D_B_model = G_A2B_model.to(device), G_B2A_model.to(device), D_A_model.to(device), D_B_model.to(device)

    # 最適化アルゴリズムの設定
    G_para = torch.optim.Adam(chain(G_A2B_model.parameters(), G_B2A_model.parameters()), lr=G_opt_para['lr'], betas=G_opt_para['betas'], weight_decay=G_opt_para['weight_decay'])
    D_A_para = torch.optim.Adam(D_A_model.parameters(), lr=D_A_opt_para['lr'], betas=D_A_opt_para['betas'], weight_decay=D_A_opt_para['weight_decay'])
    D_B_para = torch.optim.Adam(D_B_model.parameters(), lr=D_B_opt_para['lr'], betas=D_B_opt_para['betas'], weight_decay=D_B_opt_para['weight_decay'])

    # ロスの推移を保存するためのリストを確保
    result = {}
    result['G_log_loss'] = []
    result['D_A_log_loss'] = []
    result['D_B_log_loss'] = []

    df_A = pd.read_csv(train_list_A, usecols=['Path'])
    df_B = pd.read_csv(train_list_B, usecols=['Path'])
    df_test_A = pd.read_csv(test_list_A, usecols=['Path'])
    df_test_A = df_test_A.sample(frac=1)
    df_test_B = pd.read_csv(test_list_B, usecols=['Path'])
    df_test_B = df_test_B.sample(frac=1)

    train_dataset = LoadDataset(df_A, df_B, root, transform=Trans())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = LoadDataset(df_test_A[0:batch_size], df_test_B[0:batch_size], root, transform=Trans())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    output_env('{}/env.txt'.format(savedir), batch_size, G_opt_para, D_A_opt_para, D_B_opt_para, G_A2B_model, G_B2A_model, D_A_model, D_B_model)

    for epoch in range(epochs):
        print('########## epoch : {}/{} ##########'.format(epoch+1, epochs))

        G_log_loss, D_A_log_loss, D_B_log_loss = [], [], []

        for img_A, img_B in tqdm(train_loader):
            # GPU用の変数に変換
            img_A = img_A.to(device)
            img_B = img_B.to(device)

            # 真正画像をジェネレータに入力
            fake_img_A = G_B2A_model(img_B)
            fake_img_B = G_A2B_model(img_A)

            # 生成画像をジェネレータに入力
            rec_img_A = G_B2A_model(fake_img_B)
            rec_img_B = G_A2B_model(fake_img_A)

            # ディスクリミネータに真正画像と生成画像を入力
            real_pred_A = D_A_model(img_A)
            real_pred_B = D_B_model(img_B)
            fake_pred_A = D_A_model(fake_img_A)
            fake_pred_B = D_B_model(fake_img_B)
            
            # ジェネレータに出力画像と同一ドメインの画像を入力（恒等写像）
            if iden_img == 0:
                iden_img_A = None
                iden_img_B = None
            else:
                iden_img_A = G_B2A_model(img_A)
                iden_img_B = G_A2B_model(img_B)

            # ジェネレータのロス計算
            G_loss = myloss.G_loss(fake_pred_A, fake_pred_B, torch.tensor(1.0).expand_as(fake_pred_A).to(device), img_A, img_B, rec_img_A, rec_img_B, iden_img_A, iden_img_B, alpha=cycle_rate, beta=iden_rate)
            G_log_loss.append(G_loss.item())
            # ディスクリミネータのロス計算
            D_A_loss = myloss.D_A_loss(real_pred_A, torch.tensor(1.0).expand_as(real_pred_A).to(device), fake_pred_A, torch.tensor(0.0).expand_as(fake_pred_A).to(device))
            D_A_log_loss.append(D_A_loss.item())
            D_B_loss = myloss.D_B_loss(real_pred_B, torch.tensor(1.0).expand_as(real_pred_B).to(device), fake_pred_B, torch.tensor(0.0).expand_as(fake_pred_B).to(device))
            D_B_log_loss.append(D_B_loss.item())

            # ジェネレータの重み更新
            G_para.zero_grad()
            G_loss.backward(retain_graph=True)
            G_para.step()
            # ディスクリミネータの重み更新
            D_A_para.zero_grad()
            D_A_loss.backward(retain_graph=True)
            D_A_para.step()
            D_B_para.zero_grad()
            D_B_loss.backward()
            D_B_para.step()

        result['G_log_loss'].append(statistics.mean(G_log_loss))
        result['D_A_log_loss'].append(statistics.mean(D_A_log_loss))
        result['D_B_log_loss'].append(statistics.mean(D_B_log_loss))
        print('G_loss = {} , D_A_loss = {} , D_B_loss = {}'.format(result['G_log_loss'][-1], result['D_A_log_loss'][-1], result['D_B_log_loss'][-1]))

        # ロスのログを保存
        with open('{}/loss/log.txt'.format(savedir), mode='a') as f:
            f.write('##### Epoch {:03} #####\n'.format(epoch+1))
            f.write('G: {}, D_A: {}, D_B: {}\n'.format(result['G_log_loss'][-1], result['D_A_log_loss'][-1], result['D_B_log_loss'][-1]))
        
        # 定めた保存周期ごとにモデル，出力画像を保存する
        if (epoch+1)%10 == 0:
            # モデルの保存
            torch.save(G_A2B_model.module.state_dict(), '{}/model/G_A2B_model_{}.pth'.format(savedir, epoch+1))
            torch.save(G_B2A_model.module.state_dict(), '{}/model/G_B2A_model_{}.pth'.format(savedir, epoch+1))
            torch.save(D_A_model.module.state_dict(), '{}/model/D_A_model_{}.pth'.format(savedir, epoch+1))
            torch.save(D_B_model.module.state_dict(), '{}/model/D_B_model_{}.pth'.format(savedir, epoch+1))

            G_A2B_model.eval()
            G_B2A_model.eval()

            # メモリ節約のためパラメータの保存は止める（テスト時にパラメータの保存は不要）
            with torch.no_grad():
                for test_img_A, test_img_B in test_loader:
                    fake_img_test_A = G_B2A_model(test_img_B)
                    fake_img_test_B = G_A2B_model(test_img_A)
            torchvision.utils.save_image(fake_img_test_A[:batch_size], "{}/generating_image/A_epoch_{:03}.png".format(savedir, epoch+1))
            torchvision.utils.save_image(fake_img_test_B[:batch_size], "{}/generating_image/B_epoch_{:03}.png".format(savedir, epoch+1))

            G_A2B_model.train()
            G_B2A_model.train()

        # 定めた保存周期ごとにロスを保存する
        if (epoch+1)%50 == 0:
            x = np.linspace(1, epoch+1, epoch+1, dtype='int')
            plot(result['G_log_loss'], result['D_A_log_loss'], result['D_B_log_loss'], x, savedir)

    # 最後のエポックが保存周期でない場合に，保存する
    if (epoch+1)%10 != 0 and epoch+1 == epochs:
        torch.save(G_A2B_model.module.state_dict(), '{}/model/G_A2B_model_{}.pth'.format(savedir, epoch+1))
        torch.save(G_B2A_model.module.state_dict(), '{}/model/G_B2A_model_{}.pth'.format(savedir, epoch+1))
        torch.save(D_A_model.module.state_dict(), '{}/model/D_A_model_{}.pth'.format(savedir, epoch+1))
        torch.save(D_B_model.module.state_dict(), '{}/model/D_B_model_{}.pth'.format(savedir, epoch+1))

        G_A2B_model.eval()
        G_B2A_model.eval()

        # メモリ節約のためパラメータの保存は止める（テスト時にパラメータの保存は不要）
        with torch.no_grad():
            for test_img_A, test_img_B in test_loader:
                fake_img_test_A = G_B2A_model(test_img_B)
                fake_img_test_B = G_A2B_model(test_img_A)
        torchvision.utils.save_image(fake_img_test_A[:batch_size], "{}/generating_image/A_epoch_{:03}.png".format(savedir, epoch+1))
        torchvision.utils.save_image(fake_img_test_B[:batch_size], "{}/generating_image/B_epoch_{:03}.png".format(savedir, epoch+1))

        G_A2B_model.train()
        G_B2A_model.train()

        x = np.linspace(1, epoch+1, epoch+1, dtype='int')
        plot(result['G_log_loss'], result['D_A_log_loss'], result['D_B_log_loss'], x, savedir)
