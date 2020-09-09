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
from load import LoadDataset
from network import Generator, Discriminator
from util import plot, output_env


# ジェネレータ，ディスクリミネータのロス
class MyLoss():
    def __init__(self):
        self.loss_BCE = nn.BCEWithLogitsLoss()

    def gen_loss(self, x, y):
        return self.loss_BCE(x, y)

    def dis_loss(self, p, q, r, s):
        return self.loss_BCE(p, q) + self.loss_BCE(r, s)


# データセットに対する処理（正規化など）
class Trans():
    def __init__(self):
        self.norm = torchvision.transforms.ToTensor()

    def __call__(self, image):
        return self.norm(image)


def train(savedir, _list, root, epochs, batch_size, nz):
    # ジェネレータのAdam設定(default: lr=0.001, betas=(0.9, 0.999), weight_decay=0) 
    opt_para_G = {'lr': 0.0005, 'betas': (0.0, 0.99), 'weight_decay': 0}
    # ディスクリミネータのAdam設定
    opt_para_D = {'lr': 0.0005, 'betas': (0.0, 0.99), 'weight_decay': 0}

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

    device = 'cuda'

    myloss = MyLoss()

    df = pd.read_csv(_list, usecols=['Path'])
    img_id = df.values.tolist()

    check_img = Image.open('{}/{}'.format(root, img_id[0][0]))
    check_img = check_img.convert('L')
    check_img = np.array(check_img)
    height, width = check_img.shape

    gen_model, dis_model = Generator(nz, width, height, 1), Discriminator(width, height, 1)
    gen_model, dis_model = nn.DataParallel(gen_model), nn.DataParallel(dis_model)
    gen_model, dis_model = gen_model.to(device), dis_model.to(device)

    # 最適化アルゴリズムの設定
    gen_para = torch.optim.Adam(gen_model.parameters(), lr=opt_para_G['lr'], betas=opt_para_G['betas'], weight_decay=opt_para_G['weight_decay'])
    dis_para = torch.optim.Adam(dis_model.parameters(), lr=opt_para_D['lr'], betas=opt_para_D['betas'], weight_decay=opt_para_D['weight_decay'])

    # ロスを計算するためのラベル変数
    ones = torch.ones(512).to(device)
    zeros = torch.zeros(512).to(device)

    # ロスの推移
    result = {}
    result['log_loss_G'] = []
    result['log_loss_D'] = []

    imgs = LoadDataset(df, root, transform=Trans())
    train_img = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True)

    output_env('{}/env.txt'.format(savedir), batch_size, nz, opt_para_G, opt_para_D, gen_model, dis_model)

    for epoch in range(epochs):
        print('########## epoch : {}/{} ##########'.format(epoch+1, epochs))

        log_loss_G, log_loss_D = [], []

        for real_img in tqdm(train_img):
            # 入力の乱数を作成
            z = torch.randn(batch_size, nz, 1, 1)

            # GPU用の変数に変換
            real_img = real_img.to(device)
            z = z.to(device)

            # ジェネレータに入力
            fake_img = gen_model(z, epoch+1, alpha)
            # ジェネレータの出力を保存
            fake_img_tensor = fake_img.detach()

            # 生成画像をディスクリミネータに入力し，判定結果を取得
            out = dis_model(fake_img)

            # 真正画像を真，生成画像を偽と識別できるようにロスを計算
            real_out = dis_model(real_img)
            fake_out = dis_model(fake_img_tensor)

            # ジェネレータのロス計算
            loss_G = myloss.gen_loss(out, ones[:batch_size])
            log_loss_G.append(loss_G.item())
            # ディスクリミネータのロス計算
            loss_D = myloss.dis_loss(real_out, ones[:batch_size], fake_out, zeros[:batch_size])
            log_loss_D.append(loss_D.item())

            # ジェネレータの重み更新
            gen_para.zero_grad()
            loss_G.backward()
            gen_para.step()
            # ディスクリミネータの重み更新
            dis_para.zero_grad()
            loss_D.backward()
            dis_para.step()

        result['log_loss_G'].append(statistics.mean(log_loss_G))
        result['log_loss_D'].append(statistics.mean(log_loss_D))
        print('loss_G = {} , loss_D = {}'.format(result['log_loss_G'][-1], result['log_loss_D'][-1]))

        # ロスのログを保存
        with open('{}/loss/log.txt'.format(savedir), mode='a') as f:
            f.write('##### Epoch {:03} #####\n'.format(epoch+1))
            f.write('G: {}, D: {}\n'.format(epoch+1, result['log_loss_G'][-1], result['log_loss_D'][-1]))
        
        # 定めた保存周期ごとにモデル，出力画像を保存する
        if (epoch+1)%10 == 0:
            # モデルの保存
            torch.save(gen_model.module.state_dict(), '{}/model/G_model_{}.pth'.format(savedir, epoch+1))
            torch.save(dis_model.module.state_dict(), '{}/model/D_model_{}.pth'.format(savedir, epoch+1))

            # ジェネレータの出力画像を保存
            torchvision.utils.save_image(fake_img_tensor[:batch_size], "{}/generating_image/epoch_{:03}.png".format(savedir, epoch+1))

        # 定めた保存周期ごとにロスを保存する
        if (epoch+1)%50 == 0:
            x = np.linspace(1, epoch+1, epoch+1, dtype='int')
            plot(result['log_loss_G'], result['log_loss_D'], x, savedir)

    # 最後のエポックが保存周期でない場合に，保存する
    if (epoch+1)%10 != 0 and epoch+1 == epochs:
        torch.save(gen_model.module.state_dict(), '{}/model/G_model_{}.pth'.format(savedir, epoch+1))
        torch.save(dis_model.module.state_dict(), '{}/model/D_model_{}.pth'.format(savedir, epoch+1))

        torchvision.utils.save_image(fake_img_tensor[:batch_size], "{}/generating_image/epoch_{:03}.png".format(savedir, epoch+1))

        x = np.linspace(1, epoch+1, epoch+1, dtype='int')
        plot(result['log_loss_G'], result['log_loss_D'], x, savedir)
