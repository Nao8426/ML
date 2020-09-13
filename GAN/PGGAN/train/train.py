import numpy as np
import os
import pandas as pd
import statistics
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
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

    channel = 1

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

    gen_model, dis_model = Generator(channel), Discriminator(channel)
    gen_model, dis_model = nn.DataParallel(gen_model), nn.DataParallel(dis_model)
    gen_model, dis_model = gen_model.to(device), dis_model.to(device)
    gen_model_mavg = Generator(channel)
    gen_model_mavg = nn.DataParallel(gen_model_mavg)
    gen_model_mavg = gen_model_mavg.to(device)

    # 最適化アルゴリズムの設定
    gen_para = torch.optim.Adam(gen_model.parameters(), lr=opt_para_G['lr'], betas=opt_para_G['betas'], weight_decay=opt_para_G['weight_decay'])
    dis_para = torch.optim.Adam(dis_model.parameters(), lr=opt_para_D['lr'], betas=opt_para_D['betas'], weight_decay=opt_para_D['weight_decay'])

    # ロスを計算するためのラベル変数
    ones = torch.ones(512).to(device)
    zeros = torch.zeros(512).to(device)

    df = pd.read_csv(_list, usecols=['Path'])
    dataset = LoadDataset(df, root, transform=Trans())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ロスの推移
    result = {}
    result['log_loss_G'], result['log_loss_D'] = [], []

    n_ite = len(df) // batch_size
    res_step = n_ite * 30
    j = 0

    output_env('{}/env.txt'.format(savedir), batch_size, nz, opt_para_G, opt_para_D, gen_model, dis_model)

    # テスト用の一定乱数(第一引数で出力枚数指定)
    z0 = torch.randn(64, nz*16).to(device)
    z0 = torch.clamp(z0, -1.,1.)

    for epoch in range(epochs):
        print('########## epoch : {}/{} ##########'.format(epoch+1, epochs))

        if j==res_step*6.5:
            gen_para.param_groups[0]['lr'] = 0.0001
            dis_para.param_groups[0]['lr'] = 0.0001

        log_loss_G, log_loss_D = [], []

        for i, real_img in enumerate(tqdm(train_loader)):
            real_img = real_img.to(device)
            res = j/res_step

            ##### ジェネレータのトレーニング #####
            z = torch.randn(batch_size, nz*16).to(device)
            fake_img = gen_model.forward(z, res)
            fake_out = dis_model.forward(fake_img, res)

            # ジェネレータのロス計算
            loss_G = myloss.gen_loss(fake_out, ones[:batch_size])
            log_loss_G.append(loss_G.item())

            gen_para.zero_grad()
            loss_G.backward()
            gen_para.step()

            # update gen_model_mavg by moving average
            momentum = 0.995 # remain momentum
            alpha = min(1.0-(1/(j+1)), momentum)
            for p_mavg, p in zip(gen_model_mavg.parameters(), gen_model.parameters()):
                p_mavg.data = alpha*p_mavg.data + (1.0-alpha)*p.data

            ##### ディスクリミネータのトレーニング #####
            z = torch.randn(real_img.shape[0], nz*16).to(device)
            fake_img = gen_model.forward(z, res)
            real_img = F.adaptive_avg_pool2d(real_img, fake_img.shape[2:4])
            real_out = dis_model.forward(real_img, res)
            fake_out = dis_model.forward(fake_img, res)

            # ディスクリミネータのロス計算
            loss_D = myloss.dis_loss(real_out, ones[:batch_size], fake_out, zeros[:batch_size])
            log_loss_D.append(loss_D.item())

            dis_para.zero_grad()
            loss_D.backward()
            dis_para.step()

            j += 1

            if j >= res_step*7:
                break

        result['log_loss_G'].append(statistics.mean(log_loss_G))
        result['log_loss_D'].append(statistics.mean(log_loss_D))
        print('loss_G = {} , loss_D = {}'.format(result['log_loss_G'][-1], result['log_loss_D'][-1]))

        # ロスのログを保存
        with open('{}/loss/log.txt'.format(savedir), mode='a') as f:
            f.write('##### Epoch {:03} #####\n'.format(epoch+1))
            f.write('G: {}, D: {}\n'.format(result['log_loss_G'][-1], result['log_loss_D'][-1]))

        # 定めた保存周期ごとにモデル，出力画像を保存する
        if (epoch+1)%10 == 0:
            # モデルの保存
            torch.save(gen_model.module.state_dict(), '{}/model/G_model_{}.pth'.format(savedir, epoch+1))
            torch.save(dis_model.module.state_dict(), '{}/model/D_model_{}.pth'.format(savedir, epoch+1))

            gen_model_mavg.eval()
            fake_img_test = gen_model_mavg.forward(z0, res)

            # 解像度が異なるため，一定サイズに補間して出力
            dst = F.interpolate(fake_img_test, (224, 224), mode='nearest')
            dst = dst.detach()
            torchvision.utils.save_image(dst, "{}/generating_image/img_{:03}_{:05}.png".format(savedir, epoch+1, j), nrow=8)

            gen_model_mavg.train()

        # 定めた保存周期ごとにロスを保存する
        if (epoch+1)%50 == 0:
            x = np.linspace(1, epoch+1, epoch+1, dtype='int')
            plot(result['log_loss_G'], result['log_loss_D'], x, savedir)

        if j >= res_step*7:
            break
