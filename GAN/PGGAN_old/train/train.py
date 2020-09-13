import numpy as np
import os
import pandas as pd
# import statistics
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
# 自作モジュール
from load import LoadDataset
from network import Generator, Discriminator


def gradient_penalty(dis_model, real, fake, res, batch_size, gamma=1):
    device = real.device
    alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True).to(device)
    x = alpha*real + (1-alpha)*fake
    d_ = dis_model.forward(x, res)
    g = torch.autograd.grad(outputs=d_, inputs=x, grad_outputs=torch.ones(d_.shape).to(device), create_graph=True, retain_graph=True,only_inputs=True)[0]
    g = g.reshape(batch_size, -1)
    return ((g.norm(2,dim=1)/gamma-1.0)**2).mean()


# データセットに対する処理（正規化など）
class Trans():
    def __init__(self):
        self.norm = torchvision.transforms.ToTensor()

    def __call__(self, image):
        return self.norm(image)


def train(savedir, _list, root, epochs, batch_size, nz):
    # ジェネレータのAdam設定(default: lr=0.001, betas=(0.9, 0.999), weight_decay=0) 
    para_G = {'lr': 0.0005, 'betas': (0.0, 0.99), 'weight_decay': 0}
    # ディスクリミネータのAdam設定
    para_D = {'lr': 0.0005, 'betas': (0.0, 0.99), 'weight_decay': 0}

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

    gen_model, dis_model = Generator(), Discriminator()
    gen_model, dis_model = nn.DataParallel(gen_model), nn.DataParallel(dis_model)
    gen_model, dis_model = gen_model.to(device), dis_model.to(device)
    gen_model_mavg = Generator()
    gen_model_mavg = nn.DataParallel(gen_model_mavg)
    gen_model_mavg = gen_model_mavg.to(device)

    # 最適化アルゴリズムの設定
    gen_para = torch.optim.Adam(gen_model.parameters(), lr=para_G['lr'], betas=para_G['betas'], weight_decay=para_G['weight_decay'])
    dis_para = torch.optim.Adam(dis_model.parameters(), lr=para_D['lr'], betas=para_D['betas'], weight_decay=para_D['weight_decay'])

    criterion = torch.nn.BCELoss()

    df = pd.read_csv(_list, usecols=['Path'])
    dataset = LoadDataset(df, root, transform=Trans())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # training
    epochs = 10

    # ロスの推移
    result = {}
    result['log_loss_G'], result['log_loss_D'] = [], []

    res_step = 15000
    j = 0
    # constant random inputs
    z0 = torch.randn(16, 512*16).to(device)
    z0 = torch.clamp(z0, -1.,1.)
    for epoch in range(epochs):
        if j==res_step*6.5:
            gen_para.param_groups[0]['lr'] = 0.0001
            dis_para.param_groups[0]['lr'] = 0.0001

        for i, real_img in enumerate(train_loader):
            real_img = real_img.to(device)
            res = j/res_step

            ### train generator ###
            z = torch.randn(batch_size, 512*16).to(device)
            x_ = gen_model.forward(z, res)
            d_ = dis_model.forward(x_, res) # fake
            lossG = -d_.mean() # WGAN_GP

            gen_para.zero_grad()
            lossG.backward()
            gen_para.step()

            # update gen_model_mavg by moving average
            momentum = 0.995 # remain momentum
            alpha = min(1.0-(1/(j+1)), momentum)
            for p_mavg, p in zip(gen_model_mavg.parameters(), gen_model.parameters()):
                p_mavg.data = alpha*p_mavg.data + (1.0-alpha)*p.data

            ### train discriminator ###
            z = torch.randn(real_img.shape[0], 512*16).to(device)
            x_ = gen_model.forward(z, res)
            real_img = F.adaptive_avg_pool2d(real_img, x_.shape[2:4])
            d = dis_model.forward(real_img, res)   # real
            d_ = dis_model.forward(x_, res) # fake
            loss_real = -d.mean()
            loss_fake = d_.mean()
            loss_gp = gradient_penalty(dis_model, real_img.data, x_.data, res, real_img.shape[0])
            loss_drift = (d**2).mean()

            beta_gp = 10.0
            beta_drift = 0.001
            lossD = loss_real + loss_fake + beta_gp*loss_gp + beta_drift*loss_drift

            dis_para.zero_grad()
            lossD.backward()
            dis_para.step()

            print('ep: {:02} {:04} {:04} lossG={} lossD={}'.format(epoch, i, j, lossG.item(), lossD.item()))

            result['log_loss_G'].append(lossG.item())
            result['log_loss_D'].append(lossD.item())
            j += 1

            if j%500 == 0:
                gen_model_mavg.eval()
                z = torch.randn(16, 512*16).to(device)
                x_0 = gen_model_mavg.forward(z0, res)
                x_ = gen_model_mavg.forward(z, res)

                dst = torch.cat((x_0, x_), dim=0)
                dst = F.interpolate(dst, (128, 256), mode='nearest')
                dst = dst.detach()
                torchvision.utils.save_image(dst, "{}/generating_image/img_{:03}_{:05}.png".format(savedir, epoch+1, j))

                # losses_ = np.array(losses)
                # niter = losses_.shape[0]//100*100
                # x_iter = np.arange(100)*(niter//100) + niter//200
                # plt.plot(x_iter, losses_[:niter,0].reshape(100,-1).mean(1))
                # plt.plot(x_iter, losses_[:niter,1].reshape(100,-1).mean(1))
                # plt.tight_layout()
                # plt.savefig('out/loss_%03d_%05d.png' % (epoch, j))
                # plt.clf()

                gen_model_mavg.train()

            if j >= res_step*7:
                break

        if j >= res_step*7:
            break
