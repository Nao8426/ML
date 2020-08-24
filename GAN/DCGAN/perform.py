import argparse
import os
import torch
import torchvision
from torch import nn
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ジェネレータの構造
class Generator(nn.Module):
    def __init__(self, nz, width, height, channel):
        self.L1_C = 256
        self.L2_C = 128
        self.L3_C = 64
        self.L4_C = 32

        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=self.L1_C, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.L1_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L1_C, out_channels=self.L2_C, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.L2_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L2_C, out_channels=self.L3_C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.L3_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L3_C, out_channels=self.L4_C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.L4_C),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.L4_C, out_channels=channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)


def generate(savedir, model_path, num):
    nz = 32
    width = 72
    height = 72
    channel = 1

    device = 'cuda'

    model = Generator(nz, width, height, channel)
    model = nn.DataParallel(model)

    model.module.load_state_dict(torch.load(model_path))

    model.eval()    # 推論モードへ切り替え（Dropoutなどの挙動に影響）

    # 保存先のファイルを作成
    if os.path.exists(savedir):
        n = 1
        while 1:
            if os.path.exists('{}({})'.format(savedir, n)):
                n += 1
            else:
                savedir = '{}({})'.format(savedir, n)
                break
    os.makedirs(savedir, exist_ok=True)

    for i in range(num):
        # 入力の乱数を作成
        z = torch.randn(1, nz, 1, 1)
        z = z.to(device)

        gene_img = model(z)

        # ジェネレータの出力画像を保存
        torchvision.utils.save_image(gene_img, "{}/{:04}.png".format(savedir, i+1))


if __name__ == '__main__':
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='Generator of DCGAN')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--num', '-n', default=100, type=int, help='number of generating images')
    args = parser.parse_args()

    # コマンドライン引数により指定されたパラメータを変数に格納
    GPU_ID = args.gpu
    GENE_NUM = args.num

    # 設定内容を表示
    print('GPU: {}'.format(GPU_ID))
    print('Num. of generating images : {}'.format(GENE_NUM))

    # 使用するGPUの指定
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)

    generate('./generate_imgs/iroha', './progress/iroha/model/G_model_400.pth', GENE_NUM)
