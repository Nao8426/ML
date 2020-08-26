import argparse
import os
# 自作モジュール
from generate import generate
os.chdir(os.path.dirname(os.path.abspath(__file__)))


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
