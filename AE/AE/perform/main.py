# 学習済みの保存されたモデルを使用
import argparse
import os
# 自作モジュール
from generate import generate
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# 使用データセットのリストのパス
_list = '../imagelist/test.csv'
root = '../dataset' # データセットまでのパス

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description='Generate by Auto-Encoder')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# コマンドライン引数により指定されたパラメータを変数に格納
GPU_ID = args.gpu
GENE_NUM = args.num

# 設定内容を表示
print('GPU: {}'.format(GPU_ID))

# 使用するGPUの指定
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)

generate('./generate_imgs/tmp', './progress/tmp/model/tmp.pth', _list, root)
