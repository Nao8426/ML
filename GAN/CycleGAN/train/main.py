# CycleGANの学習の実行
import argparse
import os
# 自作モジュール
from train import train
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# savedir: 保存先のディレクトリのパス
# list_A: 使用データセットのパスのリストのパス
# list_B: 使用データセットのパスのリストのパス
# root: データセットまでのパス（リスト内のパスの前に付く部分）
savedir = '../progress/tmp'
train_list_A = '../imagelist/train_A.csv'
train_list_B = '../imagelist/train_B.csv'
test_list_A = '../imagelist/test_A.csv'
test_list_B = '../imagelist/test_B.csv'
root = '../dataset'

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description='CycleGAN')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epochs', '-e', default=1000, type=int, help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', default=64, type=int, help='learning minibatch size')
args = parser.parse_args()

# コマンドライン引数により指定されたパラメータを変数に格納
GPU_ID = args.gpu
EPOCHS = args.epochs
BATCH_SIZE = args.batchsize

# 設定内容を表示
print('GPU: {}'.format(GPU_ID))
print('Num. of epochs: {}'.format(EPOCHS))
print('Minibatch size: {}'.format(BATCH_SIZE))

# 使用するGPUの指定
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)

train(savedir, train_list_A, train_list_B, test_list_A, test_list_B, root, EPOCHS, BATCH_SIZE)
