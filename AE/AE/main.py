# VAEの学習の実行
import argparse
import os
# 自作モジュール
from train import train
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# 使用データセットのリストのパス
# _list = '../imagelist/CASIA_cut.csv'
_list = '../imagelist/HIT-MW_cut.csv'
# _list = '../imagelist/IAM_cut.csv'
# _list = '../imagelist/IAM_Online_cut.csv'
root = '../dataset' # データセットまでのパス

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description='Variational Auto Encoder')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epochs', '-e', default=1000, type=int, help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', default=32, type=int, help='learning minibatch size')
parser.add_argument('--start', '-s', default=0, type=int, help='start of imagelist')
parser.add_argument('--finish', '-f', default=-1, type=int, help='end of imagelist')
args = parser.parse_args()

# コマンドライン引数により指定されたパラメータを変数に格納
GPU_ID = args.gpu
EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
START = args.start
FINISH = args.finish

# 設定内容を表示
print('GPU: {}'.format(GPU_ID))
print('Num. of epochs : {}'.format(EPOCHS))
print('Minibatch size : {}'.format(BATCH_SIZE))
print('Start of imagelist: {}'.format(START))
print('End of imagelist: {}'.format(FINISH))

# 使用するGPUの指定
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)

train('../progress/HIT-MW_0_2115', _list, root, EPOCHS, BATCH_SIZE, START, FINISH)
