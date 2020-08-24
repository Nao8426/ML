# 同一入力によるジェネレータの出力画像の経過動画を作成
import argparse
import os
# 自作モジュール
from generate import generate
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# モデルまでのパス
root = '../progress/tmp'

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description='Deep Convolutional Generative Adversarial Net')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--num', '-n', default=100, type=int, help='number of models')
parser.add_argument('--batchsize', '-b', default=32, type=int, help='learning minibatch size')
parser.add_argument('--nz', '-z', default=32, type=int, help='dimensionality of random vevtors')
parser.add_argument('--width', '-w', default=256, type=int, help='width of images')
parser.add_argument('--height', '-l', default=128, type=int, help='height of images')
parser.add_argument('--channel', '-c', default=1, type=int, help='channel of images')
args = parser.parse_args()

# コマンドライン引数により指定されたパラメータを変数に格納
GPU_ID = args.gpu
NUM = args.num
BATCH_SIZE = args.batchsize
NZ = args.nz
WIDTH = args.width
HEIGHT = args.height
CHANNEL = args.channel

# 設定内容を表示
print('GPU: {}'.format(GPU_ID))
print('Num. of models: {}'.format(NUM))
print('Minibatch size: {}'.format(BATCH_SIZE))
print('Dim. of random vectors: {}'.format(NZ))
print('Widht of images: {}'.format(WIDTH))
print('Height of images: {}'.format(HEIGHT))
print('Channel of images: {}'.format(CHANNEL))

# 使用するGPUの指定
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)

generate('./output', root, NUM, BATCH_SIZE, NZ, WIDTH, HEIGHT, CHANNEL)
