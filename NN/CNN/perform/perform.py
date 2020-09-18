# 学習済みのモデルを用いてテストデータに対する精度を計算
import argparse
import os
import pandas as pd
import torch
import torchvision
from torch import nn
# 自作モジュール
from load import LoadDataset
from network import CNN
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def evaluate(model_path, root, test_list, batch_size):
    channel = 1

    device = 'cuda'

    model = CNN(channel)
    model = nn.DataParallel(model)

    model.module.load_state_dict(torch.load(model_path))

    model.eval()    # 推論モードへ切り替え（Dropoutなどの挙動に影響）

    df_test = pd.read_csv(test_list)
    test_loader = LoadDataset(df_test, root)
    test_dataset = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True, drop_last=True)

    correct = 0

    for img, label in test_dataset:
        img = img.float()
        img = img / 255
        # 3次元テンソルを4次元テンソルに変換（1チャネルの情報を追加）
        batch, height, width = img.shape
        img = torch.reshape(img, (batch_size, 1, height, width))

        img = img.to(device)

        output = model(img)
        pred = output.data.max(1, keepdim=False)[1]
        for i, l in enumerate(label):
            if l == pred[i]:
                correct += 1

    data_num = len(test_dataset.dataset)  # データの総数
    acc = correct / data_num * 100 # 精度

    print('Accuracy for test dataset: {}/{} ({:.1f}%)'.format(correct, data_num, acc))


if __name__ == '__main__':
    model_path = '../progress/tmp/model/model_10.pth'
    test_list = '../imagelist/test.csv'
    root = '../dataset'  # データセットまでのパス

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='Evaluation of CNN')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=32, type=int, help='minibatch size')
    args = parser.parse_args()

    # コマンドライン引数により指定されたパラメータを変数に格納
    GPU_ID = args.gpu
    BATCH_SIZE = args.batchsize

    # 設定内容を表示
    print('GPU: {}'.format(GPU_ID))
    print('Minibatch size : {}'.format(BATCH_SIZE))

    # 使用するGPUの指定
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)

    evaluate(model_path, root, test_list, BATCH_SIZE)
