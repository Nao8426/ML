# その他いろいろ
import matplotlib.pyplot as plt
import os


# ロスの履歴をプロット
def plot(loss, epochs, dirname):
    fig = plt.figure(figsize=(4.8, 4.8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, loss)
    ax.set_title('Loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
        
    os.makedirs(dirname, exist_ok=True)
    fig.savefig('{}/loss/loss_{}.png'.format(dirname, len(epochs)))

    plt.close()


# 環境をアウトプットしてテキストファイルで保存
def output_env(filepath, batch_size, opt_para, model):
    text_set = '##### Setting #####\nMinibatch size: {}\n\n'.format(batch_size)
    text_opt = '##### Optimizer Parameter #####\nEncoder ==> lr: {}, betas: {}, weight_decay: {}\n'.format(opt_para['lr'], opt_para['betas'], opt_para['weight_decay'])
    text_model = '##### Model #####\n{}'.format(model)
    with open(filepath, mode='w') as f:
        f.write(text_set)
        f.write(text_opt)
        f.write(text_model)
