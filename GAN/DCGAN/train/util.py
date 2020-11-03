# その他いろいろ
import matplotlib.pyplot as plt
import os


# ロスの履歴をプロット
def plot(G_loss, D_loss, epochs, dirname):
    fig = plt.figure(figsize=(6.0, 4.8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, G_loss, label='Generator Loss')
    ax.plot(epochs, D_loss, label='Discriminator Loss')
    ax.set_title('Loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.legend(loc='best')
        
    os.makedirs(dirname, exist_ok=True)
    fig.savefig('{}/loss/loss_{}.png'.format(dirname, len(epochs)))

    plt.close()


# 環境をアウトプットしてテキストファイルで保存
def output_env(filepath, batch_size, nz, G_opt_para, D_opt_para, G_model, D_model):
    text_set = '##### Setting #####\nMinibatch size: {}\nDim. of random vectors: {}\n\n'.format(batch_size, nz)
    text_G_opt = '##### Optimizer Parameter #####\nGenerator ==> lr: {}, betas: {}, weight_decay: {}\n'.format(G_opt_para['lr'], G_opt_para['betas'], G_opt_para['weight_decay'])
    text_D_opt = 'Discriminator ==> lr: {}, betas: {}, weight_decay: {}\n\n'.format(D_opt_para['lr'], D_opt_para['betas'], D_opt_para['weight_decay'])
    text_G_model = '##### Generator model #####\n{}\n\n'.format(G_model)
    text_D_model = '##### Discriminator model #####\n{}'.format(D_model)
    with open(filepath, mode='w') as f:
        f.write(text_set)
        f.write(text_G_opt)
        f.write(text_D_opt)
        f.write(text_G_model)
        f.write(text_D_model)
