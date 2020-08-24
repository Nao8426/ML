# その他いろいろ
import matplotlib.pyplot as plt
import os


# ロスの履歴をプロット
def plot(loss_G, loss_D, epochs, dirname):
    fig = plt.figure(figsize=(6.0, 4.8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, loss_G, label='Generator Loss')
    ax.plot(epochs, loss_D, label='Discriminator Loss')
    ax.set_title('Loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.legend(loc='best')
        
    os.makedirs(dirname, exist_ok=True)
    fig.savefig('{}/loss/loss_{}.png'.format(dirname, len(epochs)))

    plt.close()


# 環境をアウトプットしてテキストファイルで保存
def output_env(filepath, batch_size, nz, para_G, para_D, gen_model, dis_model):
    text_set = '##### Setting #####\nMinibatch size: {}\nDim. of random vectors: {}\n\n'.format(batch_size, nz)
    text_opt_G = '##### Optimizer Parameter #####\nGenerator ==> lr: {}, betas: {}, weight_decay: {}\n'.format(para_G['lr'], para_G['betas'], para_G['weight_decay'])
    text_opt_D = 'Discriminator ==> lr: {}, betas: {}, weight_decay: {}\n\n'.format(para_D['lr'], para_D['betas'], para_D['weight_decay'])
    text_generator = '##### Generator model #####\n{}\n\n'.format(gen_model)
    text_discriminator = '##### Discriminator model #####\n{}'.format(dis_model)
    with open(filepath, mode='w') as f:
        f.write(text_set)
        f.write(text_opt_G)
        f.write(text_opt_D)
        f.write(text_generator)
        f.write(text_discriminator)
