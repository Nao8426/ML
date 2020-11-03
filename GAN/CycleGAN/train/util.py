# その他いろいろ
import matplotlib.pyplot as plt
import os


# ロスの履歴をプロット
def plot(G_loss, D_A_loss, D_B_loss, epochs, dirname):
    fig = plt.figure(figsize=(6.0, 4.8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, G_loss, label='Generator Loss')
    ax.plot(epochs, D_A_loss, label='Discriminator A Loss')
    ax.plot(epochs, D_B_loss, label='Discriminator B Loss')
    ax.set_title('Loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.legend(loc='best')
        
    os.makedirs(dirname, exist_ok=True)
    fig.savefig('{}/loss/loss_{}.png'.format(dirname, len(epochs)))

    plt.close()


# 環境をアウトプットしてテキストファイルで保存
def output_env(filepath, batch_size, G_opt_para, D_A_opt_para, D_B_opt_para, G_A2B_model, G_B2A_model, D_A_model, D_B_model):
    text_set = '##### Setting #####\nMinibatch size: {}\n\n'.format(batch_size)
    text_G_opt = '##### Optimizer Parameter #####\nGenerator ==> lr: {}, betas: {}, weight_decay: {}\n'.format(G_opt_para['lr'], G_opt_para['betas'], G_opt_para['weight_decay'])
    text_D_A2B_opt = 'Discriminator A ==> lr: {}, betas: {}, weight_decay: {}\n'.format(D_A_opt_para['lr'], D_A_opt_para['betas'], D_A_opt_para['weight_decay'])
    text_D_B2A_opt = 'Discriminator B ==> lr: {}, betas: {}, weight_decay: {}\n\n'.format(D_B_opt_para['lr'], D_B_opt_para['betas'], D_B_opt_para['weight_decay'])
    text_G_A2B_model = '##### Generator A2B model #####\n{}\n\n'.format(G_A2B_model)
    text_G_B2A_model = '##### Generator B2A model #####\n{}\n\n'.format(G_B2A_model)
    text_D_A_model = '##### Discriminator A model #####\n{}\n\n'.format(D_A_model)
    text_D_B_model = '##### Discriminator B model #####\n{}'.format(D_B_model)
    with open(filepath, mode='w') as f:
        f.write(text_set)
        f.write(text_G_opt)
        f.write(text_D_A2B_opt)
        f.write(text_D_B2A_opt)
        f.write(text_G_A2B_model)
        f.write(text_G_B2A_model)
        f.write(text_D_A_model)
        f.write(text_D_B_model)
