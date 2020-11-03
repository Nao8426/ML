# その他いろいろ
import matplotlib.pyplot as plt
import os


# ロスの履歴をプロット
def plot(G_loss, D_loss, E_loss, CD_loss, epochs, dirname):
    fig = plt.figure(figsize=(12.8, 4.8))

    ax_GD = fig.add_subplot(1, 2, 1)
    ax_GD.plot(epochs, G_loss, label='Generator Loss')
    ax_GD.plot(epochs, D_loss, label='Discriminator Loss')
    ax_GD.set_title('Loss(Gen vs. Dis)')
    ax_GD.set_xlabel('epoch')
    ax_GD.set_ylabel('loss')
    plt.legend(loc='best')

    ax_ECD = fig.add_subplot(1, 2, 2)
    ax_ECD.plot(epochs, E_loss, label='Encoder Loss')
    ax_ECD.plot(epochs, CD_loss, label='CodeDiscriminator Loss')
    ax_ECD.set_title('Loss(Enc vs. CodeDis)')
    ax_ECD.set_xlabel('epoch')
    ax_ECD.set_ylabel('loss')
    plt.legend(loc='best')
        
    os.makedirs(dirname, exist_ok=True)
    fig.savefig('{}/loss/loss_{}.png'.format(dirname, len(epochs)))

    plt.close()


# 環境をアウトプットしてテキストファイルで保存
def output_env(filepath, batch_size, nz, G_opt_para, D_opt_para, E_opt_para, CD_opt_para, G_model, D_model, E_model, CD_model):
    text_set = '##### Setting #####\nMinibatch size: {}\nDim. of random vectors: {}\n\n'.format(batch_size, nz)
    text_G_opt = '##### Optimizer Parameter #####\nGenerator ==> lr: {}, betas: {}, weight_decay: {}\n'.format(G_opt_para['lr'], G_opt_para['betas'], G_opt_para['weight_decay'])
    text_D_opt = 'Discriminator ==> lr: {}, betas: {}, weight_decay: {}\n'.format(D_opt_para['lr'], D_opt_para['betas'], D_opt_para['weight_decay'])
    text_E_opt = 'Encoder ==> lr: {}, betas: {}, weight_decay: {}\n'.format(E_opt_para['lr'], E_opt_para['betas'], E_opt_para['weight_decay'])
    text_CD_opt = 'CodeDiscriminator ==> lr: {}, betas: {}, weight_decay: {}\n\n'.format(CD_opt_para['lr'], CD_opt_para['betas'], CD_opt_para['weight_decay'])
    text_G_model = '##### Generator model #####\n{}\n\n'.format(G_model)
    text_D_model = '##### Discriminator model #####\n{}\n\n'.format(D_model)
    text_E_model = '##### Encoder model #####\n{}\n\n'.format(E_model)
    text_CD_model = '##### CodeDiscriminator model #####\n{}'.format(CD_model)
    with open(filepath, mode='w') as f:
        f.write(text_set)
        f.write(text_G_opt)
        f.write(text_D_opt)
        f.write(text_E_opt)
        f.write(text_CD_opt)
        f.write(text_G_model)
        f.write(text_D_model)
        f.write(text_E_model)
        f.write(text_CD_model)
