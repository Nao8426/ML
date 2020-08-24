# その他いろいろ
import matplotlib.pyplot as plt
import os


# ロスの履歴をプロット
def plot(loss_G, loss_D, loss_E, loss_CD, epochs, dirname):
    fig = plt.figure(figsize=(12.8, 4.8))

    ax_GD = fig.add_subplot(1, 2, 1)
    ax_GD.plot(epochs, loss_G, label='Generator Loss')
    ax_GD.plot(epochs, loss_D, label='Discriminator Loss')
    ax_GD.set_title('Loss(Gen vs. Dis)')
    ax_GD.set_xlabel('epoch')
    ax_GD.set_ylabel('loss')
    plt.legend(loc='best')

    ax_EcD = fig.add_subplot(1, 2, 2)
    ax_EcD.plot(epochs, loss_E, label='Encoder Loss')
    ax_EcD.plot(epochs, loss_CD, label='CodeDiscriminator Loss')
    ax_EcD.set_title('Loss(Enc vs. CodeDis)')
    ax_EcD.set_xlabel('epoch')
    ax_EcD.set_ylabel('loss')
    plt.legend(loc='best')
        
    os.makedirs(dirname, exist_ok=True)
    fig.savefig('{}/loss/loss_{}.png'.format(dirname, len(epochs)))

    plt.close()


# 環境をアウトプットしてテキストファイルで保存
def output_env(filepath, batch_size, nz, para_G, para_D, para_E, para_CD, gen_model, dis_model, enc_model, cdis_model):
    text_set = '#####Setting#####\nMinibatch size: {}\nDim. of random vectors: {}\n\n'.format(batch_size, nz)
    text_opt_G = '##### Optimizer Parameter #####\nGenerator ==> lr: {}, betas: {}, weight_decay: {}\n'.format(para_G['lr'], para_G['betas'], para_G['weight_decay'])
    text_opt_D = 'Discriminator ==> lr: {}, betas: {}, weight_decay: {}\n'.format(para_D['lr'], para_D['betas'], para_D['weight_decay'])
    text_opt_E = 'Encoder ==> lr: {}, betas: {}, weight_decay: {}\n'.format(para_E['lr'], para_E['betas'], para_E['weight_decay'])
    text_opt_CD = 'CodeDiscriminator ==> lr: {}, betas: {}, weight_decay: {}\n\n'.format(para_CD['lr'], para_CD['betas'], para_CD['weight_decay'])
    text_generator = '##### Generator model #####\n{}\n\n'.format(gen_model)
    text_discriminator = '##### Discriminator model #####\n{}\n\n'.format(dis_model)
    text_encoder = '##### Encoder model #####\n{}\n\n'.format(enc_model)
    text_codediscriminator = '##### CodeDiscriminator model #####\n{}'.format(cdis_model)
    with open(filepath, mode='w') as f:
        f.write(text_set)
        f.write(text_opt_G)
        f.write(text_opt_D)
        f.write(text_opt_E)
        f.write(text_opt_CD)
        f.write(text_generator)
        f.write(text_discriminator)
        f.write(text_encoder)
        f.write(text_codediscriminator)
