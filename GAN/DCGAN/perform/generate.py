import torch
import torchvision
# 自作モジュール
from network import Generator


def generate(savedir, model_path, num):
    nz = 32
    width = 72
    height = 72
    channel = 1

    device = 'cuda'

    model = Generator(nz, width, height, channel)
    model = nn.DataParallel(model)

    model.module.load_state_dict(torch.load(model_path))

    model.eval()    # 推論モードへ切り替え（Dropoutなどの挙動に影響）

    # 保存先のファイルを作成
    if os.path.exists(savedir):
        n = 1
        while 1:
            if os.path.exists('{}({})'.format(savedir, n)):
                n += 1
            else:
                savedir = '{}({})'.format(savedir, n)
                break
    os.makedirs(savedir, exist_ok=True)

    for i in range(num):
        # 入力の乱数を作成
        z = torch.randn(1, nz, 1, 1)
        z = z.to(device)

        gene_img = model(z)

        # ジェネレータの出力画像を保存
        torchvision.utils.save_image(gene_img, "{}/{:04}.png".format(savedir, i+1))
