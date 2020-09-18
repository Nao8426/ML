import os
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# csvファイルを作成したいデータセットが格納されているディレクトリのパス
dataset_path = './dataset'
# 保存先のディレクトリのパス（カレントディレクトリに作成する場合は'.'でOK）
savedir = './output'
# 作成するcsvファイルの名前（拡張子は自動で付くので，付けないこと．）
name = 'list'
# csvにindexを付与するかを決定('True'なら付与する)
ext_ind = True
# transを "True" にすると，Windowsにおける階層区切りの "\\" を "/" に変換する．
trans = False


def main(dataset_path, savedir, name, ext_ind, trans=False):
    df = pd.DataFrame(columns=['Path'])

    path = []    
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            text = '{}/{}'.format(root, f)
            if trans:
                text = text.replace('\\', '/')
            path.append(text)

    df['Path'] = path

    os.makedirs(savedir, exist_ok=True)
    # 同じ名前のファイルが存在する場合はナンバリングを付与する
    if os.path.exists('{}/{}.csv'.format(savedir, name)):
        num = 1
        while 1:
            if os.path.exists('{}/{}({}).csv'.format(savedir, name, num)):
                num += 1
            else:
                name = '{}({})'.format(name, num)
                break
    df.to_csv('{}/{}.csv'.format(savedir, name), index=ext_ind)


if __name__ == '__main__':
    main(dataset_path, savedir, name, ext_ind, trans=trans)
