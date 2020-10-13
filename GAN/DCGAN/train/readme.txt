main.py: 学習を回すメインプログラム．学習を行う際はこのプログラムを実行．
train.py: 学習内容（パラメータや順序など）を設定．
network.py: ジェネレータとディスクリミネータの層構造．
load.py: データセットを読み込む際の設定．
util.py: 環境出力など，その他の自作関数．

########## 備考 ##########
"Batch Normalization"をディスクリミネータだけに置くか，ジェネレータにも置くかなど，"Batch Normalization"の位置には諸説あるが，とりあえずディスクリミネータだけに置くのがよいっぽいので，とりあえずそれで実装してます．
ジェネレータの出力層，ディスクリミネータの入力層には"Batch Normalization"を置かない方がよい（学習が不安定になる）らしい．
"Batch Normalization"よりも　"Spectral Normalization"の方が学習が安定するかも．（検証した感じ多分そう）
Spectral Normalizationを使用する場合は"nn.BatchNorm2d(channel)"を消して"nn.utils.spectral_norm(module)"を追加していけばよい．（"module"は層をそのまま囲めばOK）
一般的にジェネレータの出力層は活性化関数として"Tanh"を用いることが多いが，（学習モデルを使用するときの正規化をどうするのか微妙にややこしかったので）ディスクリミネータに合わせて"Sigmoid"を使用してます．"Tanh"を用いるなら，画像の正規化を0~1ではなく-0.5~0.5に変えて下さい．
ロスに関しては"BCELoss"か"BCEWithLogitsLoss"のどちらを使用するか沼った（ここらへんも"Sigmoid"使った理由の一つ）ので，とりあえず"BCELoss"使ってます．（ただ"BCELoss"を使用するとロスがたまにバグったりするかも・・・）
