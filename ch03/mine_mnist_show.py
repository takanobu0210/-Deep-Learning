# coding: utf-8
# minstの画像を表示するプログラム
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image # 画像を表示するモジュール

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# minstのデータを読み込む
# flatten=Trueで読み込むと、読み込んだデータは１次元配列になるため、
# のちほど元の画像サイズに変更する必要がある（Image.fromarray）
# (画像, ラベル)で取得できる
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]

# 0番目のデータを表示してみる
print(label) # 5
print(img.shape) # (784,)
img = img.reshape(28, 28) # 形状を元の画像サイズに変形
print(img.shape) # (28, 28)

img_show(img)
