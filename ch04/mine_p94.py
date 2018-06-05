# ミニバッチ学習
# データをランダムに抽出する方法
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

# ランダムに10枚だけ抜き出す
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# この出力は実行の度に変化する
# ex)[42046 53515  8543 48925 23975 16930 51302 58674 14433 35769]
# 0~60000までのインデックスの中からランダムで出力される
print(batch_mask)
