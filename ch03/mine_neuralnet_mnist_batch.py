# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist # datasetフォルダのnmistファイルからload_mnistモジュールを読み込み
from common.functions import sigmoid, softmax # commonフォルダのfunctionファイルからsigmoid,softmaxモジュールを読み込み


# mnistからデータを取得
def get_data():
    (x_train, t_test), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 学習済みパラメータの読み込み
# sample_weight.pklには重みとバイアスのパラメータがディクショナリ型の変数として保存されている
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 予測を行う
# この関数は各ラベルの確率がNumPy配列として出力される
# ex)[0.1, 0.3, 0.2, ... , 0.04]
# 0の確率が0.1, 1の確率が0.3 ... という解釈できる
def predict(network, x):
    # 重みの設定
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # バイアスの設定
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 出力 = 入力 * 重み + バイアス
    # 隠れ層の活性化関数はシグモイド関数を利用する
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    # 出力層で使用する関数はソフトマックス関数
    y = softmax(a3)

    return y

# データの取得
x, t = get_data()

# 重みとバイアス設定
network = init_network()

accuracy_cnt = 0 # 正解の精度

# xに格納された画像データを1枚ずつ取り出して処理
for i in range(len(x)):
    y = predict(network, x[i])
    # 引数の配列の中で、一番スカラの大きいインデックスを取得
    # (一番確率の高い要素のインデックスを取得)
    p = np.argmax(y)

    # 予測値と正解ラベルを比較
    if p == t[i]:
        accuracy_cnt += 1

# 正答率の出力
print("Accuracy: " + str(float(accuracy_cnt) / len(x))) # Accuracy: 0.9352
