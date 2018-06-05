# coding:utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    # コンストラクタ
    def __init__(self):
        # インスタンス変数の初期化 2*3の重みパラメータ
        # 最初の重みはテキトーに決定する必要がある
        self.W = np.random.randn(2,3) # ガウス分布(平均0、標準偏差1)で重みを初期化

    # 予測(入力値*重み)
    def predict(self, x):
        return np.dot(x, self.W)

    # 損失関数（交差エントロピー）
    # xには入力データ、tには正解ラベルがone-hotで入力される
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
