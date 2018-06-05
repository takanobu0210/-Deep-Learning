# coding: utf-8
# 全ての変数を一度に偏微分します
# すべての変数の偏微分をベクトルとしてまとめたもの：勾配
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # xと同じ形状の配列を作成（全要素が0）

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)

        # 中心差分をとって微分
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 値を元に戻す

    return grad

# 勾配を求める関数
def numerical_gradient(f, X):
    # 1次元にも対応
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X) # xと同じ形状の配列を作成（全要素が0）

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad

# y = x0**2 + x1**2 の関数
def function_2(x):
    # 1次元の配列の場合とそうでない場合で場合分け(ndimで次元数を取得)
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

# メイン処理
if __name__ == '__main__':
    # [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)

    # 格子点を生成
    X, Y = np.meshgrid(x0, x1)

    # ネストされた配列を1次元配列にフラット化
    X = X.flatten()
    Y = Y.flatten()

    # 勾配(gradient)を求める
    grad = numerical_gradient(function_2, np.array([X, Y]))

    # グラフの描画
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
