# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from mine_gradient_2d import numerical_gradient
%matplotlib inline

# 勾配降下法
# 引数fは最適化したい関数、init_xは初期値、lrは学習率、step_numは勾配法の繰り返し回数
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    # 0~100までループ
    for i in range(step_num):
        # 関数の勾配を求める
        grad = numerical_gradient(f, x)

        # 勾配法の数式に当てはめる
        x -= lr * grad
    return x

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100) # array([-6.11110793e-10,  8.14814391e-10])
