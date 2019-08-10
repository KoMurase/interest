import numpy as np
#入力
x = np.array([2,3,1])

#正解
t = np.array([20])
# 0-1層間のパラメータ
w1 = np.array([[3, 1, 2], [-2, -3, -1]])
b1 = np.array([0, 0])
# 2-3層間のパラメータ
w2 = np.array([[3, 2]])
b2 = np.array([0])
#中間層の計算
u1 = w1.dot(x) + b1
h1 = 1. / (1 + np.exp(-u1))

#出力の計算
y = w2.dot(h1) + b2

print(y)

# dL / dy
dLdy = -2 * (t - y)

# dy / dw_2
dydw2 = h1
# dL / dw_2: 求めたい勾配
dLdw2 = dLdy * dydw2

print(dLdw2)

# d y / d h1
dydh1 = w2

# d h1 / d u1
dh1du1 = h1 * (1 - h1)

# d u_1 / d w1
du1dw1 = x

# 上から du1 / dw1 の直前までを一旦計算
dLdu1 = dLdy * dydh1 * dh1du1

# du1dw1は (3,) というshapeなので、g_u1w1[None]として(1, 3)に変形
du1dw1 = du1dw1[None]

# dL / dw_1: 求めたい勾配
dLdw1 = dLdu1.T.dot(du1dw1)

print(dLdw1)
