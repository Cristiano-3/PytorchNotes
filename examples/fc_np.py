# coding:utf-8
import numpy as np 

'''
三层网络 1000-100-10
y_pred = relu(x*w1)*w2
loss = square(y_pred - y).sum()
'''
N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# epoch=500, batch=whole_dataset
learning_rate = 1e-6
for t in range(500):
    # part 1 -----------------------------
    # forward
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # part 2 -----------------------------
    # backward, 
    # 连式法则求导 grad_w1 和 grad_w2, 
    # 求 grad_w2 时注意 relu 的地方
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    # update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    