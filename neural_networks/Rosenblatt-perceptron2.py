# -*- coding:utf-8 -*-
# content: dichotomous model(二分类模型)

import numpy as np

b = 1  # bias
a = 0.5  # learning rate
x = np.array([[1, 3, 1], [2, 5, 1], [1, 8, 1], [2, 15, 1]])  # input
d = np.array([1, 1, -1, -1])  # target output
w = np.array([0, 0, b])  # initial weight value
err = 0  # error
train_count = 50  # num of training


def sgn(v):
    if v > 0:
        return 1
    else:
        return -1


def y(x, w):
    return sgn(np.dot(w.T, x))


def gradient(x, w, aa):
    i = 0;
    sum_x = np.array([0,0,0])
    for xx in x:
        if y(xx, w) != d[i]:
            sum_x += d[i] * xx
        i += 1
    return aa * sum_x


if __name__ == '__main__':
    i = 0
    print('Before training, w is', w)
    while True:
        tidu = gradient(x, w, a)
        w = w + tidu
        i += 1
        if (abs(tidu.sum()) < err) or (i >= train_count):
            break
    test1 = np.array([9,19,1])
    test2 = np.array([9,64,1])
    print('After training, w is', w)
    print('(%d, %d) --> class %d' % (test1[0], test1[1], y(test1, w)))
    print('(%d, %d) --> class %d' % (test2[0], test2[1], y(test2, w)))
