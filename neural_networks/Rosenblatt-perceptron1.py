# -*- coding: utf-8 -*-
# content: train 'or' operation model

import numpy as np

b = 0  # bias
a = 0.5  # learning rate
x = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]])  # input
d = np.array([1, 1, 1, 0])  # target output
w = np.array([0, 0, b])  # initial weight value


def sgn(v):
    if v > 0:
        return 1
    else:
        return 0


def y(ww, xx):
    return sgn(np.dot(ww, xx))


def next_w(old_w, xx, a, d):
    return old_w + a * (d - y(old_w, xx)) * xx


if __name__ == '__main__':
    i = 0
    print('Before learning, w =', w)
    for xx in x:
        print('%d or %d = %d' % (xx[0], xx[1], y(w, xx)))
    for xx in x:
        w = next_w(w, xx, a, d[i])
        i += 1
    print('After learning, w =', w)
    for xx in x:
        print('%d or %d = %d' % (xx[0], xx[1], y(w, xx)))
