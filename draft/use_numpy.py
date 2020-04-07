# -*- coding: utf-8 -*-

import numpy as np


def test1():
    a = np.linspace(1, 10, 10, dtype=np.int)
    print(type(a))
    # print(a)
    # print(np.sin(a))
    b = np.linspace(0.01, 10000*0.01, 10000, dtype=np.float)
    print(b)
    print(len(b))


def test2():
    a = np.arange(1.0, 10.0, 0.1)
    print(type(a))
    print(a)


if __name__ == '__main__':
    test2()
