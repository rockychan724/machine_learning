# -*- coding: utf-8 -*-

import tensorflow as tf


def main():
    w = tf.constant([[2.0, 3.0], [3.0, 2.0], [1.0, 1.0]])
    sess = tf.InteractiveSession()
    print(tf.contrib.layers.l1_regularizer(0.5)(w).eval())
    print(tf.contrib.layers.l2_regularizer(0.5)(w).eval())
    print(tf.argmax(w, 1).eval())
    print(tf.argmax(w, 0).eval())

    sess.close()


if __name__ == '__main__':
    main()
