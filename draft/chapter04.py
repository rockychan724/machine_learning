# -*- encoding: utf-8 -*-

import tensorflow as tf
import math


def main():
    # y_ = tf.constant([1., 2., 3.])
    # y = tf.constant([0.8, 1.5, 3.1])
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
    # with tf.Session().as_default():
    #     print(cross_entropy.eval())
    #     print(tf.nn.softmax(v).eval())
    #     print(sum(tf.nn.softmax(v).eval()))
    #     print(tf.sparse_softmax(v).eval())
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[1, 0, 3], [1, 5, 1]]
    with tf.Session() as sess:
        print(sess.run(tf.equal(a, b)))


if __name__ == '__main__':
    main()
