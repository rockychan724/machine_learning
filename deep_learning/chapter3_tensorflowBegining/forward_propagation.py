# -*- coding: utf-8 -*-
# content: forward-propagation algorithm

import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print('********* begin **********')
    # x = tf.constant([[0.7, 0.9]])
    x = tf.placeholder(dtype=tf.float32, shape=(1,2), name='input')
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    with tf.Session() as sess:
        # sess.run(w1.initializer)
        # sess.run(w2.initializer)
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        # print(sess.run(w1))
        # print(sess.run(w2))
        # print(sess.run(a))
        print(sess.run(y, feed_dict={x:[[0.7,0.9]]}))

    print('********** end ***********')


if __name__ == '__main__':
    main()
