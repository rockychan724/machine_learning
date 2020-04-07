# -*- coding: utf-8 -*-

import tensorflow as tf
from numpy.random import RandomState


def main():
    batch_size = 8
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x_input')
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y_input')
    w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
    y = tf.matmul(x, w)

    # 自定义损失函数
    loss_more = 1
    loss_less = 10
    loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 通过随机数生成模拟数据集
    random = RandomState(1)
    dataset_size = 128
    X = random.rand(dataset_size, 2)
    Y = [[x1 + x2 + random.rand() / 10.0 - 0.05] for (x1, x2) in X]

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        steps = 5000
        print('*********start********')
        for i in range(steps):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            # print(X[start:end])
            # print(Y[start:end])
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            loss_ = sess.run(loss, feed_dict={x: X[start:end], y_: Y[start:end]})
            if i % 1000 == 0:
                print('After %d training steps, the loss is %f' % (i, loss_))
                print('w = ', w.eval())
        print('*********end**********')


if __name__ == '__main__':
    main()
