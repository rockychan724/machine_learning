# coding: utf-8

import tensorflow as tf


def main():
    v1 = tf.Variable(0, dtype=tf.float32)
    decay = 0.99
    # 定义训练轮数
    step = tf.Variable(0, trainable=False)
    # 定义滑动平均的类
    ema = tf.train.ExponentialMovingAverage(decay=decay, num_updates=step)
    # 指定变量列表，每次执行这个操作时都会更新列表中的变量
    maintain_average_op = ema.apply([v1])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # ema.average(v1)：获取变量v1的滑动平均值
        print(sess.run([v1, ema.average(v1)]))

        sess.run(tf.assign(v1, 5))
        # 更新v1的滑动平均值
        sess.run(maintain_average_op)
        print(sess.run([v1, ema.average(v1)]))


if __name__ == '__main__':
    main()
