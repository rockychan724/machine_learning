# -*- coding: utf-8 *--

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import inference

# 配置神经网络参数
TRAINING_STEPS = 30000  # 训练轮数
BATCH_SIZE = 100  # 每组数据的大小。batch越大，训练越接近梯度下降；batch越小，训练越接近随机梯度下降
LEARNING_RATE_BASE = 0.8  # 学习率初始值
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化项的系数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 模型文件的保存路径
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'digital_recognition.ckpt'


def train(minist):
    x = tf.placeholder(tf.float32, shape=[None, inference.INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, shape=[None, inference.OUTPUT_NODE], name='y_output')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x, regularizer)

    # 定义滑动平均操作
    global_step = tf.Variable(0, trainable=False)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    moving_average_op = ema.apply(tf.trainable_variables())

    # 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('regularization'))

    # 设置学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, minist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    # 使用梯度下降法优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 使用tf.group()同时更新反向传播中神经网络的参数以及变量的滑动平均值
    train_op = tf.group(train_step, moving_average_op)

    # 初始化tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            x_next, y_next = minist.train.next_batch(BATCH_SIZE)
            train_op_value, loss_value, global_step_value = sess.run([train_op, loss, global_step],
                                                               feed_dict={x: x_next, y_: y_next})
            if i%1000 == 0:
                print('After %d training steps, loss on training batch is %f' % (global_step_value, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step_value)


def main(argv=None):
    minist = input_data.read_data_sets('D:\\PycharmProjects\\datasets\\minist', one_hot=True)
    train(minist)


if __name__ == '__main__':
    tf.app.run()
