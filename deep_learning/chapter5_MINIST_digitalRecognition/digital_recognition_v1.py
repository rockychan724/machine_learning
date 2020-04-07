# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# 神经网络结构相关参数
INPUT_NODE = 784  # 输入层的节点数（28*28=784，图片像素）
OUTPUT_NODE = 10  # 输出层的节点数（0~9十个数字表示十个类别）
LAYER_NODE = 500  # 隐藏层节点数，这里只有一个隐藏层

# 训练相关参数
TRAINING_STEPS = 30000  # 训练轮数
BATCH_SIZE = 100  # 每组数据的大小。batch越大，训练越接近梯度下降；batch越小，训练越接近随机梯度下降

# 其他参数
LEARNING_RATE_BASE = 0.8  # 学习率初始值
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化项的系数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 函数inference()用于计算神经网络前向传播结果
def inference(x, moving_average_class, weight1, bias1, weight2, bias2):
    # 当没有使用滑动平均模型时直接使用当前参数的取值
    if moving_average_class == None:
        # tf.nn.relu()使用ReLU激活函数去线性化
        layer = tf.nn.relu(tf.matmul(x, weight1) + bias1)
        y = tf.matmul(layer, weight2) + bias2
    else:
        # 使用滑动平均模型时，首先使用moving_average_class.average()函数来计算变量的滑动平均值，
        # 再计算相应的神经网络前向传播结果
        layer = tf.nn.relu(tf.matmul(x, moving_average_class.average(weight1)) + moving_average_class.average(bias1))
        y = tf.nn.relu(tf.matmul(layer, moving_average_class.average(weight2)) + moving_average_class.average(bias2))
    return y


def train(minist):
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='y_output')

    # 生成隐藏层的参数
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE]))
    # 生成输出层的参数
    weight2 = tf.Variable(tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 定义训练轮数
    global_step = tf.Variable(0, trainable=False)
    # 初始化滑动平均类
    moving_average_class = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, num_updates=global_step)
    # 将所有可训练的（trainable=True）变量使用滑动平均
    moving_average_update_op = moving_average_class.apply(tf.trainable_variables())

    # 不使用滑动平均模型时计算神经网络前向传播结果
    y = inference(x, None, weight1, bias1, weight2, bias2)
    # 使用滑动平均模型时计算神经网络前向传播结果
    average_y = inference(x, moving_average_class, weight1, bias1, weight2, bias2)

    # 计算损失函数
    # 关于tf.nn.sparse_softmax_cross_entropy_with_logits()函数的说明：
    # 1、传入的labels已经经过one-hot处理，shape为[batch_size], 传入的logits的shape为[batch_size, 10]。
    # 2、tf.nn.sparse_softmax_cross_entropy_with_logits()函数在计算交叉熵时会先将labels转化为one-hot格式（例如，3对应[0,0,0,1,0,0,0,0,0,0]）
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weight1) + regularizer(weight2)  # 计算正则化项，一般只计算权重的正则化损失，而不考虑偏置项
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, minist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    # 使用梯度下降算法优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 使用tf.group()同时更新反向传播中神经网络的参数以及变量的滑动平均值
    train_op = tf.group(train_step, moving_average_update_op)

    # 检验使用滑动平均模型的神经网络的前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 计算每一组数据的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validation_feed = {x: minist.validation.images, y_: minist.validation.labels}
        test_feed = {x: minist.test.images, y_: minist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validation_accuracy = sess.run(accuracy, feed_dict=validation_feed)
                print('After %d training steps, validation accuracy is %f' % (i, validation_accuracy))
            x_next, y_next = minist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: x_next, y_: y_next})

        test_accuracy = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training steps, test accuracy is %f' % (TRAINING_STEPS, test_accuracy))


def main(argv=None):
    minist = input_data.read_data_sets('D:\\PycharmProjects\\datasets\\minist', one_hot=True)
    start = time.process_time()
    train(minist)
    end = time.process_time()
    print('Running time is ', (end - start))


if __name__ == '__main__':
    tf.app.run()
