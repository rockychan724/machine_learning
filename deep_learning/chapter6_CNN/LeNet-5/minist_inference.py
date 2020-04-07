# coding: utf-8

import tensorflow as tf

# 配置神经网络参数
# 图片的大小、通道数、类别标注
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 输入层和输出层的节点数
INPUT_NODE = 28 * 28
OUTPUT_NODE = 10

# 第一层卷积层过滤器的尺寸和深度（有些地方把过滤器的深度称为过滤器的个数）
CONV1_SIZE = 5
CONV1_DEEP = 32

# 第二层卷积层过滤器的尺寸和深度
CONV2_SIZE = 5
CONV2_DEEP = 64

# 全连接层的节点数
FC_SIZE = 512


def inference(input_tensor, regularizer, train):
    # 第一层（卷积层1）
    with tf.variable_scope('layer1-conv1'):
        lay1_weight = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        lay1_bias = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用尺寸为5*5，深度为32的过滤器，过滤器的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, lay1_weight, [1, 1, 1, 1], padding='SAME')
        # y1大小为28*28*32
        y1 = tf.nn.relu(tf.nn.bias_add(conv1, bias=lay1_bias))

    # 第二层（池化层1）
    with tf.variable_scope('layer2-pool1'):
        # 池化层过滤器的尺寸为2*2
        # y2大小为14*14*32
        y2 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第三层（卷积层2）
    with tf.variable_scope('layer3-conv2'):
        lay3_weight = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        lay3_bias = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(y2, lay3_weight, [1, 1, 1, 1], padding='SAME')
        # y3大小14*14*64
        y3 = tf.nn.relu(tf.nn.bias_add(conv2, bias=lay3_bias))

    # 第四层（池化层2）
    with tf.variable_scope('layer4-pool2'):
        # y4大小为7*7*64
        y4 = tf.nn.max_pool(y3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    y4_shape = y4.get_shape().as_list()  # y4_shape[0]为batch_size
    nodes = y4_shape[1] * y4_shape[2] * y4_shape[3]  # nodes = 3136
    # fc_input = tf.reshape(y4, [y4_shape[0], nodes])
    fc_input = tf.layers.flatten(y4)  # 查看flatten函数的功能，张量fc_input的shape为(?, 3136)

    # 第五层（全连接层1）
    with tf.variable_scope('layer5-fc1'):
        lay5_weight = tf.get_variable('weight', [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        lay5_bias = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(lay5_weight))
        # y5的大小为batch_size*512
        y5 = tf.nn.relu(tf.matmul(fc_input, lay5_weight) + lay5_bias)
        if train:
            y5 = tf.nn.dropout(y5, 0.5)  # 避免过拟合问题

    # 第六层（全连接层2或输出层）
    with tf.variable_scope('layer6-fc2'):
        lay6_weight = tf.get_variable('weight', [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        lay6_bias = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(lay6_weight))
        y6 = tf.matmul(y5, lay6_weight) + lay6_bias

    return y6
