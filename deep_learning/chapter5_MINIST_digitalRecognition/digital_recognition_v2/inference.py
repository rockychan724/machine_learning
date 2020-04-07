# -*- coding: utf-8 *--

import tensorflow as tf

# 神经网络结构相关参数
INPUT_NODE = 784  # 输入层的节点数（28*28=784，图片像素）
OUTPUT_NODE = 10  # 输出层的节点数（0~9十个数字表示十个类别）
LAYER_NODE = 500  # 隐藏层节点数，这里只有一个隐藏层


def get_weight_variable(shape, regularizer):
    weight = tf.get_variable('weight', shape=shape, dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('regularization', regularizer(weight))
    return weight


def inference(x, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER_NODE], regularizer)
        bias = tf.get_variable('bias', [LAYER_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(x, weights) + bias)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER_NODE, OUTPUT_NODE], regularizer)
        bias = tf.get_variable('bias', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + bias
    return layer2
