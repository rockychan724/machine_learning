# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

TRAIN_EXAMPLES = 10000
TEST_EXAMPLES = 1000
SAMPLE_GAP = 0.01

NUM_LAYERS = 2
HIDDEN_SIZE = 30

LEARNING_RTAE = 0.1

TRAIN_STEP = 10000
BATCH_SIZE = 32


def lstm_model(x, y, training=False):
    lstm = tf.nn.rnn_cell.BasicLSTMCell
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = output[:, -1, :]
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    if not training:
        return predictions, loss, None
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer='Adagrad',
                                               learning_rate=LEARNING_RTAE)
    return predictions, loss, train_op


def train(sess, train_x, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model'):
        predictions, loss, train_op = lstm_model(x, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAIN_STEP):
        l, _ = sess.run([loss, train_op])
        if (i + 1) % 100 == 0:
            print('train step: {}, loss: {}'.format(i, l))


def test(sess, test_x, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model', reuse=True):
        prediction, loss, _ = lstm_model(x, y, False)

    predicitons = []
    losses = []
    for i in range(TEST_EXAMPLES):
        p, l = sess.run([prediction, loss])
        predicitons.append(p)
        losses.append(l)

    predicitons = np.array(predicitons).squeeze()
    losses = np.array(predicitons).squeeze()
    rmse = np.sqrt(((predicitons - losses)**2).mean(axis=0))
    print('MSE: ', rmse)
    plt.figure()
    plt.plot(predicitons, label='prediction')
    plt.plot(y, label='real sin value')
    plt.legend()
    plt.show()


def generate_data():
    train_x = np.linspace(SAMPLE_GAP, TRAIN_EXAMPLES * SAMPLE_GAP, TRAIN_EXAMPLES, dtype=np.float)
    train_y = np.sin(train_x)
    test_x = np.linspace(SAMPLE_GAP, TEST_EXAMPLES * SAMPLE_GAP, TEST_EXAMPLES, dtype=np.float)
    test_y = np.sin(test_x)
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = generate_data()
    with tf.Session() as sess:
        train(sess, train_x, train_y)
        test(sess, test_x, test_y)
