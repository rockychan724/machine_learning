# coding: utf-8

import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import minist_inference
import minist_train

# 每10秒加载一次最新的模型进行评测
EVAL_INTERVAL_SECS = 10


def evaluate(minist):  # 需要加载图操作吗？
    x = tf.placeholder(tf.float32, shape=[None, minist_inference.IMAGE_SIZE, minist_inference.IMAGE_SIZE,
                                          minist_inference.NUM_CHANNELS], name='x_input')
    y_ = tf.placeholder(tf.float32, shape=[None, minist_inference.OUTPUT_NODE], name='y_output')
    x_feed = minist.validation.images  # shape = (5000, 784)
    x_feed = np.reshape(x_feed,
                        (len(x_feed[:, 0]), minist_inference.IMAGE_SIZE, minist_inference.IMAGE_SIZE,
                         minist_inference.NUM_CHANNELS))
    validation_feed = {x: x_feed, y_: minist.validation.labels}

    y = minist_inference.inference(x, None, False)
    correct_preditction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))  # 传进去的第二个参数实际上是y的滑动平均值，y是影子变量的重命名
    accuracy = tf.reduce_mean(tf.cast(correct_preditction, tf.float32))

    # 将影子变量重命名
    ema = tf.train.ExponentialMovingAverage(minist_train.MOVING_AVERAGE_DECAY)
    saver = tf.train.Saver(ema.variables_to_restore())
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(minist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_value, y1 = sess.run([accuracy, y], feed_dict=validation_feed)
                # print('y: ', y)
                # print('y = ', y1)
                print('After %s training steps, validation accuracy is %f' % (global_step, accuracy_value))
            else:
                print('There is no checkpoint files!')
                return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    minist = input_data.read_data_sets('D:\\PycharmProjects\\datasets\\minist', one_hot=True)
    evaluate(minist)


if __name__ == '__main__':
    tf.app.run()
