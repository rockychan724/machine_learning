# -*- coding: utf-8 *--

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import inference, train

# 每10秒加载一次最新的模型进行评测
EVAL_INTERVAL_SECS = 10


def evaluate(minist): # 需要加载图操作吗？
    x = tf.placeholder(tf.float32, shape=[None, inference.INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, shape=[None, inference.OUTPUT_NODE], name='y_output')
    validation_feed = {x: minist.validation.images, y_: minist.validation.labels}

    y = inference.inference(x, None)
    correct_preditction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1)) # 传进去的第二个参数实际上是y的滑动平均值，y是影子变量的重命名
    accuracy = tf.reduce_mean(tf.cast(correct_preditction, tf.float32))

    # 将影子变量重命名
    ema = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
    saver = tf.train.Saver(ema.variables_to_restore())
    # saver = tf.train.Saver()
    # saver = tf.train.Saver({'layer2/add/ExponentialMovingAverage': y})
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_value, y1 = sess.run([accuracy, y], feed_dict=validation_feed)
                print('y: ', y)
                print('y = ', y1)
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
