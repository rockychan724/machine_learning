# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

INPUT_DATA = '../flower_data_processed.npy'
INCEPTION_V3 = './inception_v3_model/inception_v3.ckpt'
MODEL_PATH = './model/model'

# 定义训练中的参数
LEARNING_RATE = 0.0001
TRAINING_STEPS = 300
BATCH_SIZE = 32
NUM_CLASSES = 5

# CHECKPOINT_EXCLUDE_SCOPES：不需要从inception_v3模型中加载的参数的前缀。这些参数是最后的全连接层，在新的问题中将重新训练这些参数
# TRAINABLE_SCOPES：需要训练的参数的前缀，在重新训练的过程中，这些参数就是最后的全连接层的参数
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'


def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []

    for var in slim.get_model_variables():
        exclude = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                exclude = True
                break
        if not exclude:
            variables_to_restore.append(var)
    return variables_to_restore


def get_trainable_variables():
    trainable_scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []

    for scope in trainable_scopes:
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(var)
    return variables_to_train


def main():
    processed_data = np.load(INPUT_DATA, allow_pickle=True)
    train_images = processed_data[0]
    train_labels = processed_data[1]
    validate_images = processed_data[2]
    validate_labels = processed_data[3]
    test_images = processed_data[4]
    test_labels = processed_data[5]
    print('There are %d traing examples, %d validation examples, and %d testing examples.' % (
        len(train_images), len(validate_images), len(test_images)))

    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='output_labels')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(inputs=images, num_classes=NUM_CLASSES)  # 查看logits, _的大小

    # 获取需要训练的变量
    trainable_variables = get_trainable_variables()
    # 定义交叉熵损失
    tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, NUM_CLASSES), logits=logits)
    total_loss = tf.losses.get_total_loss()
    # 定义训练过程
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(total_loss)

    # 计算正确率
    with tf.name_scope('evaluation'):
        corrent_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))

    # 定义加载模型的函数
    load_func = slim.assign_from_checkpoint_fn(INCEPTION_V3, get_tuned_variables(), ignore_missing_vars=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Loading tuned variables from ', INCEPTION_V3)
        load_func(sess)

        start = 0
        end = start + BATCH_SIZE
        for i in range(TRAINING_STEPS):
            _, loss = sess.run([train_step, total_loss],
                               feed_dict={images: train_images[start:end], labels: train_labels[start:end]})
            if i % 30 == 0 or i + 1 == TRAINING_STEPS:
                saver.save(sess, MODEL_PATH, global_step=i)
                validation_accuracy = sess.run(accuracy, feed_dict={images: validate_images, labels: validate_labels})
                print('Step %d, validation accuracy is %f' % (i, validation_accuracy))
            start = end
            if start == len(train_images):
                start = 0
            end = min(start + BATCH_SIZE, len(train_images))

        test_accuracy = sess.run(accuracy, feed_dict={images: test_images, labels: test_labels})
        print('Finnaly, test accuracy is ', test_accuracy)


if __name__ == '__main__':
    main()
