# coding: utf-8

import os
import glob
import numpy as np
import tensorflow as tf

INPUT_DATA = './flower_photos/'
OUTPUT_FILE = './flower_data_processed.npy'

VALIDATE_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def create_image_list(sess, validate_percentage, test_percentage):
    # 获取INPUT_DATA目录下的所有子目录（包括根目录），将子目录名称存入sub_dirs列表，
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    train_images = []
    train_labels = []
    validate_images = []
    validate_labels = []
    test_images = []
    test_labels = []
    current_label = 0

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        image_classes = ['jpg', 'jpeg', 'JPG', 'JPEG']  # 定义图片类型
        file_list = []
        # 获取当前目录下所有图片的文件名
        for image_class in image_classes:
            file_name_format = os.path.join(sub_dir, '*.' + image_class)
            # print(file_name)
            file_list.extend(glob.glob(file_name_format))
        if not file_list:
            continue

        print('Current dir: {0}, num of images: {1}'.format(sub_dir, file_list.__len__()))
        i = 0
        # 读取图片数据
        for file_name in file_list:
            image_data_raw = tf.gfile.FastGFile(file_name, 'rb').read()
            image_data = tf.image.decode_jpeg(image_data_raw)
            # 将图像数据转化为浮点型
            if image_data.dtype is not tf.float32:
                image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
            image_data = tf.image.resize_images(image_data, [299, 299])
            image_value = sess.run(image_data)  # 获取张量的值，方便后续存储

            # 随机划分数据集
            rand = np.random.randint(100)
            if rand < validate_percentage:
                validate_images.append(image_value)  # 注意append()和extend()方法的区别
                validate_labels.append(current_label)
            elif rand < (validate_percentage + test_percentage):
                test_images.append(image_value)
                test_labels.append(current_label)
            else:
                train_images.append(image_value)
                train_labels.append(current_label)
            i += 1
            if i % 200 == 0:
                print('%d images processed!' % i)
        current_label += 1

        # 将训练数据随机打乱以获得更好的训练效果
        state = np.random.get_state()
        np.random.shuffle(train_images)
        np.random.set_state(state)
        np.random.shuffle(train_labels)

        return np.asarray([train_images, train_labels, validate_images, validate_labels, test_images, test_labels])


def main():
    with tf.Session() as sess:
        processed_data = create_image_list(sess, VALIDATE_PERCENTAGE, TEST_PERCENTAGE)
        np.save(OUTPUT_FILE, processed_data)


if __name__ == '__main__':
    main()
