# coding: utf-8

import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    img_raw_data = tf.gfile.FastGFile('D:/Picture/imageRecognition/car.png', 'rb').read()
    img_data = tf.image.decode_png(img_raw_data)
    img_size = img_data.shape
    with tf.Session() as sess:
        print(sess.run(img_data))
        print(img_size)
        print(img_data)
        plt.imshow(sess.run(img_data))
        plt.show()
        img_encode = tf.image.encode_png(img_data)
        tf.gfile.FastGFile('./output.png', 'wb').write(img_encode.eval())
