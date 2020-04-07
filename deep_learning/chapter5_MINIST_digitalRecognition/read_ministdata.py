# coding: utf-8

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


def main():
    minist = input_data.read_data_sets('D:/PycharmProjects/datasets/minist', one_hot=True)
    print('Training data size: ', minist.train.num_examples)
    print('Validating data size: ', minist.validation.num_examples)
    print('Test data size: ', minist.test.num_examples)

    # print('Example training data: ', minist.train.images[0])
    print('Example training label: ', minist.train.labels[0])

    print(minist.train.images.shape)
    print(len(minist.train.images[:, 0]))
    img_data = np.reshape(minist.train.images[0], (28, 28, 1))
    # print(img_data)


    # plt.imshow(img_data)
    # plt.show()


if __name__ == '__main__':
    main()
