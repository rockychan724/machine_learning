# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def test_LSTM():
    lstm = tf.nn.rnn_cell.BasicLSTMCell()


def main():
    x = [1, 2]
    state = [0.0, 0.0]

    w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    w_cell_input = np.asarray([0.5, 0.6])
    b_cell = np.asarray([0.1, -0.1])

    w_output = np.asarray([[1.0], [2.0]])
    b_output = np.asarray([0.1])

    for i in range(len(x)):
        tmp1 = np.dot(state, w_cell_state)
        tmp2 = x[i] * w_cell_input
        temp_state = np.dot(state, w_cell_state) + x[i] * w_cell_input + b_cell
        state = np.tanh(temp_state)
        output = np.dot(state, w_output) + b_output
        print('*** i = ', i)
        print('\ttmp1 = ', tmp1)
        print('\ttmp2 = ', tmp2)
        print('\tstate = ', state)
        print('\toutput = ', output)


if __name__ == '__main__':
    main()
