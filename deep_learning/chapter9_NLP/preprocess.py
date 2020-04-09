# -*- coding: utf-8 -*-
# @Time : 2020/4/9 11:49

import os
import collections
import operator


def get_vocab():
    RAW_DATA = 'E:/Documents/PycharmProjects/datasets/PTB_data/ptb.train.txt'
    VOCAB_OUTPUT = './data/ptb.vocab'

    counter = collections.Counter()
    with open(RAW_DATA, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            for word in line.strip().split():
                counter[word] += 1
    # print(type(counter))
    # print(type(counter[0]))
    # print(type(counter.items()))
    # for i, item in enumerate(counter.items()):
    #     if i == 0:
    #         print(type(item))
    #     print(item, end=',')
    sorted_words_ = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
    sorted_words = [w[0] for w in sorted_words_]
    sorted_words = ['<eos>'] + sorted_words
    with open(VOCAB_OUTPUT, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(word + '\n')


def generate_id():
    def get_id(word):
        return word_id[word] if word in word_id else word_id['<unk>']

    modes = ['train', 'valid', 'test']
    mode = modes[0]
    VOCAB = './data/ptb.vocab'
    RAW_DATA = 'E:/Documents/PycharmProjects/datasets/PTB_data/ptb.' + mode + '.txt'
    OUTPUT_DATA = './data/ptb.' + mode

    with open(VOCAB, mode='r', encoding='utf-8') as f:
        vocab = [w.strip() for w in f.readlines()]
    word_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    num = 0
    with open(RAW_DATA, mode='r', encoding='utf-8') as fin, open(OUTPUT_DATA, mode='w', encoding='utf-8') as fout:
        for line in fin.readlines():
            words = line.strip().split() + ['<eos>']
            ids = [get_id(w) for w in words]
            out_line = ' '.join(str(id) for id in ids)
            fout.write(out_line + '\n')
            num += len(words)
    print(num)


def main():
    generate_id()


if __name__ == '__main__':
    main()
