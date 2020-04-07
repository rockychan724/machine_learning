# coding: utf-8
import tensorflow as tf
import sys
import py2

def main():
    print('From py1')
    py2.test('Jack!')
    print(sys.path)

if __name__ == "__main__":
    main()