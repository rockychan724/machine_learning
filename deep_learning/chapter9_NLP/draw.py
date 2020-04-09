# -*- coding: utf-8 -*-
# @Time : 2020/4/9 12:03


def main():
    a = [2, 5, 2, 7, 9]
    b = sorted(a)
    print(b)
    b.append(3)
    print(a)
    print(b)
    b += [10]
    print(b)


if __name__ == '__main__':
    main()
