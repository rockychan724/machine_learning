import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_mean(img, i, j):
    ran = [-1, 0, 1]
    sum = 0
    k = []
    for m in ran:
        for n in ran:
            sum += img[i+m, j+n]
            k.append(img[i+m, j+n])
    k.sort()
    return sum/9
    # return k[4]


def Robert(img, i, j):
    return (abs(int(img[i+1, j+1]) - int(img[i, j])) + abs(int(img[i, j+1]) - int(img[i+1, j]))) % 255


def Laplacian(img, i, j):
    suanzi = (int(img[i+1, j]) + int(img[i-1, j]) + int(img[i, j+1]) + int(img[i, j-1])) - 4*int(img[i, j])
    if suanzi < 0:
        result = img[i, j] - suanzi
    else:
        result = img[i, j] + suanzi
    return result


if __name__ == '__main__':
    picture = 'E:/Picture/imageRecognition/animal1_gray.png'
    img = cv2.imread(picture, 0)
    a = img.shape
    h = a[0]
    w = a[1]
    cv2.imshow('preview', img)

    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_img', gray_img)
    #
    # cv2.imwrite('E:/Picture/imageRecognition/animal1_gray.png', gray_img)

    print(type(img))
    print(type(img[0, 0]))
    new_img = img
    for i in range(1, h-1):
        for j in range(1, w-1):
            # new_img[i, j] = get_mean(img, i, j)s
            # new_img[i, j] = Robert(img, i, j)
            new_img[i, j] = Laplacian(img, i, j)
    cv2.imshow('change', new_img)

    # for i in range(0, h):
    #     for j in range(0, w):
    #         img[i, j, 0] = int(img[i, j, 0] * 0.3)
    #         img[i, j, 1] = int(img[i, j, 1] * 0.3)
    #         img[i, j, 2] = int(img[i, j, 2] * 0.3)
    # cv2.imshow('img1', img)

    # for i in range(0, h):
    #     for j in range(0, w):
    #         img[i, j, 0] = int(img[i, j, 0] * 10)
    #         img[i, j, 1] = int(img[i, j, 1] * 10)
    #         img[i, j, 2] = int(img[i, j, 2] * 10)
    # cv2.imshow('img2', img)

    # test_img = np.zeros((200, 300, 3))
    # for i in range(200):
    #     for j in range(300):
    #         if i == 100:
    #             test_img[i, j, 0] = 255
    #             test_img[i, j, 1] = 255
    #             test_img[i, j, 2] = 255
    #         else:
    #             test_img[i, j, 2] = 255
    # cv2.imshow('red', test_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # img = np.zeros((200, 300, 3))
    # rand1 = np.random.randint(200, size=1000)
    # rand2 = np.random.randint(300, size=1000)
    # for i in range(1000):
    #     img[rand1[i],  g[i], 0] = np.random.randint(0, 255)
    #     img[rand1[i], rand2[i], 1] = np.random.randint(0, 255)
    #     img[rand1[i], rand2[i], 2] = np.random.randint(0, 255)
    # cv2.imshow('preview', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

# x = np.arange(0, 5, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
# print(y)
# plt.show()


# a = np.array([[1,2],[2,4]])
# print(a)
# b = np.arange(15)
# c = np.array([3,4,5])
# print(b, c)
# print(type(a),type(b),type(c))
# d = [[1,2],[2,4]]

# print(d)
