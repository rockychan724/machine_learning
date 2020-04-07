# -*- coding: utf-8 -*-
# content: linear classifier

import numpy as np
import matplotlib.pyplot as plt
import math


class LinearClassifier:
    def __init__(self):
        self.bias = 1
        self.learning0 = 0.1  # 初始学习率
        self.learning = 0.0
        self.tao = 50  # 时间常数τ
        self.expectError = 0.9
        self.maxTrainCount = 500
        self.x = np.array(
            [[9, 25, 1], [5, 8, 1], [15, 31, 1], [35, 62, 1], [19, 40, 1], [28, 65, 1], [20, 59, 1], [9, 41, 1],
             [12, 60, 1], [2, 37, 1]])
        self.d = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
        self.w = np.array([0.0, 0.0, self.bias])
        self.testPoint = []

    def sgn(self, v):
        if v > 0:
            return 1
        else:
            return 0

    def getY(self, w, xi):
        return self.sgn(np.dot(w.T, xi))

    def nextW(self, w, xi, di, trainCount):
        error = di - self.getY(w, xi)
        self.learning = self.learning0 / (1 + trainCount / self.tao)
        return w + self.learning * error * xi, error

    def train(self):
        trainCount = 0
        while True:
            mse = 0.0
            for i in range(len(self.x)):
                self.w, error = self.nextW(self.w, self.x[i], self.d[i], trainCount)
                mse += error ** 2
            mse /= len(self.x)
            # mse = math.sqrt(mse)
            trainCount += 1
            print('第%d次调整，误差为：%f' % (trainCount, mse))
            if abs(mse) < self.expectError or trainCount >= self.maxTrainCount:
                if trainCount >= self.maxTrainCount:
                    print('训练已达最大次数！')
                break
        print('%dx + %dy + %d = 0' % (self.w[0], self.w[1], self.w[2]))

    def classify(self, testData):
        return self.getY(self.w, np.array(testData + [1]))

    def getTestY(self, x):
        return -(self.w[0] * x + self.w[2]) / self.w[1]

    def addTestPoint(self, testData):
        self.testPoint.append(testData)

    def draw(self):
        rawX = []
        rawY = []
        testX = []
        testY = []
        for i in range(len(self.x)):
            rawX.append(self.x[i][0])
            rawY.append(self.x[i][1])
            if (self.d[i] > 0):
                plt.plot(rawX[i], rawY[i], 'or')
            else:
                plt.plot(rawX[i], rawY[i], 'og')
        for i in range(len(self.testPoint)):
            testX.append(self.testPoint[i][0])
            testY.append(self.testPoint[i][1])
            if self.classify(self.testPoint[i]) > 0:
                plt.plot(testX[i], testY[i], '*r')
            else:
                plt.plot(testX[i], testY[i], '*g')
        xMax = max(max(rawX), max(testX)) + 2
        xMin = min(min(rawX), min(testX)) - 2
        yMax = max(max(rawY), max(testY)) + 2
        yMin = min(min(rawY), min(testY)) - 2
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
        plt.title('Linear Classifier')
        lineX = [xMin + 2, xMax - 2]
        lineY = []
        lineY.append(self.getTestY(lineX[0]))
        lineY.append(self.getTestY(lineX[1]))
        print('(%f,%f), (%f,%f)' % (lineX[0], lineY[0], lineX[1], lineY[1]))
        plt.plot(lineX, lineY, 'b--')
        plt.show()


if __name__ == '__main__':
    linearClassifier = LinearClassifier()
    linearClassifier.train()
    linearClassifier.addTestPoint([35,20])
    linearClassifier.addTestPoint([35,100])
    linearClassifier.draw()
