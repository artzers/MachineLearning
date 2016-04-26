import numpy as np
import random
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        pass

    def LinearRegressionDemo(self, eleNum, trueQuadraticPoly, trainSampleNum, trainInterval):
        trainingData = self._GenerateQuadraticData(trueQuadraticPoly, \
                                                    trainSampleNum, trainInterval, eleNum)
        regPoly, regData = self.QuadraticLinearRegression(trainingData, eleNum)
        print regPoly
        trueData = np.zeros((trainInterval[1] - trainInterval[0]+1, 2), trainingData.dtype)
        for i in xrange(trainInterval[0], trainInterval[1]+1):
            trueData[i - trainInterval[0], 0] = i
            y = 0
            for k in xrange(0, len(trueQuadraticPoly)):
                y += trueQuadraticPoly[k] * i ** (len(trueQuadraticPoly) - k - 1)
            trueData[i - trainInterval[0], 1] = y
        plt.plot(trainingData[:, eleNum - 1], trainingData[:, 0], 'ro')
        plt.plot(trueData[:, 0], trueData[:, 1], 'r-')
        plt.plot(regData[:, 1], regData[:, 0], 'bx')
        plt.show()

    def _GenerateGaussianDistribute(self, sigma, x):
        res = np.exp(- x ** x / (2 * sigma ** sigma))
        return res

    def _GenerateQuadraticData(self, polyData, num, interval, eleNum):
        trainData = np.zeros((num, eleNum + 1), np.float32)
        length = interval[1] - interval[0]
        for i in xrange(0, num):
            x = random.random() * length + interval[0]
            y = 0
            for k in xrange(0, len(polyData) ):
                y += polyData[k] * x ** (len(polyData) - k - 1)
            y += random.gauss(0,0.5) * length
            trainData[i, 0] = y
            for k in xrange(1, eleNum + 1):
                trainData[i, k] = x ** (eleNum - k)
                #trainData[i, k] = x ** 2
                #trainData[i, 3] = x
                #trainData[i, 4] = 1

        return trainData

    def QuadraticLinearRegression(self, trainingData, eleNum):
        X = np.mat(trainingData[:, 1:eleNum+1])#X is 100x4
        B = np.mat(trainingData[:, 0])#B is converted into col vector, 1x100
        B = B.T
        regPolyData = X.T * X
        regPolyData = regPolyData.I
        regPolyData = regPolyData * X.T
        regPolyData = regPolyData * B
        regData = np.zeros((trainingData.shape[0], 2), trainingData.dtype)
        for i in xrange(0, trainingData.shape[0]):
            x = trainingData[i, trainingData.shape[1]-2]
            regData[i, 1] = x
            for k in xrange(1, eleNum + 1):
                regData[i, 0] += regPolyData[k-1] * x ** (eleNum - k)
        return regPolyData, regData
