import numpy as np
import random
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        pass

    def ExponentLinearRegressionDemo(self, eleNum, trueQuadraticPoly, trainSampleNum, trainInterval, sigma):
        trainingData = self._GenerateExponentData(trueQuadraticPoly, \
                                                  trainSampleNum, trainInterval, sigma)
        regPoly = self.ExponentLinearRegression(trainingData, eleNum)
        regData = np.zeros((trainInterval[1] - trainInterval[0]+1, 2), trainingData.dtype)
        for i in xrange(trainInterval[0], trainInterval[1] + 1):
            regData[i - trainInterval[0], 1] = i
            for k in xrange(1, eleNum + 1):
                regData[i - trainInterval[0], 0] += regPoly[k - 1] * i ** (eleNum - k)
        print regPoly
        trueData = np.zeros((trainInterval[1] - trainInterval[0]+1, 2), trainingData.dtype)
        for i in xrange(trainInterval[0], trainInterval[1]+1):
            trueData[i - trainInterval[0], 0] = i
            y = 0
            for k in xrange(0, len(trueQuadraticPoly)):
                y += trueQuadraticPoly[k] * i ** (len(trueQuadraticPoly) - k - 1)
            trueData[i - trainInterval[0], 1] = y
        plt.plot(trainingData[:, 1], trainingData[:, 0], 'ro')
        plt.plot(trueData[:, 0], trueData[:, 1], 'r-')
        plt.plot(regData[:, 1], regData[:, 0], 'b-')
        plt.show()

    def RegularExponentLinearRegressionDemo(self, eleNum, trueQuadraticPoly, trainSampleNum, trainInterval, sigma, regValue):
        trainingData = self._GenerateExponentData(trueQuadraticPoly, \
                                                  trainSampleNum, trainInterval, sigma)
        regPoly = self.RegularExponentLinearRegression(trainingData, eleNum, regValue)
        regData = np.zeros((trainInterval[1] - trainInterval[0]+1, 2), trainingData.dtype)
        for i in xrange(trainInterval[0], trainInterval[1] + 1):
            regData[i - trainInterval[0], 1] = i
            for k in xrange(1, eleNum + 1):
                regData[i - trainInterval[0], 0] += regPoly[k - 1] * i ** (eleNum - k)
        print regPoly
        trueData = np.zeros((trainInterval[1] - trainInterval[0] + 1, 2), trainingData.dtype)
        for i in xrange(trainInterval[0], trainInterval[1] + 1):
            trueData[i - trainInterval[0], 0] = i
            y = 0
            for k in xrange(0, len(trueQuadraticPoly)):
                y += trueQuadraticPoly[k] * i ** (len(trueQuadraticPoly) - k - 1)
            trueData[i - trainInterval[0], 1] = y
        noRegPoly = self.ExponentLinearRegression(trainingData, eleNum)
        noRegData = np.zeros((trainInterval[1] - trainInterval[0]+1, 2), trainingData.dtype)
        for i in xrange(trainInterval[0], trainInterval[1] + 1):
            noRegData[i - trainInterval[0], 1] = i
            for k in xrange(1, eleNum + 1):
                noRegData[i - trainInterval[0], 0] += noRegPoly[k - 1] * i ** (eleNum - k)
        print noRegPoly
        plt.plot(trainingData[:, 1], trainingData[:, 0], 'ro')
        plt.plot(trueData[:, 0], trueData[:, 1], 'r-')
        plt.plot(noRegData[:, 1], noRegData[:, 0], 'b-')
        plt.plot(regData[:, 1], regData[:, 0], 'g-')
        plt.show()

    def _GenerateGaussianDistribute(self, sigma, x):
        res = np.exp(- x ** x / (2 * sigma ** sigma))
        return res

    def _GenerateExponentData(self, polyData, num, interval, sigma):
        trainData = np.zeros((num, 2), np.float32)
        length = interval[1] - interval[0]
        for i in xrange(0, num):
            x = random.random() * length + interval[0]
            trainData[i, 1] = x
            y = 0
            for k in xrange(0, len(polyData) ):
                y += polyData[k] * x ** (len(polyData) - k - 1)
            y += random.gauss(0,sigma) * length
            trainData[i, 0] = y

        return trainData

    def ExponentLinearRegression(self, trainingData, eleNum):
        X = np.zeros((trainingData.shape[0], eleNum), trainingData.dtype)
        for i in xrange(0, trainingData.shape[0]):
            for k in xrange(0, eleNum):
                X[i, k] = trainingData[i, 1] ** (eleNum - k - 1)
        X = np.mat(X)#X is 100x4
        B = np.mat(trainingData[:, 0])#B is converted into col vector, 1x100
        B = B.T
        regPolyData = X.T * X
        regPolyData = regPolyData.I
        regPolyData = regPolyData * X.T
        regPolyData = regPolyData * B
        return regPolyData

    def RegularExponentLinearRegression(self, trainingData, eleNum, regValue):
        X = np.zeros((trainingData.shape[0], eleNum), trainingData.dtype)
        for i in xrange(0, trainingData.shape[0]):
            for k in xrange(0, eleNum):
                X[i, k] = trainingData[i, 1] ** (eleNum - k - 1)
        X = np.mat(X)  # X is 100x4
        B = np.mat(trainingData[:, 0])  # B is converted into col vector, 1x100
        B = B.T
        regPolyData = X.T * X
        regPolyData += regValue * np.eye(regPolyData.shape[0])
        regPolyData = regPolyData.I
        regPolyData = regPolyData * X.T
        regPolyData = regPolyData * B
        return regPolyData

