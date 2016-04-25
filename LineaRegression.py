import numpy as np
import random

class LinearRegression:
    def __init__(self):
        pass

    def GenerateGaussianDistribute(self, sigma, x):
        res = np.exp(- x ** x / (2 * sigma ** sigma))
        return res

    def GenerateQuadraticData(self, polyData, num, interval, eleNum):
        trainData = np.zeros((num, eleNum + 1), np.float32)
        length = interval[1] - interval[0]
        for i in xrange(0, num):
            x = random.random() * length + interval[0]
            y = polyData[0] * x ** 2 + polyData[1] * x + polyData[2] + random.gauss(0,0.5) * length
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
