import os
import matplotlib.pyplot as plt
import numpy as np
import LineaRegression

reger = LineaRegression.LinearRegression()
eleNum = 4#x3,x2,x1,1
trainingData = reger.GenerateQuadraticData((1,4,4), 100, (-10, 10), eleNum)
print "hehe"
regPoly, regData = reger.QuadraticLinearRegression(trainingData, eleNum)
print regPoly
plt.plot(trainingData[:, eleNum - 1], trainingData[:, 0], 'r.')
plt.plot(regData[:, 1], regData[:, 0], 'bx')
plt.show()
