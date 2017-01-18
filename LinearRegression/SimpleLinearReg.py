#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:23:01 2017

@author: zhouhang
"""

import numpy as np
from matplotlib import pyplot as plt 

X = np.arange(-5.,9.,0.1)
print X
X=np.random.permutation(X)
print X

b=5.
y=0.5 * X ** 2.0 +3. * X + b + np.random.random(X.shape)* 10.

#plt.scatter(X,y)
#plt.show()

#
X_ = np.mat(X).T
X_ = np.hstack((np.square(X_) , X_))
X_ = np.hstack((X_, np.mat(np.ones(len(X))).T))

A=(X_.T*X_).I*X_.T * np.mat(y).T

y_ = X_ * A

plt.hold(True)
plt.plot(X,y,'r.',fillstyle='none')
plt.plot(X,y_,'bo',fillstyle='none')
plt.show()