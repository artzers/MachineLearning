#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:17:08 2017

@author: zhouhang
"""

from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt 

X = np.arange(-5.,9.,0.1)
X=np.random.permutation(X)
X_=[[i] for i in X]
#print X
b=5.
y=0.5 * X ** 2.0 +3. * X + b + np.random.random(X.shape)* 10.
y_=[i for i in y]

rbf1=svm.SVR(kernel='rbf',C=1, )#degree=2,,gamma=, coef0=
rbf2=svm.SVR(kernel='rbf',C=20, )#degree=2,,gamma=, coef0=
poly=svm.SVR(kernel='poly',C=1,degree=2)
rbf1.fit(X_,y_)
rbf2.fit(X_,y_)
poly.fit(X_,y_)
result1 = rbf1.predict(X_)
result2 = rbf2.predict(X_)
result3 = poly.predict(X_)
plt.hold(True)
plt.plot(X,y,'bo',fillstyle='none')
plt.plot(X,result1,'r.')
plt.plot(X,result2,'g.')
plt.plot(X,result3,'c.')
plt.show()

#X = [[0, 0], [1, 1], [1, 0]]  # training samples   
#y = [0, 1, 1]  # training target  
#clf = svm.SVC()  # class   
#clf.fit(X, y)  # training the svc model  
#  
#result = clf.predict([2, 2]) # predict the target of testing samples   
#print result  # target   
#  
#print clf.support_vectors_  #support vectors  
#  
#print clf.support_  # indeices of support vectors  
#  
#print clf.n_support_  # number of support vectors for each class 

