#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: anchen
# @Date:   2016-12-21 10:45:09
# @Last Modified by:   anchen
# @Last Modified time: 2016-12-21 14:48:24

import mxnet as mx
import mxnet.optimizer as opt
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

X_data = np.linspace(-100, 100, 1000)
y_data = np.zeros(X_data.shape)
ind = 0;
for x in X_data:
    if x < -50:
        y_data[ind]=0
    if x >= -50 and x < 0:
        y_data[ind]=1
    if x >= 0  and x < 50:
        y_data[ind]=2
    if x >= 50:
        y_data[ind]=3
    ind += 1

X = mx.symbol.Variable('data')
Y = mx.symbol.Variable('softmax_label')
 
#
fc1 = mx.symbol.FullyConnected(data=X, num_hidden=50, name='fc1')
sig1 = mx.symbol.Activation(data=fc1,name='sig1',act_type='relu')
#relu1 = mx.symbol.Activation(data=fc1,name='relu1',act_type='relu')
fc2 = mx.symbol.FullyConnected(data=sig1, num_hidden=4, name='fc2')
#sig2 = mx.symbol.Activation(data=fc2,name='sig2',act_type='relu')
#fc3 = mx.symbol.FullyConnected(data=sig2, num_hidden=4, name='fc3')
#sig3 = mx.symbol.Activation(data=fc3,name='sig3',act_type='relu')
#fc4 = mx.symbol.FullyConnected(data=sig3, num_hidden=4, name='fc4')
softmax = mx.symbol.SoftmaxOutput(data=fc2, label=Y, name='softmax')

batch_size = 100  
data_shape = (batch_size, 200) 
#mx.viz.plot_network(softmax, shape={"data":data_shape}, node_attrs={"shape":'oval',"fixedsize":'false'}).view()

sgd_opt = opt.SGD(learning_rate=0.0005,wd=0.0005, momentum=0.9,rescale_grad=(1.0/batch_size))#

def lr_callback(epoch, symbol, arg_params, aux_params):
    step = 100
    if epoch % step == 0:
        sgd_opt.lr = np.max(sgd_opt.lr * 0.85, 0.00001)
        #print 'epoch:%d, learning rate:%f' % (epoch, sgd_opt.lr)

model = mx.model.FeedForward(
            ctx=mx.cpu(),
            symbol=softmax,
            num_epoch=800,
            numpy_batch_size=500,
            optimizer=sgd_opt
        )

model.fit(X=X_data, y=y_data,eval_data=(X_data, y_data),
            eval_metric="acc"#,logger=logger
            )
 
#
#test = np.array([20])
#prediction = model.predict(test)
test = np.random.permutation(X_data)
prediction = model.predict(test[0:10])
for i in xrange(0, prediction.shape[0]):
    print test[i], prediction[i,:].argmax()
#print [test,prediction.argmax()]