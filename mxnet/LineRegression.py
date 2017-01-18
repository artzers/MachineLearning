import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
 
#
#X_data = np.linspace(-1, 1, 100)
#noise = np.random.normal(0, 0.5, 100)
#y_data = 5 * X_data + noise

X = np.arange(-5.,9.,0.1)
X=np.random.permutation(X)
#X_=[[i] for i in X]
X_=X
b=5.
y=0.5 * X ** 2.0 +3. * X + b + np.random.random(X.shape)* 10.
y_=y
#y_=[i for i in y]
 
#

 
#
X = mx.symbol.Variable('data')
Y = mx.symbol.Variable('softmax_label')
 
#
fc1 = mx.symbol.FullyConnected(data=X, num_hidden=4, name='fc1')
ac1 = mx.symbol.Activation(data=fc1,name='ac1',act_type='sigmoid')
fc2 = mx.symbol.FullyConnected(data=ac1, num_hidden=4, name='fc2')
fc3 = mx.symbol.FullyConnected(data=fc2, num_hidden=1, name='fc3')
loss = mx.symbol.LinearRegressionOutput(data=fc3, label=Y, name='loss')
 
#
model = mx.model.FeedForward(
            ctx=mx.cpu(),
            symbol=loss,
            num_epoch=600,
            learning_rate=0.0005,
            numpy_batch_size=1
        )
 
#
train_data = mx.io.NDArrayIter(data=X_, label=y_, batch_size=30)
#eval_data = mx.io.NDArrayIter(data=X_, batch_size=10, shuffle=True)
model.fit(X=train_data)
 
#
prediction = model.predict(X_)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_, y_,'ro')
lines = ax.plot(X_, prediction, 'b.', lw=5)
plt.show()