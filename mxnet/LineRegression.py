import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
 
#
X_data = np.linspace(-1, 1, 100)
noise = np.random.normal(0, 0.5, 100)
y_data = 5 * X_data + noise
 
#
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X_data, y_data)
 
#
X = mx.symbol.Variable('data')
Y = mx.symbol.Variable('softmax_label')
 
#
Y_ = mx.symbol.FullyConnected(data=X, num_hidden=1, name='pre')
loss = mx.symbol.LinearRegressionOutput(data=Y_, label=Y, name='loss')
 
#
model = mx.model.FeedForward(
            ctx=mx.gpu(),
            symbol=loss,
            num_epoch=100,
            learning_rate=0.001,
            numpy_batch_size=1
        )
 
#
model.fit(X=X_data, y=y_data)
 
#
prediction = model.predict(X_data)
lines = ax.plot(X_data, prediction, 'r-', lw=5)
plt.show()