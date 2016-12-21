import numpy as np

X = np.linspace(-50,50,100)
y = X * 2.0 + 5.0

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

#fc1 sigmoid fc2
lr1 = 0.000005
lr2 = 0.000005
inputDim = 1
fc1Dim = 40
fc2Dim = 1
W1 = np.random.randn(fc1Dim, inputDim)
b1 = np.zeros((fc1Dim,1))
W2 = np.random.randn(fc2Dim, fc1Dim)
b2 = np.zeros((fc2Dim,1)) 

def forward(x):
	fc1=W1.dot(x)+b1
	#sig1=sigmoid(fc1)
	sig1 = fc1
	fc2=W2.dot(sig1)+b2
	return fc1,sig1,fc2

def backward(x,fc1,sig1,fc2,loss,W1,b1,W2,b2):
	dW2 = np.dot(loss,sig1.T)#nx1->1xn
	#print dW2
	#print 'dw2:',dW2.shape
	db2 = loss.T
	#print 'db2:',db2.shape
	dfc2up = np.dot(W2.T,loss)#1xn->nx1
	#print 'dfc2up:',dfc2up.shape
	#dsigup = np.multiply(sig1,(np.add(-sig1,1.0)))#1x1
	#dsigup = np.multiply(dfc2up,dsigup)#nx1
	dsigup = dfc2up
	#print 'dsigup:',dsigup.shape#nx1
	dW1 = np.dot(dsigup,x.T)#nx1
	#print 'dw1:',dW1.shape
	db1 = dsigup#nx1
	#print 'db1:',db1.shape
	W2 -= lr1 * dW2
	b2 -= lr1 * db2
	W1 -= lr2 * dW1
	b1 -= lr2 * db1
	return W1,b1,W2,b2

num = 1000
for i in xrange(0,num):
	#x=np.mat([3.0])
	for x in X:
		fc1,sig1,fc2 = forward(x)
		#print fc1,sig1,fc2
		W1,b1,W2,b2=backward(x,fc1,sig1,fc2,fc2-x*2.0-5.0,W1,b1,W2,b2)
	if i % (num/5) == 0:
			print 'fc2:',fc2,'true value is :', x * 2.0 + 5.0

test = np.mat([-65.0])
_,_,res = forward(test)
print res," true is : ", test[0]*2.0+5.0