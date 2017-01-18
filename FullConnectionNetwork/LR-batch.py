import numpy as np

X = np.linspace(-50,50,100)
y = np.mat(X>0)*1.0

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

#fc1 sigmoid fc2
#lr1 = 0.000005
lr2 = 0.0000005
inputDim = 1
fc1Dim = 2
W1 = np.random.randn(fc1Dim, inputDim)
b1 = np.zeros((fc1Dim,1))

def forward(x):
	fc1=W1.dot(x)+b1
	#sig1=sigmoid(fc1)
	sig1 = fc1
	#fc2=W2.dot(sig1)+b2
	softmax=np.exp(sig1)
	softmax=softmax/np.sum(softmax)
	return fc1,sig1,softmax

def backward(x,fc1,sig1,loss,W1,b1):
	dfc2up = loss#1xn->nx1
	#print 'dfc2up:',dfc2up.shape
	#dsigup = np.multiply(sig1,(np.add(-sig1,1.0)))#1x1
	#dsigup = np.multiply(dfc2up,dsigup)#nx1
	dsigup = dfc2up
	#print 'dsigup:',dsigup.shape#nx1
	dW1 = np.dot(dsigup,x.T)#nx1
	#print 'dw1:',dW1.shape
	db1 = dsigup#nx1
	#print 'db1:',db1.shape
	W1 -= lr2 * dW1
	b1 -= lr2 * db1
	return W1,b1

num = 1000
for i in xrange(0,num):
	#x=np.mat([3.0])
	for x in X:
		fc1,sig1,softmax = forward(x)
		#print fc1,sig1,fc2
		if x > 0:
			loss = softmax-np.mat([[0.0],[1.0]])
		else:
			loss = softmax-np.mat([[1.0],[0.0]])

		W1,b1=backward(x,fc1,sig1,loss,W1,b1)
	if i % (num/5) == 0:
			print 'class is', int(softmax.argmax()),'softmax:',softmax,'actual value is :', x >0

test = np.mat([10.0])
_,_,res = forward(test)
print res,res.argmax()," actually is : ", test[0]>0
test = np.mat([-10.0])
_,_,res = forward(test)
print res,res.argmax()," actually is : ", test[0]>0
test = np.mat([-60.0])
_,_,res = forward(test)
print res,res.argmax()," actually is : ", test[0]>0
test = np.mat([100.0])
_,_,res = forward(test)
print res,res.argmax()," actually is : ", test[0]>0