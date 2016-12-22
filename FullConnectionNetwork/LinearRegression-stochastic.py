import numpy as np

X = np.mat(np.linspace(-50,50,100)).T
y = X * 2.0 + 5.0
trainData=np.column_stack((X,y))

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

#fc1 sigmoid fc2
lr1 = 0.000005
lr2 = 0.000005
inputDim = 1
fc1Dim = 20
fc2Dim = 1
W1 = np.random.randn(inputDim,fc1Dim)
b1 = np.zeros((1,fc1Dim))
W2 = np.random.randn(fc1Dim, fc2Dim)
b2 = np.zeros((1,1)) 

def forward(x):
	fc1=np.dot(x,W1)+b1
	#sig1=sigmoid(fc1)
	sig1 = fc1
	fc2=np.dot(sig1,W2)+b2
	return fc1,sig1,fc2

def backward(x,fc1,sig1,fc2,loss,W1,b1,W2,b2):
	dW2 = np.dot(sig1.T,loss)#nx1->1xn
	db2 = loss
	dfc2up = np.dot(loss,W2.T)#1xn->nx1
	dsigup = dfc2up
	dW1 = np.dot(x.T,dsigup)#nx1
	db1 = dsigup#nx1
	W2 -= lr1 * dW2
	b2 -= lr1 * db2
	W1 -= lr2 * dW1
	b1 -= lr2 * db1
	return W1,b1,W2,b2

def predict(data):
	res=np.zeros(data.shape)
	ind = 0
	for i in data:
		_,_,res[ind]=forward(i)
		ind+=1
	return res

def BuildStochastic(batchSize):
	batch = np.random.permutation(trainData)[0:batchSize,:]
	return batch

num = 1000
for i in xrange(0,num):
	#x=np.mat([3.0])
	batch = BuildStochastic(10)
	loss = 0
	for x in batch:
		fc1,sig1,fc2 = forward(x[0])
		loss = fc2 - x[1]
		W1,b1,W2,b2=backward(x[0],fc1,sig1,fc2,loss,W1,b1,W2,b2)

	if i % (num/5) == 0:
			print 'fc2:',fc2,'true value is :', x[1]

test = np.mat([-65.0,50,25])
test=test.T
res = predict(test)
print res
print " true is : ", test*2.0+5.0