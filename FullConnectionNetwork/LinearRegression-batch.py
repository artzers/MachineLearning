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
batchSize = 10
W1 = np.random.randn(inputDim,fc1Dim)#1x20
b1 = np.zeros((batchSize,fc1Dim))#1x20
W2 = np.random.randn(fc1Dim, fc2Dim)#20x1
b2 = np.zeros((batchSize,fc2Dim)) #10x1

def forward(x):
	fc1=np.dot(x,W1)+b1
	#sig1=sigmoid(fc1)
	sig1 = fc1
	fc2=np.dot(sig1,W2)+b2
	return fc1,sig1,fc2

def backward(x,fc1,sig1,fc2,loss,W1,b1,W2,b2):
	dW2 = np.dot(sig1.T,loss)#nx1->1xn
	#print dW2.shape
	db2 = loss
	dfc2up = np.dot(loss,W2.T)#1xn->nx1
	dsigup = dfc2up
	dW1 = np.dot(x.T,dsigup)#nx1
	db1 = dsigup#nx1
	#print 'db1',db1.shape
	W2 -= lr1 * dW2
	b2 -= lr1 * db2
	W1 -= lr2 * dW1
	b1 -= lr2 * db1
	return W1,b1,W2,b2

def predict(data):
	_,_,res=forward(data)
	return res

def BuildBatch(batchSize):
	batch = np.random.permutation(trainData)[0:batchSize,:]
	return batch

num = 10000
for i in xrange(0,num):
	#x=np.mat([3.0])
	batch = BuildBatch(batchSize)
	#loss = 0
	fc1,sig1,fc2 = forward(np.mat(batch[:,0]).T)
	#print 'fc2',fc2.shape
	loss = fc2 - np.mat(batch[:,1]).T
	loss = a=np.ones(shape=loss.shape)*np.sum(loss)
	#print 'loss',loss.shape
	W1,b1,W2,b2=backward(np.mat(batch[:,0]).T,fc1,sig1,fc2,loss,W1,b1,W2,b2)

	if i % (num/5) == 0:
			print 'fc2:',fc2[0],'true value is :', batch[0,1]

test = np.mat(np.random.randn(batchSize)*100-50)
test=test.T
res = predict(test)
print res
print " true is : ", test*2.0+5.0