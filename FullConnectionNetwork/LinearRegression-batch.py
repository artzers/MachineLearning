import numpy as np

#X1 = np.mat(np.linspace(-50,50,100)).T
#X2 = np.mat(np.linspace(-50,50,100)).T
X1 = np.mat(np.random.randn(100)*100-50).T
X2 = np.mat(np.random.randn(100)*100-50).T
X=np.column_stack((X1,X2))
y = X1 * 3.0 + X2 * 2.0
trainData=np.column_stack((X1,X2,y))

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

#fc1 sigmoid fc2
lr1 = 0.000001
lr2 = 0.000001
inputDim = 2
fc1Dim = 20
fc2Dim = 1
batchSize = 1
W1 = np.random.randn(inputDim,fc1Dim)#1x20
b1 = np.zeros((1,fc1Dim))#1x20
W2 = np.random.randn(fc1Dim, fc2Dim)#20x1
b2 = np.zeros((1,fc2Dim)) #10x1

def forward(x):
	#print x.shape
	fc1=np.dot(x,W1)+b1
	#print fc1.shape
	#sig1=sigmoid(fc1)
	sig1 = fc1
	#print np.dot(sig1,W2).shape
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
	res=np.zeros((data.shape[0],1))
	ind = 0
	for i in xrange(0,data.shape[0]):
		print 'data',data[i,:]
		_,_,res[ind]=forward(data[i,:])
		ind +=1

	return res

def BuildBatch(batchSize):
	batch = np.random.permutation(trainData)[0:batchSize,:]
	return batch

num = 100
for i in xrange(0,num):
	#x=np.mat([3.0])
	batch = BuildBatch(batchSize)
	loss = 0
	for x in batch:
		fc1,sig1,fc2 = forward(np.mat(x[0:2]))
		#print fc2.shape
		#print np.mat(x[1]).T.shape
		loss += fc2 - np.mat(x[2]).T

	#loss /= batchSize
	#print 'fc2',fc2.shape
	#print 'loss',loss.shape
	W1,b1,W2,b2=backward(np.mat(batch[-1,0:2]),fc1,sig1,fc2,loss,W1,b1,W2,b2)

	if i % (num/5) == 0:
			print 'fc2:',fc2,'true value is :', x[2]

test1 = np.mat(np.random.randn(batchSize)*100-50).T
test2 = np.mat(np.random.randn(batchSize)*100-50).T
test=np.column_stack((test1,test2))
#print 'test',test
#res=predict(test)
_,_,res=forward(test[0,:])
print res
print " true is : ", test[:,0]*3.0+test[:,1]*2.0