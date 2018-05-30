import torch
from torch import nn, autograd, optim
import pickle

data = torch.randn(500, 48, 16)
labels = torch.LongTensor(500).random_(0, 49)
valData = data[1:200]
valLabels = labels[1:200]

def getData(batchSize, index):
	return (data[index:index + batchSize], labels[index:index+batchSize])




#input should be batchSize x numJoints x time 

numJoints = 48

config = [ 
             [(1,7,64)],
             [(1,7,64)],
             [(1,7,64)],
             [(2,7,128)],
             [(1,7,128)],
             [(1,7,128)],
             [(1,7,256)],
             [(1,7,256)],
             [(1,7,256)],
           ]
#config is the network in brief, each entry is a residual network, with 1d convolutions strictly along the temporal dimension.
#each tuple is stride, length of convolution filter (ie 8 in our case, analogous to the 5x5 filter in 2d), numChannels


class residual1D(nn.Module):
	def __init__(self, inpFeatures, layerConfig, dropout = 0.5):
		super(residual1D, self).__init__()
		
		self.inpFeatures = inpFeatures
		self.stride, self.convLength, self.outFeatures = layerConfig
		self.batchNorm = nn.BatchNorm1d(self.inpFeatures)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p = dropout)

		#make sure the temporal dimension is last because that is what the 1d convolution should be over
		self.conv = nn.Conv1d(self.inpFeatures, self.outFeatures, self.convLength, stride = self.stride, padding = 3)#(int(self.convLength/2), int(self.convLength/2 + 0.9999)))
		
		#if convolutions are strided, then the temporal dimensions will not match and so we may need to downsample
		if self.stride == 2 or self.inpFeatures != self.outFeatures:
			self.skipConv = nn.Conv1d(self.inpFeatures, self.outFeatures, 1, stride = self.stride, padding = 0)
		else:
			self.skipConv = None

	def forward(self, input):
		x = input
		if self.skipConv:
			y = self.skipConv(x)
		else:
			y = x
		x = self.batchNorm(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.conv(x)
		x = x + y
		return x

class tcn(nn.Module):
	def __init__(self, config, temporalExtent = 16,  numClasses = 49):
		super(tcn, self).__init__()
		global numJoints
		self.numClasses = numClasses
		self.numLayers = len(config)
		
		#lot of 1d convolutions (with one downsampling along the temporal dimension) and then a fully connected (the above config gave the best results from what 
		#I could make out from their results and paper and code)
		#the fully connected is getting pretty big, can reduce once along numChannels as well if there is overfitting
		self.convolutions = []
		numFeatures = numJoints
		for layer in config:
			self.convolutions.append(residual1D(numFeatures, layer[0]))
			numFeatures = layer[0][2]
			temporalExtent = temporalExtent/layer[0][0]
		#print(numFeatures, temporalExtent, numFeatures*temporalExtent)
		self.fc = nn.Linear(int(numFeatures*temporalExtent), int(numClasses))
		#self.softmax = nn.Softmax()


	def forward(self, input):
		x = input
		for convolution in self.convolutions:
			x = convolution(x)
		#print(x.shape)
		x = x.view(x.size(0), -1)
		#print(x.shape)
		x = self.fc(x)
		#print(x.shape)
		#x = self.softmax(x)
		return x



lossFunction = nn.CrossEntropyLoss()
#cross entropy loss applied lofsoftmax as well bedfore computing the loss, so there is no softmax layer at the end of the network

model = tcn(config)

def checkAcc(data, labels, l = 500, start = 0, printing = False):
	global model
	if l == -1:
		length = len(data)
	else:
		length = l
	dataLength = len(data)
	score = 0
	for i in range(start, start + length):
		index = i % dataLength
		#print(data[index].shape)
		pred = model(data[index].view(1, data[index].shape[0], data[index].shape[1]))
		_, predClass = pred.max(1)
		#print(pred.data.max(1)[1].item(), labels[index].item())
		if predClass.item() == labels[index].data:
			score += 1
		if i%200 == 0 and printing and i > 0:
			print(i, "of", length, "done! Accuracy yet is", float(score)/float(i))
	if printing:
		print("The accuracy is", float(score)/float(length))
	return float(score)/float(length)

losses = []
trainAccuracies = []
valAccuracies = []

def saveData():
	global losses, trainAccuracies, valAccuracies
	l = [losses, trainAccuracies, valAccuracies]
	pickle.dump(l, open("tcnData.npy", 'wb'))


def train(numEpoch = 1, batchSize = 8, lr = 1e-5, numIter = -1, recInterval = 10):
	global data, labels, model, valData, valLabels
	global losses, trainAccuracies, valAccuracies
	optimizer = optim.Adam(model.parameters(), lr)
	totalLoss = 0
	for epoch in range(numEpoch):
		if numIter == -1:
			iterations = len(data)//batchSize
		else:
			iterations = numIter//batchSize
		for iteration in range(iterations):
			model.zero_grad()		

			currentData, Y = getData(batchSize, iteration*batchSize) ##get data in the current format

			X = autograd.Variable(currentData)
			Y = autograd.Variable(Y)
			y_hat = model(X)
			loss = lossFunction(y_hat, Y)
			totalLoss += loss.data[0]
			loss.backward(retain_graph = True)
			optimizer.step()

			if iteration % recInterval == 0 and iteration > 0:
				losses.append(totalLoss/recInterval)
				trainAccuracies.append(checkAcc(data, labels, l = 200, start = int(iteration*batchSize)))
				valAccuracies.append(checkAcc(valData, valLabels,l = 200, start = int(iteration*batchSize/3.0)))
				print("Completed iteration", iteration, "of", iterations, "for epoch", epoch, "with loss", losses[-1].item(), "trainAcc", trainAccuracies[-1], "valAcc", valAccuracies[-1])
				totalLoss = 0

		print("Epoch", epoch, "over!")
		saveData()

			







