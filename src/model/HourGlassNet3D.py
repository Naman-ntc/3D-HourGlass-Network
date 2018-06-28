from __future__ import print_function
import torch
import torch.nn as nn
from .Layers3D import *
from .HourGlass3D import *
import ref

def help(x):
	print(x.std(dim=2).mean())

flatten=lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]

def robust(x, temporal):
	M = max(max(flatten(temporal)),max(flatten(ref.temporal[1])))
	D = int(x.size()[2])
	if D>M:
		diff = (D-M)//2
		x = x[:,:,diff:-diff,:,:]
		print("robusting")
	return x

class HourglassNet3D(nn.Module):
	"""docstring for HourglassNet3D"""
	def __init__(self, nChannels = 128, nStack = 2, nModules = 2, numReductions = 4, nJoints = 16, temporal=-1):
		super(HourglassNet3D, self).__init__()
		self.nChannels = nChannels
		self.nStack = nStack
		self.nModules = nModules
		self.numReductions = numReductions
		self.nJoints = nJoints
		self.temporal = temporal
		self.convStart = myConv3d(3, 64, (1,7,7), (1,2,2), (3,3))
		self.bnStart = myBatchNorm3D(64)
		self.reluStart = nn.ReLU()

		self.res1 = Residual3D(64,128, temporal[1])
		self.mp = nn.MaxPool3d((1,2,2),(1,2,2))

		self.res2 = Residual3D(128,128, temporal[2])
		self.res3 = Residual3D(128,self.nChannels, temporal[3])

		_hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [],[],[],[],[],[]
		for i in range(self.nStack):
			_hourglass.append(Hourglass3D(self.nChannels, self.numReductions, self.nModules, (1,2,2), (1,2,2), 2, temporal[4+i][0]))
			_ResidualModules = []
			for j in range(self.nModules):
				_ResidualModules.append(Residual3D(self.nChannels, self.nChannels, temporal[4+i][1][j]))
			_ResidualModules = nn.Sequential(*_ResidualModules)

			_Residual.append(_ResidualModules)
			_lin1.append(nn.Sequential(myConv3d(self.nChannels, self.nChannels,(temporal[4+i][2],1,1)), myBatchNorm3D(self.nChannels), nn.ReLU()))
			_chantojoints.append(myConv3d(self.nChannels, self.nJoints,(temporal[4+i][3],1,1)))
			_lin2.append(myConv3d(self.nChannels, self.nChannels,(temporal[4+i][4],1,1)))
			_jointstochan.append(myConv3d(self.nJoints,self.nChannels,(temporal[4+i][5],1,1)))

		self.hourglass = nn.ModuleList(_hourglass)
		self.Residual = nn.ModuleList(_Residual)
		self.lin1 = nn.ModuleList(_lin1)
		self.chantojoints = nn.ModuleList(_chantojoints)
		self.lin2 = nn.ModuleList(_lin2)
		self.jointstochan = nn.ModuleList(_jointstochan)

	def forward(self, input):
		x = input
		x = self.convStart(x)
		x = robust(x, self.temporal[1:])
		x = self.bnStart(x)
		x = self.reluStart(x)
		x = self.res1(x)
		x = robust(x, self.temporal[2:])
		x = self.mp(x)
		x = self.res2(x)
		x = robust(x, self.temporal[3:])
		x = self.res3(x)
		x = robust(x, self.temporal[4:])
		out = []

		for i in range(self.nStack):
			x1 = self.hourglass[i](x)
			x1 = robust(x1, self.temporal[min(5+i,5):])
			x1 = self.Residual[i](x1)
			x1 = robust(x1, self.temporal[min(5+i,5):])
			x1 = self.lin1[i](x1)
			out.append(self.chantojoints[i](x1))
			x1 = self.lin2[i](x1)
			x = x + x1 + self.jointstochan[i](out[i])
		x = robust(x, [1])
		return (out,x)
