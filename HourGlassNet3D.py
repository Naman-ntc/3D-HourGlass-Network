import torch
import torch.nn as nn
from Layers3D import *
from HourGlass3D import *

class HourglassNet3D(nn.Module):
	"""docstring for HourglassNet3D"""
	def __init__(self, nChannels, nStack = 1, nModules = 1, numReductions = 3, nJoints = 25):
		super(HourglassNet3D, self).__init__()
		self.nChannels = nChannels
		self.nStack = nStack
		self.nModules = nModules
		self.numReductions = numReductions
		self.nJoints = nJoints

		self.cbrStart = ConvBnRelu3D(3, 64, (1,7,7), (1,2,2), (0,3,3)) ## self.convStart = nn.ConvBnRelu3D(3, 64, (3,7,7), (1,2,2), (1,3,3))
		
		self.res1 = Residual3D(64,128)
		self.mp = nn.MaxPool3d(2,2)

		self.res2 = Residual3D(128,128)
		self.res3 = Residual3D(128,self.nChannels)

		_hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [],[],[],[],[],[]
		for _ in range(self.nStack):
			_hourglass.append(Hourglass3D(self.nChannels, self.numReductions, self.nModules))
			_ResidualModules = []
			for _ in range(self.nModules):
				_ResidualModules.append(Residual3D(self.nChannels, self.nChannels))
			_ResidualModules = nn.Sequential(*_ResidualModules)
			
			_Residual.append(_ResidualModules)	
			_lin1.append(ConvBnRelu3D(self.nChannels, self.nChannels))
			_chantojoints.append(nn.Conv3d(self.nChannels, self.nJoints,1))
			_lin2.append(nn.Conv3d(self.nChannels, self.nChannels,1))
			_jointstochan.append(nn.Conv3d(self.nJoints,self.nChannels,1))

		self.hourglass = nn.ModuleList(_hourglass)
		self.Residual = nn.ModuleList(_Residual)
		self.lin1 = nn.ModuleList(_lin1)
		self.chantojoints = nn.ModuleList(_chantojoints)
		self.lin2 = nn.ModuleList(_lin2)
		self.jointstochan = nn.ModuleList(_jointstochan)

	def forward(self, input):
		x = input
		x = self.cbrStart(x)
		x = self.res1(x)
		x = self.mp(x)
		x = self.res2(x)
		x = self.res3(x)

		out = []

		for i in range(self.nStack):
			x1 = self.hourglass[i](x)
			x1 = self.Residual[i](x1)
			x1 = self.lin1[i](x1)
			out.append(self.chantojoints[i](x1))
			x1 = self.lin2[i](x1)
			x = x + x1 + self.jointstochan[i](out[i])

		return out
