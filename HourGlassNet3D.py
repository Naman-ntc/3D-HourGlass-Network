import torch
import torch.nn as nn
from Layers3D import *
from HourGlass3D import *

class HourglassNet3D(object):
	"""docstring for HourglassNet3D"""
	def __init__(self, nChannels, nStack = 2, nModules = 1, numReductions):
		super(HourglassNet3D, self).__init__()
		self.nChannels = nChannels
		self.nStack = nStack
		self.nModules = nModules
		self.numReductions = numReductions
		
		self.cbrStart = nn.ConvBnRelu3D(3, 64, (1,7,7), (1,2,2), (0,3,3)) ## self.convStart = nn.ConvBnRelu3D(3, 64, (3,7,7), (1,2,2), (1,3,3))
		
		self.res1 = Residual3D(64,128)
		self.mp = nn.MaxPool3D(2,2)

		self.res2 = Residual3D(128,128)
		self.res3 = Residual3D(128,self.nChannels)

		_hourglass = []
		for _ in range(self.nStack):
			_hourglass.append(HourGlass3D(self.nChannels, self.))