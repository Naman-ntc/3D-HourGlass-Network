import torch
import torch.nn as nn
from Layers3D import *

class Hourglass3D(nn.Module):
	"""docstring for Hourglass3D"""
	def __init__(self, nChannels, numReductions = 3, nModules = 1, poolKernel = (2,2,2), poolStride = (2,2,2), upSampleKernel = 2):
		super(Hourglass3D, self).__init__()
		self.numReductions = numReductions
		self.nModules = nModules
		self.nChannels = nChannels
		self.poolKernel = poolKernel
		self.poolStride = poolStride
		self.upSampleKernel = upSampleKernel
		"""
		For the skip connection, a residual3D module (or sequence of residuaql modules)
		"""
		
		_skip = []
		for _ in range(self.nModules):
			_skip.append(Residual3D(self.nChannels,self.nChannels))

		self.skip = nn.Sequential(*_skip)
		
		"""
		Hourglass3D pooling to go to smaller dimension and subsequent cases:
			either pass through Hourglass3D of numReductions-1
			or pass through Residual3D Module or sequence of Modules
		"""

		self.mp = nn.MaxPool3d(self.poolKernel, self.poolStride)
		if (numReductions > 1):
			self.hg = Hourglass3D(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
		else:
			_num1res = []
			for _ in range(self.nModules):
				_num1res.append(Residual3D(self.nChannels,self.nChannels))
			
			self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?
		
		"""
		Now another Residual3D Module or sequence of Residual3D Modules
		"""
		
		_lowres = []
		for _ in range(self.nModules):
			_lowres.append(Residual3D(self.nChannels,self.nChannels))

		self.lowres = nn.Sequential(*_lowres)

		"""
		Upsampling Layer (Can we change this??????)  
		As per Newell's paper upsamping recommended
		"""
		self.up = nn.Upsample(scale_factor = self.upSampleKernel)

	def forward(self, input):
		out1 = input
		out1 = self.skip(out1)

		out2 = input
		out2 = self.mp(out2)

		if self.numReductions>1:
			out2 = self.hg(out2)
		else:
			out2 = self.num1res(out2)

		out2 = self.lowres(out2)
		out2 = self.up(out2)

		return out2