from .layers.Residual import Residual
import torch.nn as nn
import math
import ref

class Hourglass(nn.Module):
	def __init__(self, n, nModules, nFeats):
		super(Hourglass, self).__init__()
		self.n = n
		self.nModules = nModules
		self.nFeats = nFeats
		
		_up1_, _low1_, _low2_, _low3_ = [], [], [], []
		for j in range(self.nModules):
			_up1_.append(Residual(self.nFeats, self.nFeats))
			self.low1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		for j in range(self.nModules):
			_low1_.append(Residual(self.nFeats, self.nFeats))
		
		if self.n > 1:
			self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
		else:
			for j in range(self.nModules):
				_low2_.append(Residual(self.nFeats, self.nFeats))
		  	self.low2_ = nn.ModuleList(_low2_)
		
		for j in range(self.nModules):
			_low3_.append(Residual(self.nFeats, self.nFeats))
		
		self.up1_ = nn.ModuleList(_up1_)
		self.low1_ = nn.ModuleList(_low1_)
		self.low3_ = nn.ModuleList(_low3_)
		
		self.up2 = nn.Upsample(scale_factor = 2)
	
	def forward(self, x):
		up1 = x
		for j in range(self.nModules):
			up1 = self.up1_[j](up1)
		
		low1 = self.low1(x)
		for j in range(self.nModules):
			low1 = self.low1_[j](low1)
		
		if self.n > 1:
			low2 = self.low2(low1)
		else:
			low2 = low1
		  	for j in range(self.nModules):
				low2 = self.low2_[j](low2)
		
		low3 = low2
		for j in range(self.nModules):
			low3 = self.low3_[j](low3)
		up2 = self.up2(low3)
		
		return up1 + up2
