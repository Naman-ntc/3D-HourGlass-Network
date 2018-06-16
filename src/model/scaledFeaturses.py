import torch.nn as nn
from .Layers2D import *
from .Layers3D import *

class scaledFeaturesNet(object):
	"""docstring for scaledFeaturesNet"""
	def __init__(self, inChannels, config):
		super(scaledFeaturesNet, self).__init__()
		self.inChannels = inChannels
		self.reduceTemporal = nn.AvgPool3d((3,1,1),(2,1,1))
		self.layers = []
		for c in config:
			if c=='r3D':
				self.layers.append(Residual3D(inChannels,inChannels))
			if c=='rr3D':
				self.layers.append(Residual3D(inChannels,inChannels//2))
			if c=='r2D':
				self.layers.append(Residual2D(inChannels,inChannels))
			if c=='rr2D':
				self.layers.append(Residual2D(inChannels,inChannels//2))		
			if c=='m':
				self.layers.append(nn.MaxPool3d((1,2,2),(1,2,2)))
			if c=='mt':
				self.layers.append(nn.MaxPool3d(2,2))		
		self.layers = nn.Sequential(* self.layers)

	def forward(self, input):
		out = input
		out = self.reduceTemporal(out)
		out = self.layers(out)
		return out