import torch.nn as nn
from .Layers2D import *

class scaledFeaturesNet(object):
	"""docstring for scaledFeaturesNet"""
	def __init__(self, inChannels, inpScale, config):
		super(scaledFeaturesNet, self).__init__()
		self.inChannels = inChannels
		self.inpScale = inpScale
		self.layers = []
		for layer in config:
			if layer == 'const':
				self.layers.append(Residual2D(self.inChannels, self.inChannels))
			if layer == 'half':
				self.layers.append(Residual2D(self.inChannels, self.inChannels, stride = 2))
				self.inpScale/=2
		self.layers = nn.Sequential(* self.layers)

	def forward(self, input):
		out = self.layers(input)
		return out