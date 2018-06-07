import torch.nn as nn
from .Layers2D import *
from .Layers3D import *

class scaledFeaturesNet(object):
	"""docstring for scaledFeaturesNet"""
	def __init__(self, inChannels):
		super(scaledFeaturesNet, self).__init__()
		self.inChannels = inChannels
		self.reduceTemporal = nn.AvgPool3d((3,1,1),(2,1,1))
		self.res1 = Residual3D(inChannels,inChannels)
		self.res2 = Residual3D(inChannels,inChannels)
		self.mp = nn.MaxPool3D()
		self.layers = nn.Sequential(* self.layers)

	def forward(self, input):
		out = self.layers(input)
		return out