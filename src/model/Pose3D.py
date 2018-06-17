import torch
import torch.nn as nn
from .HourGlassNet3D import *
from .DepthRegressor3D import *

class Pose3D(nn.Module):
	"""docstring for Pose3D"""
	def __init__(self, nChannels = 128, nStack = 2, nModules = 2, numReductions = 4, nRegModules = 2, nRegFrames = 8, nJoints = 16):
		super(Pose3D, self).__init__()
		self.nChannels = nChannels
		self.nStack = nStack
		self.nModules = nModules
		self.numReductions = numReductions
		self.nRegModules = nRegModules
		self.nRegFrames = nRegFrames
		self.nJoints = nJoints

		self.hg = HourglassNet3D(self.nChannels, self.nStack, self.nModules, self.numReductions, self.nJoints)

		self.dr = DepthRegressor3D(self.nChannels, self.nRegModules, self.nRegFrames, self.nJoints)

	def forward(self, input):
		heatmaps, regInput = self.hg(input)
		#print(regInput[0,:,2,:,:])
		z = self.dr(regInput)
		return heatmaps + [z]
