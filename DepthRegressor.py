import torch
import torch.nn as nn
from Layers3D import *

class DepthRegressor(nn.Module):
	"""docstring for DepthRegressor"""
	def __init__(self, nChannels = 128, nRegModules = 4, nFrames = 16, nJoints = 16):
		super(DepthRegressor, self).__init__()
		self.nChannels = nChannels
		self.nRegModules = nRegModules
		self.nFrames = nFrames
		self.nJoints = nJoints
		reg_ = []
		for _ in range(4):
			for _ in range(self.nRegModules):
				reg_.append(Residual3D(self.nChannels,self.nChannels))
			reg_.append(nn.Maxpool3d((1,2,2), (1,2,2)))

		self.reg = nn.Sequential(* reg_)
		
		self.fc = nn.Linear(self.nChannels*self.nFrames*4*4, 16*self.nFrames)

	def forward(self, input):
		out = self.reg(input)
		D = out.size()[2]
		slides = D/ self.nFrames
		z = torch.zeros(D)
		for i in range(slides):
			z[16*i:16*i+16] = self.fc(out[:,:,16*i:16*i+16,:,:])
		rem = D % self.nFrames

		if (rem != 0):
			z[16*slides:D] = self.fc(out[:,:,D-16:D,:,:])