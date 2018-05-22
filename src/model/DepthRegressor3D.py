import torch
import torch.nn as nn
from .Layers3D import *

class DepthRegressor3D(nn.Module):
	"""docstring for DepthRegressor3D"""
	def __init__(self, nChannels = 128, nRegModules = 2, nRegFrames = 16, nJoints = 16):
		super(DepthRegressor3D, self).__init__()
		self.nChannels = nChannels
		self.nRegModules = nRegModules
		self.nRegFrames = nRegFrames
		self.nJoints = nJoints
		reg_ = []
		for _ in range(4):
			for _ in range(self.nRegModules):
				reg_.append(Residual3D(self.nChannels,self.nChannels))
			reg_.append(nn.MaxPool3d((1,2,2), (1,2,2)))

		self.reg = nn.Sequential(* reg_)

		self.fc = nn.Linear(self.nChannels*self.nRegFrames*4*4, 16*self.nRegFrames)

	def forward(self, input):
		out = self.reg(input)
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
		N = out.size()[0]
		D = out.size()[2]
		slides = D/ self.nRegFrames
		z = torch.zeros(N, 16, D, 1)
		for i in range(int(slides)):
			assert (out[:,:,self.nRegFrames*i,:,:] == out[:,:,self.nRegFrames*i+self.nRegFrames-1,:,:]).all()
			z[:,:,16*i:16*i+16,:] = self.fc(out[:,:,16*i:16*i+16,:,:].reshape(-1, 16*self.nRegFrames*self.nChannels)).reshape(self.nRegFrames, 16).t().reshape(16, self.nRegFrames).unsqueeze(0).unsqueeze(-1)
			assert (z[:,:,self.nRegFrames*i,:] == z[:,:,self.nRegFrames*i+self.nRegFrames-1,:]).all()
		rem = D % self.nRegFrames

		if (rem != 0):
			z[:,:,16*int(slides):D,:] = self.fc(out[:,:,D-16:D,:,:].reshape(-1, 16*self.nRegFrames*self.nChannels)).reshape(self.nRegFrames, 16).t().reshape(16, self.nRegFrames).unsqueeze(0).unsqueeze(-1)[:,:,16 + 16*int(slides) - D:16,:]
		return z
