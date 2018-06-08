import torch
import torch.nn as nn
from .Layers3D import *

class DepthRegressor3D(nn.Module):
	"""docstring for DepthRegressor3D"""
	def __init__(self, nChannels = 128, nRegModules = 2, nRegFrames = 8, nJoints = 16):
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

		self.fc = nn.Linear(self.nChannels*self.nRegFrames*4*4, self.nJoints*self.nRegFrames)

	def forward(self, input):
		out = self.reg(input)
		N = out.size()[0]
		D = out.size()[2]
		slides = D/ self.nRegFrames
		z = torch.autograd.Variable(torch.zeros(N, self.nJoints, D, 1)).cuda()
		for i in range(int(slides)):
			temp1 = out[:,:,self.nRegFrames*i:self.nRegFrames*(i+1),:,:].transpose(1,2).contiguous().view(N, -1)
			temp2 = self.fc(temp1).view(N,self.nRegFrames, self.nJoints).transpose(1,2).unsqueeze(-1)
			z[:,:,self.nRegFrames*i:self.nRegFrames*(i+1),:] = temp2
		rem = D % self.nRegFrames

		if (rem != 0):
			"""
			INCORRECT XXXXXXXX
			"""
			temp1 = out[:,:,self.nRegFrames*i:self.nRegFrames*(i+1),:,:].transpose(1,2).view(N, -1)
			temp2 = self.fc(temp1).view(N,self.nRegFrames, self.nJoints).transpose(1,2).unsqueeze(-1)
			z[:,:,self.nRegFrames*i:self.nRegFrames*(i+1),:] = temp2
			z[:,:,self.nRegFrames*int(slides):D,:] = self.fc(out[:,:,D-self.nRegFrames:D,:,:].reshape(-1, self.nJoints*self.nRegFrames*self.nChannels)).reshape(self.nRegFrames, self.nJoints).t().reshape(self.nJoints, self.nRegFrames).unsqueeze(0).unsqueeze(-1)[:,:,self.nRegFrames + self.nRegFrames*int(slides) - D:self.nRegFrames,:]
		return z
