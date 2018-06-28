import torch
import torch.nn as nn
from .Layers3D import *

class DepthRegressor3D(nn.Module):
	"""docstring for DepthRegressor3D"""
	def __init__(self, nChannels = 128, nRegModules = 2, nRegFrames = 8, nJoints = 16, temporal):
		super(DepthRegressor3D, self).__init__()
		self.nChannels = nChannels
		self.nRegModules = nRegModules
		self.nRegFrames = nRegFrames
		self.nJoints = nJoints
		reg_ = []
		for i in range(4):
			for j in range(self.nRegModules):
				reg_.append(Residual3D(self.nChannels,self.nChannels, temporal[i][j]))
			reg_.append(nn.MaxPool3d((1,2,2), (1,2,2)))

		self.reg = nn.Sequential(* reg_)

		self.fc = nn.Linear(self.nChannels*self.nRegFrames*4*4, self.nJoints)

	def forward(self, input):
		out = self.reg(input)
		N = out.size()[0]
		D = out.size()[2]
		reg = torch.autograd.Variable(torch.zeros(N,self.nJoints,D,1).float().cuda())
		for i in range(1):
			fcin = torch.cat((out[:,:,i:i+1,:,:], out[:,:,i:i+2,:,:]), dim=2).contiguous()
			fcin = fcin.transpose(1,2).contiguous().view(N, -1)
			reg[:,:,i,:] = self.fc(fcin).unsqueeze(-1)
		for i in range(1,D-1):
			fcin = out[:,:,i-1:i+2:,:]
			fcin = fcin.transpose(1,2).contiguous().view(N, -1)
			reg[:,:,i,:] = self.fc(fcin).unsqueeze(-1)
		for i in range(D-1,D):
			fcin = torch.cat((out[:,:,i-1:i+1,:,:], out[:,:,i:i+1,:,:]), dim=2).contiguous()
			fcin = fcin.transpose(1,2).contiguous().view(N, -1)
			reg[:,:,i,:] = self.fc(fcin).unsqueeze(-1)	
		return reg	
