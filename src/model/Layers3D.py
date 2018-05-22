import torch
import torch.nn as nn

class ConvBnRelu3D(nn.Module):
	"""docstring for ConvBnRelu3D"""
	def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
		super(ConvBnRelu3D, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.kernelSize = kernelSize
		self.stride = stride
		self.padding = padding
		self.bn = nn.BatchNorm3d(self.inChannels)
		self.relu = nn.LeakyReLU()
		self.conv = nn.Conv3d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)

	def forward(self, input):
		out = input
		out = self.bn(out)
		out = self.conv(out)
		out = self.relu(out)
		return out


class ConvBlock3D(nn.Module):
	"""docstring for convBlock3D"""
	def __init__(self, inChannels, outChannels):
		super(ConvBlock3D, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.cbr1 = ConvBnRelu3D(self.inChannels, self.outChannels//2, 1, 1, 0)
		self.padded = nn.ReplicationPad3d(1)
		self.cbr2 = ConvBnRelu3D(self.outChannels//2, self.outChannels//2, 3, 1, 1)
		self.cbr3 = ConvBnRelu3D(self.outChannels//2, self.outChannels, 1, 1, 0)

	def forward(self, input):	
		out = input
		out = self.cbr1(out)
		out = self.padded(out)
		out = self.cbr2(out)
		out = self.cbr3(out)
		return out

class SkipLayer3D(nn.Module):
	"""docstring for SkipLayer3D"""
	def __init__(self, inChannels, outChannels):
		super(SkipLayer3D, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		if (self.inChannels == self.outChannels):
			self.conv = None
		else:
			self.conv = nn.Conv3d(self.inChannels, self.outChannels, 1)

	def forward(self, input):
		out = input
		if self.conv is not None:
			out = self.conv(out)
		return out	
		
class Residual3D(nn.Module):
	"""docstring for Residual3D"""
	def __init__(self, inChannels, outChannels):
		super(Residual3D, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.cb = ConvBlock3D(inChannels, outChannels)
		self.skip = SkipLayer3D(inChannels, outChannels)

	def forward(self, input):
		out = 0
		out = out + self.cb(input)
		out = out + self.skip(input)
		return out
