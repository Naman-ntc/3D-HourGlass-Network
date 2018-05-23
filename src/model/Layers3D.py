import torch
import torch.nn as nn

class myBatchNorm3D(nn.Module):
	"""docstring for myBatchNorm3D"""
	def __init__(self, inChannels):
		super(myBatchNorm3D, self).__init__()
		self.inChannels = inChannels
		self.bn = nn.BatchNorm2d(self.inChannels)

	def forward(self, input):
		out = input
		N,C,D,H,W = out.size()
		out = out.squeeze(0).t().reshape(D,C,H,W)
		out = self.bn(out.contiguous())
		out = out.t().reshape(C,D,H,W).unsqueeze(0)
		return out



class ConvBnRelu3D(nn.Module):
	"""docstring for ConvBnRelu3D"""
	def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0, padLayer=None):
		super(ConvBnRelu3D, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.kernelSize = kernelSize
		self.stride = stride
		self.padding = padding
		self.padLayer = padLayer
		self.bn = myBatchNorm3D(self.inChannels)
		self.relu = nn.ReLU()
		self.conv = nn.Conv3d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)

	def forward(self, input):
		out = input
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
		out = self.bn(out)
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
		out = self.relu(out)
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
		if (self.padLayer is not None):
			out = self.padLayer(out)
		out = self.conv(out)
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
		return out


class ConvBlock3D(nn.Module):
	"""docstring for convBlock3D"""
	def __init__(self, inChannels, outChannels):
		super(ConvBlock3D, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.cbr1 = ConvBnRelu3D(self.inChannels, self.outChannels//2, 1, 1, 0)
		self.padded = nn.ReplicationPad3d((0,0,0,0,1,1))
		self.cbr2 = ConvBnRelu3D(self.outChannels//2, self.outChannels//2, 3, 1, (0,1,1))
		self.cbr3 = ConvBnRelu3D(self.outChannels//2, self.outChannels, 1, 1, 0)

	def forward(self, input):
		out = input
		out = self.cbr1(out)
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
		out = self.cbr2(out)
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
		out = self.cbr3(out)
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
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
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
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
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
		out = out + self.skip(input)
		assert (out[:,:,0,:,:] == out[:,:,1,:,:]).all()
		return out
