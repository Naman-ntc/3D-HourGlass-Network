import torch
import torch.nn as nn

class ConvBnRelu(nn.Module):
	"""docstring for ConvBnRelu"""
	def __init__(self, in_channels = 256, out_channels = 256, kernel_size = 1, stride = 1, padding = 0):
		super(ConvBnRelu, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
		self.bn = nn.BatchNorm2d(self.out_channels)
		self.relu = nn.LeakyReLU()

	def forward(self, inputs):
		out = inputs
		out = self.conv(out)
		out = self.bn(out)
		out = self.relu(out)
		return out

class ConvBlock(nn.Module):
	"""docstring for ConvBlock"""
	def __init__(self, in_channels = 256, out_channels = 256):
		super(ConvBlock, self).__init__()
		self.in_channels = in_channels 
		self.out_channels = out_channels
		self.cbr1 = ConvBnRelu(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
		self.cbr2 = ConvBnRelu(in_channels=int(self.out_channels/2),out_channels=int(self.out_channels/2),3,1,1)
		self.cbr3 = ConvBnRelu(in_channels=int(self.out_channels/2),out_channels=self.out_channels)

	def forward(self,inputs):
		out = inputs
		out = self.cbr1(out)
		out = self.cbr2(out)
		out = self.cbr3(out)
		return out

class SkipLayer(nn.Module):
	"""docstring for SkipLayer"""
	def __init__(self, in_channels,out_channels, kernel_size = 1, stride = 1, padding = 0):
		super(SkipLayer, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		if (self.in_channels == self.out_channels):
			self.conv = None
		else :
			self.kernel_size = kernel_size
			self.stride = stride
			self.padding = padding
			self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

	def	forward(self,inputs):
		out = inputs
		if self.conv is not None:
			out = self.conv(out)
		return out

class Residual(nn.Module):
	"""docstring for Residual"""
	def __init__(self, in_channels = 256, out_channels = 256):
		super(Residual, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.cb = ConvBlock(self.in_channels,self.out_channels)
		self.skip = SkipLayer(self.in_channels,self.out_channels)

	def forward(self,inputs):
		out1 = inputs
		out2 = inputs
		out2 = self.skip(out2)
		out = torch.add(out1,out2)
		return out		

class HourGlass(object):
	"""docstring for HourGlass"""
	def __init__(self, in_channels,out_channels,pool_padding,num_reductions):
		super(HourGlass, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_reductions = num_reductions
		self.res_up1 = Residual(in_channels,out_channels)
		self.pool = nn.MaxPool2d(2,2)
		self.res_low1 = Residual(out_channels,out_channels)
		self.res_low2 = Residual(out_channels,out_channels)
		self.res_low3 = Residual(out_channels,out_channels)
		self.nn_up2 = nn.Upsample(scale_factor=2)

	def forward(self,inputs):
		out = inputs
		out = self.res_up1(out)
		if num_reductions > 1 :
			self.hg = Hourglass(self.out_channels,self.out_channels,self.pool_padding,self.num_reductions-1)
			out = self.hg(out)
		out = self.res_low3(out)
		out = self.nn_up2(out)	
