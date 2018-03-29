import torch
import torch.nn as nn
from 2Dlayers import *

class StackedHourGlass(nn.Module):
	"""docstring for HourGlass"""
	def __init__(self,in_channels=256,out_channels=256,,nStacks=6,nFeat=256,initial_padding=0,outDim=16,nLow=3):
		super(HourGlass, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.nStacks = nStacks
		self.nFeat = nFeat
		self.outDim = outDim
		self.nLow = nLow
		self.initial_padding = initial_padding
		self.conv1 = nn.Conv2D(self.in_channels,64,kernel_size=6,stride=2,padding=self.initial_padding)
		self.res1 = Residual(64,128)
		self.pool1 = nn.MaxPool2d(2,2)
		self.res2 = Residual(128,128)
		self.res3 = Residual(128,self.nFeat)
		

	def forward(self,inputs):
		#Preprocessing
		