import torch
import torch.nn as nn
from torch.autograd import Variable

class SoftArgMax(nn.Module):
	"""docstring for SoftArgMax"""
	def __init__(self):
		super(SoftArgMax, self).__init__()
		self.softmaxLayer = nn.Softmax(dim=-1)

	def forward(self, input):
		N,C,D,H,W = input.size()
		reshapedInput = input.view(N,C,D,-1)
		weights = self.softmaxLayer(reshapedInput)
		#print(input.size())
		#print(weights.size())
		#print(torch.arange(H*W).unsqueeze(0).unsqueeze(0).size())
		semiIndices = ((weights) * Variable(torch.arange(H*W).expand(weights.size())).cuda()).sum(dim=-1)
		indicesX = semiIndices % H
		indicesY = semiIndices / H
		indices = torch.cat((indicesX.unsqueeze(-1), indicesY.unsqueeze(-1)), dim=-1)
		return indices
