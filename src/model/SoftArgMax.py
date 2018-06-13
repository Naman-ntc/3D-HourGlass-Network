import torch
import torch.nn as nn

class SoftArgMax(nn.Module):
	"""docstring for SoftArgMax"""
	def __init__(self):
		super(SoftArgMax, self).__init__()
		self.softmaxLayer = nn.Softmax(dim=-1)

	def forward(self, input, factor = 10000):
		N,C,D,H,W = input.size()
		reshapedInput = factor*input.view(N,C,D,-1)
		weights = self.softmaxLayer(reshapedInput)
		semiIndices = ((weights) * (torch.arange(H*W).expand(weights.size())).cuda()).sum(dim=-1)
		indicesX = semiIndices % W
		indicesY = semiIndices / W
		indices = torch.cat((indicesX.unsqueeze(-1), indicesY.unsqueeze(-1)), dim=-1)
		return indices
