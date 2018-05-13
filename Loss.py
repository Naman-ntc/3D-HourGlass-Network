import torch
import torch.nn as nn

def JointSquaredError(input, target):
	"""
	Takes input as (N,C,D,2) and similar target (Here C is number of channels equivalent to number of joints)
	"""
	N = input.size()[0]
	lossfunc = nn.MSELoss()
	return lossfunc(input.view(N,-1), target.view(N,-1))