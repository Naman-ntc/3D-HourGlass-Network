import torch
import torch.nn as nn

lossfunc = nn.MSELoss().cuda().float()

#@torch.set_default_tensor_type('torch.cuda.FloatTensor')

def Joints2DHeatMapsSquaredError(input, target):
	global lossfunc
	"""
	Takes input as (N,C,D,H,W) and similar target (Here C is number of channels equivalent to number of joints
	and H,W are equal to the input image dimensions (i.e. 256 each))
	"""
	assert input.shape == target.shape
	#assert len(input.shape) == 5
	input = input.cuda()
	return lossfunc(input, target)

def Joints2DArgMaxSquaredError(input, target):
	global lossfunc
	"""
	Takes input as (N,C,D,2) and similar target (Here C is number of channels equivalent to number of joints)
	"""
	assert input.shape == target.shape
	#assert len(input.shape) == 4
	input = input.cuda()
	return lossfunc(input, target)

def JointsDepthSquaredError(input, target):
	global lossfunc
	"""
	Takes input as (N,C,D,1) and similar target (Here C is number of channels equivalent to number of joints)
	"""
	assert input.shape == target.shape
	#assert len(input.shape) == 4
	input = input.cuda()
	return lossfunc(input, target)

"""
Temporal Losses Below


s1 = u * t + 0.5 * a * t^2
s2 = u * 2 * t + 0.5 * a * (2*t)^2

s2 - 2 * s1 = 0.5 * a * 2 * t^2

a = (s2 - 2 * s1)/(t^2)
"""

def DistanceMatchingError(input, target):
	global lossfunc
	"""
	Takes input as (N,C,D,3) 3D coordinates and similiar targets (Here C is number of channels equivalent to number of joints)
	"""
	assert input.shape == target.shape
	assert len(input.shape) == 4
	input = input.cuda()
	inputdistances = input[:,:,1:,:] - input[:,:,:-1,:]
	inputdistances = torch.norm(inputdistances, dim=3)
	targetdistances = target[:,:,1:,:] - target[:,:,:-1,:]
	targetdistances = torch.norm(targetdistances, dim=3)
	return lossfunc(inputdistances, targetdistances)


def AccelerationMatchingError(input, target):
	global lossfunc
	"""
	Takes input as (N,C,D,3) 3D coordinates and similiar targets (Here C is number of channels equivalent to number of joints)
	"""
	assert input.shape == target.shape
	assert len(input.shape) == 4
	#print('\n')
	#print(input[0,:8,0,:])
	#print(target[0,:8,0,:])
	input = input.cuda()
	inputdistances = input[:,:,1:,:] - input[:,:,:-1,:]
	inputdistances = torch.norm(inputdistances, dim=3)
	inputaccn = inputdistances[:,:,2:] + inputdistances[:,:,:-2] - 2*inputdistances[:,:,1:-1]
	targetdistances = target[:,:,1:,:] - target[:,:,:-1,:]
	targetdistances = torch.norm(targetdistances, dim=3)
	targetaccn = targetdistances[:,:,2:] + targetdistances[:,:,:-2] - 2*targetdistances[:,:,1:-1]
	return lossfunc(inputaccn, targetaccn)
