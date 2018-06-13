import ref
import numpy as np


from model.SoftArgMax import *
SoftArgMaxLayer = SoftArgMax()

def softGetPreds(hm):
	global SoftArgMaxLayer
	assert len(hm.shape) == 5, 'Input must be a 5-D tensor'
	return SoftArgMaxLayer(hm)

def give3D(output2D,output3D,meta):
	out = torch.cat((softGetPreds(output2D), output3D), dim=3)
	out[:,:,:,2] = (out[:,:,:,2] + 1) / 2 * ref.outputRes
	h36mSumLen = 4296.99233013
	root = 6
	lens = torch.zeros_like(out[:,0,:,0]).cuda().float()
	for e in ref.edges:
		lens += torch.norm(out[:,e[0],:,:] - out[:,e[1],:,:], dim=-1)
	rootRelative = h36mSumLen * ((out[:,:,:,:] - out[:,root:root+1,:,:]) / lens.unsqueeze(dim=-1).unsqueeze(dim=1).expand(out.size())) + meta[:,root:root+1,:,:]
	rootRelative[:,7,:,:] = (rootRelative[:,6,:,:] + rootRelative[:,8,:,:])/2
	return rootRelative


def smartMPJPE(pred,gt):
	sqddiff = torch.norm(pred-gt,dim=-1)
	return sqddiff.mean()
