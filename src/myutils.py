import ref
import numpy as np

from model.SoftArgMax import *
SoftArgMaxLayer = SoftArgMax()

def getPreds(hm):
	assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
	res = hm.shape[2]

	hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
	idx = np.argmax(hm, axis = 2)
	preds = np.zeros((hm.shape[0], hm.shape[1], 2))
	for i in range(hm.shape[0]):
		for j in range(hm.shape[1]):
			preds[i, j, 0], preds[i, j, 1] = idx[i, j] % res, idx[i, j] / res
	return preds

def softGetPreds(hm):
	global SoftArgMaxLayer
	assert len(hm.shape) == 5, 'Input must be a 5-D tensor'
	return SoftArgMaxLayer(hm)

def give3D(output2D,output3D,meta):
	#out = torch.cat((softGetPreds(output2D), output3D), dim=3)
	p = np.zeros((output2D.shape[0], ref.nJoints, 3))
	p[:, :, :2] = getPreds(output2D).copy()

	hm = output2D.reshape(output2D.shape[0], output2D.shape[1], ref.outputRes, ref.outputRes)
	for i in range(hm.shape[0]):
		for j in range(hm.shape[1]):
			pX, pY = int(p[i, j, 0]), int(p[i, j, 1])
			scores = hm[i, j, pX, pY]
			if pX > 0 and pX < ref.outputRes - 1 and pY > 0 and pY < ref.outputRes - 1:
				diffY = hm[i, j, pX, pY + 1] - hm[i, j, pX, pY - 1]
				diffX = hm[i, j, pX + 1, pY] - hm[i, j, pX - 1, pY]
				p[i, j, 0] = p[i, j, 0] + 0.25 * (1 if diffX >=0 else -1)
				p[i, j, 1] = p[i, j, 1] + 0.25 * (1 if diffY >=0 else -1)
	p = p + 0.5
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
