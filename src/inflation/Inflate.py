import torch
from math import copysign

nChannels = 128
nStack = 2
nModules = 2
nRegFrames = 8
nJoints = 16
scheme = 1
tempKernel = 3
mult = 0.1

def inflatePose3D(model3d, model):
	inflateHourglassNet(model3d.hg, model)
	inflateDepthRegressor(model3d.dr, model)

def inflateDepthRegressor(model3d, model):
	for i in range(4):
		inflateResidual(model3d.reg[3*i], model.reg_[2*i])
		inflateResidual(model3d.reg[3*i+1], model.reg_[2*i+1])
		inflateMaxPool(model3d.reg[3*i+2], model.maxpool)
	inflateFullyConnected(model3d.fc, model.reg)	

def inflateFullyConnected(model3d, model):
	val = 4*4*nChannels
	for i in range(1):
		model3d.bias.data[nJoints*i:nJoints*(i+1)] = model.bias.data
		for j in range(nRegFrames):
			if (j == 1) :
				model3d.weight.data[nJoints*(i):nJoints*(i+1), val*(j):val*(j+1)] = model.weight.data #/ (1.0*nRegFrames)
			elif j<1 :
				model3d.weight.data[nJoints*(i):nJoints*(i+1), val*(j):val*(j+1)] = model.weight.data * mult#/ (1.0*nRegFrames)
			elif j>1 :
				model3d.weight.data[nJoints*(i):nJoints*(i+1), val*(j):val*(j+1)] = model.weight.data * mult#/ (1.0*nRegFrames)

def inflateHourglassNet(model3d, model):
	inflateconv(model3d.convStart, model.conv1_)
	model3d.bnStart.bn = inflatebn(model3d.bnStart, model.bn1)
	inflaterelu(model3d.reluStart, model.relu)
	inflateResidual(model3d.res1, model.r1)
	inflateResidual(model3d.res2, model.r4)
	inflateResidual(model3d.res3, model.r5)
	inflateMaxPool(model3d.mp, model.maxpool)
	for i in range(nStack):
		inflatehourglass(model3d.hourglass[i], model.hourglass[i])
	for i in range(nStack):
		for j in range(nModules):
			inflateResidual(model3d.Residual[i][j],model.Residual[nModules*i+j])
	for i in range(nStack):
		inflateconv(model3d.lin1[i][0], model.lin_[i][0])
		model3d.lin1[i][1].bn = inflatebn(model3d.lin1[i][1].bn, model.lin_[i][1])
		inflaterelu(model3d.lin1[i][2], model.lin_[i][2])
	for i in range(nStack):
		inflateconv(model3d.chantojoints[i], model.tmpOut[i])
		inflateconv(model3d.lin2[i], model.ll_[i])
		inflateconv(model3d.jointstochan[i], model.tmpOut_[i])
	return

def inflatehourglass(model3d, model):
	for i in range(nModules):
		inflateResidual(model3d.skip[i], model.up1_[i])
	inflateMaxPool(model3d.mp, model.low1)
	
	for i in range(nModules):
		inflateResidual(model3d.afterpool[i], model.low1_[i])

	if model3d.numReductions > 1:
		inflatehourglass(model3d.hg, model.low2)
	else :
		for i in range(nModules):
			inflateResidual(model3d.num1res[i], model.low2_[i])

	for i in range(nModules):
		inflateResidual(model3d.lowres[i], model.low3_[i])

	inflateupsampling(model3d.up, model.up2)
	return

def inflateconv(conv3d, conv):
	tempSize = conv3d.conv.weight.data.size()[2]
	center = (tempSize-1)//2
	if scheme==1:
		factor = torch.FloatTensor([copysign(mult**abs(center-i), center-i) for i in range(tempSize)]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(conv3d.conv.weight).cuda()
		conv3d.conv.weight.data = conv.weight.data[:,:,None,:,:].expand_as(conv3d.conv.weight).clone() * factor
	elif scheme==3:
		conv3d.conv.weight.data = conv.weight.data[:,:,None,:,:].expand_as(conv3d.conv.weight).clone() * (1./tempSize)
	conv3d.conv.bias.data = conv.bias.data
	conv3d.conv.weight.data = conv3d.conv.weight.data.contiguous()
	conv3d.conv.bias.data = conv3d.conv.bias.data.contiguous()
	return

def inflatebn(bn3d, bn):
	"""
	bn3d.bn.weight.data = bn.weight.data
	bn3d.bn.bias.data = bn.bias.data
	bn3d.bn.running_mean = bn.running_mean
	bn3d.bn.running_var = bn.running_var
	bn3d.bn.weight.data = bn3d.bn.weight.data.contiguous()
	bn3d.bn.weight.data = bn3d.bn.weight.data.contiguous()
	bn3d.bn.running_mean = bn3d.bn.running_mean.contiguous()
	bn3d.bn.running_var = bn3d.bn.running_var.contiguous()
	"""
	bn.track_running_stats = True
	return bn

def inflaterelu(relu3d, relu):
	return

def inflateMaxPool(mp3d, mp):
	return		

def inflateResidual(res3d, res):
	res3d.cb.cbr1.bn.bn = inflatebn(res3d.cb.cbr1.bn, res.bn)
	inflaterelu(res3d.cb.cbr1.relu, res.relu)
	inflateconv(res3d.cb.cbr1.conv, res.conv1)
	res3d.cb.cbr2.bn.bn = inflatebn(res3d.cb.cbr2.bn, res.bn1)
	inflaterelu(res3d.cb.cbr2.relu, res.relu)
	inflateconv(res3d.cb.cbr2.conv, res.conv2)	
	res3d.cb.cbr3.bn.bn = inflatebn(res3d.cb.cbr3.bn, res.bn2)
	inflaterelu(res3d.cb.cbr3.relu, res.relu)
	inflateconv(res3d.cb.cbr3.conv, res.conv3)
	if (res3d.inChannels != res3d.outChannels):
		inflateconv(res3d.skip.conv, res.conv4)
	return

def inflateupsampling(up3d, up):
	return	
