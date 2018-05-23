import torch


nChannels = 128
nStack = 2
nModules = 2
nRegFrames = 16

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
	for i in range(nRegFrames):
		model3d.bias.data[16*i:16*(i+1)] = model.bias.data
		for j in range(nRegFrames):
			if (i == j) :
				model3d.weight.data[16*(i):16*(i+1), 2048*(j):2048*(j+1)] = model.weight.data / 16.0
			else :
				model3d.weight.data[16*(i):16*(i+1), 2048*(j):2048*(j+1)] = model.weight.data / 16.0
def inflateHourglassNet(model3d, model):
	inflateconv(model3d.convStart, model.conv1_)
	model3d.bnStart = inflatebn(model3d.bnStart, model.bn1)
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
		inflateconv(model3d.lin1[i].conv, model.lin_[i][0])
		model3d.lin1[i].bn = inflatebn(model3d.lin1[i].bn, model.lin_[i][1])
		inflaterelu(model3d.lin1[i].relu, model.lin_[i][2])
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
		inflateResidual(model3d.beforepool[i], model.low1_[i])

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
	conv3d.weight.data = conv.weight.data[:,:,None,:,:].expand(conv3d.weight.data.size()) * (1./(conv3d.weight.data.shape[2]))
	conv3d.bias.data = conv.bias.data
	conv3d.weight.data = conv3d.weight.data.contiguous()
	conv3d.bias.data = conv3d.bias.data.contiguous()
	return

def inflatebn(bn3d, bn):
	"""
	bn3d.weight.data = bn.weight.data
	bn3d.bias.data = bn.bias.data
	bn3d.running_mean = bn.running_mean
	bn3d.running_var = bn.running_var
	bn3d.weight.data = bn3d.weight.data.contiguous()
	bn3d.weight.data = bn3d.weight.data.contiguous()
	bn3d.running_mean = bn3d.running_mean.contiguous()
	bn3d.running_var = bn3d.running_var.contiguous()
	"""
	#bn3d = bn
	return bn

def inflaterelu(relu3d, relu):
	return

def inflateMaxPool(mp3d, mp):
	return		

def inflateResidual(res3d, res):
	res3d.cb.cbr1.bn = inflatebn(res3d.cb.cbr1.bn, res.bn)
	inflaterelu(res3d.cb.cbr1.relu, res.relu)
	inflateconv(res3d.cb.cbr1.conv, res.conv1)
	res3d.cb.cbr2.bn = inflatebn(res3d.cb.cbr2.bn, res.bn1)
	inflaterelu(res3d.cb.cbr2.relu, res.relu)
	inflateconv(res3d.cb.cbr2.conv, res.conv2)	
	res3d.cb.cbr3.bn = inflatebn(res3d.cb.cbr3.bn, res.bn2)
	inflaterelu(res3d.cb.cbr3.relu, res.relu)
	inflateconv(res3d.cb.cbr3.conv, res.conv3)
	if (res3d.inChannels != res3d.outChannels):
		inflateconv(res3d.skip.conv, res.conv4)
	return

def inflateupsampling(up3d, up):
	return	
