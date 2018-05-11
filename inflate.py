import torch


nChannels = 
nStack = 
nModules =

def inflateHourglassNet(model3d, model):
	inflateconv(model3d.cbrStart.conv, model.conv1_)
	inflatebn(model3d.cbrStart.bn, model.bn1)
	inflaterelu(model3d.cbrStart.relu, model.relu)
	inflateResidual(model3d.r1, model.res1)
	inflateResidual(model3d.r4, model.res2)
	inflateResidual(model3d.r5, model.res3)
	inflateMaxPool(model3d.mp, model.maxpool)
	for i in range(nStack):
		inflatehourglass(model3d.hourglass, model.hourglass)
	for i in range(nStack):
		for j in range(nModules):
			inflateResidual(model3d.Residual[i][j],model.Residual[nModules*i+j])
	for i in range(nStack):
		inflateconv(model3d.lin1[i].conv, model.lin_[i][0])
		inflatebn(model3d.lin1[i].bn, model.lin_[i][1])
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
	conv3d.weight.data = conv.weight.data[:,:,None,:,:].expand(torch.size())
	conv3d.bias.data = conv.bias.data
	return

def inflatebn(bn3d, bn):
	bn3d.weight.data = bn.weight.data
	bn3d.bias.data = bn.bias.data
	bn3d.running_mean.data = bn.running_mean.data
	bn3d.running_var.data = bn.running_var.data
	return

def inflaterelu(relu3d, relu):
	return

def inflateMaxPool(mp3d, mp):
	return		