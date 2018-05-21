import cv2
import ref
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.SoftArgMax import *
from progress.bar import Bar

from utils.utils import AverageMeter

#from utils.debugger import Debugger

from Losses import *

SoftArgMaxLayer = SoftArgMax()

def step(split, epoch, opt, dataLoader, model, optimizer = None):
	if split == 'train':
		model.train()
	else:
		model.eval()
	Loss2D, Loss3D, = AverageMeter(), AverageMeter()

	nIters = len(dataLoader)
	bar = Bar('==>', max=nIters)
	
	for i, (input, targetMaps, target2D, target3D, meta) in enumerate(dataLoader):
		input_var = (input).float().cuda()
		targetMaps = (targetMaps).float().cuda()
		target2D_var = (target2D).float().cuda()
		target3D_var = (target3D).float().cuda()
		output = model(input_var)

		reg = output[opt.nStack]
		
		if opt.DEBUG >= 2:
			gt = getPreds(target2D.cpu().numpy()) * 4
			pred = getPreds((output[opt.nStack - 1].data).cpu().numpy()) * 4
			debugger = Debugger()
			debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
			debugger.addPoint2D(pred[0], (255, 0, 0))
			debugger.addPoint2D(gt[0], (0, 0, 255))
			debugger.showImg()
			debugger.saveImg('debug/{}.png'.format(i))

		loss = opt.regWeight * JointsDepthSquaredError(reg,target3D_var)
		
		Loss3D.update(loss.item(), input.size(0))
		
		loss = 0
		for k in range(opt.nStack):
			#loss += Joints2DArgMaxSquaredError(SoftArgMaxLayer(output[k]), target2D_var)
			# plt.imshow(output[k].detach().cpu().numpy()[0,0,0,:,:], cmap='hot', interpolation='nearest')
			# plt.show()
			# plt.imshow(targetMaps.cpu().numpy()[0,0,0,:,:], cmap='hot', interpolation='nearest')
			# plt.show()
			loss += Joints2DHeatMapsSquaredError(output[k], targetMaps)
		Loss2D.update(loss.item(), input.size(0))
		

		if split == 'train':
			loss.backward()
			if ((i+1)%(opt.trainBatch/2) == 0):
				optimizer.step()
				optimizer.zero_grad()

 
		Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss2D {loss.avg:.6f} | Loss3D {loss3d.avg:.6f}'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss2D, split = split, loss3d = Loss3D)
		bar.next()

	bar.finish()
	return Loss2D.avg, Loss3D.avg


def train(epoch, opt, train_loader, model, optimizer):
	return step('train', epoch, opt, train_loader, model, optimizer)
	
def val(epoch, opt, val_loader, model):
	return step('val', epoch, opt, val_loader, model)
