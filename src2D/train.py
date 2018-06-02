import matplotlib
import cv2
import ref
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.SoftArgMax import *
from progress.bar import Bar

from utils.utils import AverageMeter

#from utils.debugger import Debugger
from utils.eval import *
from Losses import *
from visualise_model import *


SoftArgMaxLayer = SoftArgMax()

def step(split, epoch, opt, dataLoader, model, optimizer = None):
	if split == 'train':
		model.train()
	else:
		model.eval()
	Loss2D, Acc = AverageMeter(), AverageMeter()

	nIters = len(dataLoader)
	bar = Bar('==>', max=nIters)

	for i, (input, targetMaps, target2D, target3D, meta) in enumerate(dataLoader):
		input_var = (input).float().cuda()
		# for i in range(16):
		# 	ploter = input_var[0,:,i,:,:].transpose(0,1).transpose(1,2).data.cpu().numpy()
		# 	plt.imshow(ploter)
		# 	plt.show()
		targetMaps = (targetMaps).float().cuda()
		target2D_var = (target2D).float().cuda()
		target3D_var = (target3D).float().cuda()
		model = model.float()
		output = model(input_var)[0]
		loss = 0
			
		if opt.DEBUG >= 2:
			for i in range(16):
				plt.imshow(targetMaps[0,0,i,:,:].data.cpu().numpy(), cmap='hot', interpolation='nearest')
				plt.show()
				a = getPreds(targetMaps[:,:,i,:,:].cpu().numpy())
				b = getPreds(output[opt.nStack - 1][:,:,i,:,:].data.cpu().numpy())
				print(torch.argmax(targetMaps[0,0,i,:,:]), a[0,0,:], b[0,0,:])
		
		for k in range(opt.nStack):
			loss += Joints2DHeatMapsSquaredError(output[k], targetMaps)
		
		Loss2D.update(loss.item(), input.size(0))

		Acc.update(Accuracy((output[opt.nStack - 1].data).transpose(1,2).reshape(-1,ref.nJoints,ref.outputRes,ref.outputRes).cpu().numpy(), (targetMaps.data).transpose(1,2).reshape(-1,ref.nJoints,ref.outputRes,ref.outputRes).cpu().numpy()))

		if split == 'train':
			loss = loss/opt.trainBatch
			loss.backward()
			if ((opt.dataloaderSize*(i+1))%opt.trainBatch == 0):
				optimizer.step()
				optimizer.zero_grad()
		# for j in range(16):
		# 	a = np.zeros((16,3))
		# 	b = np.zeros((16,3))	
		# 	a[:,:2] = getPreds(output[opt.nStack - 1][:1,:,i,:,:].detach().cpu().numpy())
		# 	b[:,:2] = getPreds(targetMaps[:1,:,i,:,:].detach().cpu().numpy())
			#print((targetMaps[:1,:,0,:,:] == targetMaps[:1,:,i,:,:]).all())
			#print((input_var[:1,:,0,:,:] == input_var[:1,:,i,:,:]).all())
		#	visualise3d(b,a,epoch,j)
		Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss2D {loss.avg:.6f} | PCK {PCK.avg:.6f} {PCK.val:.6f}'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss2D, split = split, PCK = Acc)
		bar.next()

	bar.finish()
	return Loss2D.avg, Acc.avg


def train(epoch, opt, train_loader, model, optimizer):
	return step('train', epoch, opt, train_loader, model, optimizer)

def val(epoch, opt, val_loader, model):
	with torch.no_grad():
		return step('val', epoch, opt, val_loader, model)
