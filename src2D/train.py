import cv2
import ref
import torch
import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt


from my import *
from Losses import *
from utils.eval import *
from visualise_model import *
from utils.utils import *

from model.SoftArgMax import *
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
		targetMaps = (targetMaps).float().cuda()
		target2D_var = (target2D).float().cuda()
		target3D_var = (target3D).float().cuda()
		model = model.float()
		output = model(input_var)[0]

		if opt.DEBUG == 2:
			print("WHAT")
			for i in range(input_var.shape[2]):
				plt.imshow(input_var.data[0,:,i,:,:].transpose(0,1).transpose(1,2).cpu().numpy())

				a = np.zeros((16,3))
				b = np.zeros((16,3))
				a[:,:2] = getPreds(targetMaps[:,:,i,:,:].cpu().numpy())
				b[:,:2] = getPreds(output[opt.nStack - 1][:,:,i,:,:].data.cpu().numpy())
				visualise3d(b,a,epoch,i)
		
		loss = 0
		for k in range(opt.nStack):
			loss += Joints2DHeatMapsSquaredError(output[k], targetMaps)
		
		Loss2D.update(loss.item(), input.size(0))

		tempAcc = Accuracy((output[opt.nStack - 1].data).transpose(1,2).reshape(-1,ref.nJoints,ref.outputRes,ref.outputRes).cpu().numpy(), (targetMaps.data).transpose(1,2).reshape(-1,ref.nJoints,ref.outputRes,ref.outputRes).cpu().numpy())
		Acc.update(tempAcc)


		if opt.DEBUG == 3 and (float(tempAcc) < 0.80):
			print("LEFT")
			for j in range(input_var.shape[2]):
				a = np.zeros((16,3))
				b = np.zeros((16,3))
				a[:,:2] = getPreds(targetMaps[:,:,j,:,:].cpu().numpy())
				b[:,:2] = getPreds(output[opt.nStack - 1][:,:,j,:,:].data.cpu().numpy())
				visualise3d(b,a,'val-errors',i,j,input_var.data[0,:,j,:,:].transpose(0,1).transpose(1,2).cpu().numpy())


		if split == 'train':
			loss = loss/opt.trainBatch
			loss.backward()
			if ((opt.dataloaderSize*(i+1))%opt.trainBatch == 0):
				optimizer.step()
				optimizer.zero_grad()
		
		Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss2D {loss.avg:.6f} | PCK {PCK.avg:.6f} {PCK.val:.6f}'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss2D, split = split, PCK = Acc)
		bar.next()

	bar.finish()
	return Loss2D.avg, Acc.avg


def train(epoch, opt, train_loader, model, optimizer):
	return step('train', epoch, opt, train_loader, model, optimizer)

def val(epoch, opt, val_loader, model):
	with torch.no_grad():
		return step('val', epoch, opt, val_loader, model)
