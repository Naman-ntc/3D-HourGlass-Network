import os
import time
import datetime

import ref
import torch
import torch.utils.data
from opts import opts
from model.Pose3D import Pose3D

from datahelpers.dataloaders.fusedDataLoader import FusionDataset
from datahelpers.dataloaders.h36mLoader import h36m
from datahelpers.dataloaders.mpiiLoader import mpii
from datahelpers.dataloaders.posetrackLoader import posetrack

from utils.utils import adjust_learning_rate
from utils.logger import Logger

from xingy_train import train,val
from inflateScript import *


def main():
	opt = opts().parse()
	torch.cuda.set_device(opt.gpu_id) 
	print('Using GPU ID: ' ,str(torch.cuda.current_device())) 
	now = datetime.datetime.now()
	logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

	import pickle
	from functools import partial
	pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
	pickle.load = partial(pickle.load, encoding="latin1")
	model = torch.load('models/xingy.pth').cuda()

	val_loader = torch.utils.data.DataLoader(
		h36m('val', opt),
		batch_size = 1,
		shuffle = False,
		num_workers = int(ref.nThreads)
	)

	
	if opt.completeTest:
		mp = 0.
		cnt = 0.
		for i in range(6000//opt.nVal):
			opt.startVal = 120*i
			opt.nVal = opt.nVal
			a,b = val(i, opt, val_loader, model)
			mp += a*b
			cnt += b
		print("------Finally--------")
		print("Final MPJPE ==> :" +  str(mp/cnt))	
		return

	if (opt.test):
		val(0, opt, val_loader, model)
		return


	train_loader = torch.utils.data.DataLoader(
		FusionDataset('train',opt) if opt.loadMpii else h36m('train',opt),
		batch_size = opt.dataloaderSize,
		shuffle = True,
		num_workers = int(ref.nThreads)
	)

	optimizer = torch.optim.RMSprop(
		[{'params': model.hg.parameters(), 'lr': opt.LRhg},
		{'params': model.dr.parameters(), 'lr': opt.LRdr}], 
		alpha = ref.alpha, 
		eps = ref.epsilon, 
		weight_decay = ref.weightDecay, 
		momentum = ref.momentum
	)
	

	def hookdef(grad):
		newgrad = grad.clone()
		if (grad.shape[2]==1):
			newgrad = grad*opt.freezefac
		else:
			newgrad[:,:,1,:,:] = grad[:,:,1,:,:]*opt.freezefac
		return newgrad
			
	def hookdef1(grad):
		newgrad = grad.clone()
		newgrad[:,4096:8192] = newgrad[:,4096:8192]*opt.freezefac
		return newgrad

	for i in (model.parameters()):
		if len(i.shape)==5:
			_ = i.register_hook(hookdef)
		if len(i.shape)==2:
			_ = i.register_hook(hookdef1)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = opt.dropMag, patience = opt.patience, verbose = True, threshold = opt.threshold)

	for epoch in range(1, opt.nEpochs + 1):
		loss_train, loss3d_train, mpjpe_train, acc_train = train(epoch, opt, train_loader, model, optimizer)
		logger.scalar_summary('loss_train', loss_train, epoch)
		#logger.scalar_summary('acc_train', acc_train, epoch)
		logger.scalar_summary('mpjpe_train', mpjpe_train, epoch)
		logger.scalar_summary('loss3d_train', loss3d_train, epoch)
		if epoch % opt.valIntervals == 0:
			loss_val, loss3d_val, mpjpe_val, acc_val = val(epoch, opt, val_loader, model)
			logger.scalar_summary('loss_val', loss_val, epoch)
		# 	logger.scalar_summary('acc_val', acc_val, epoch)
			logger.scalar_summary('mpjpe_val', mpjpe_val, epoch)
			logger.scalar_summary('loss3d_val', loss3d_val, epoch)
			torch.save(model.state_dict(), os.path.join(opt.saveDir, 'model_{}.pth'.format(epoch)))
			logger.write('{:8f} {:8f} {:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, mpjpe_train, loss3d_train, acc_val, loss_val, mpjpe_val, loss3d_val, acc_train))
		else:
			logger.write('{:8f} {:8f} {:8f} \n'.format(loss_train, mpjpe_train, loss3d_train, acc_train))
		#adjust_learning_rate(optimizer, epoch, opt.dropLR, opt.LR)
		if opt.scheduler == 1:
			scheduler.step(int(loss_train))
		elif opt.scheduler == 2:
			scheduler.step(int(loss3d_train))
		elif opt.scheduler == 3:
			scheduler.step(int(loss_train + loss3d_train))	
		elif opt.scheduler == 4:
			scheduler.step(int(mpjpe_train))

				
	logger.close()

if __name__ == '__main__':
	#torch.set_default_tensor_type('torch.DoubleTensor')
	main()
