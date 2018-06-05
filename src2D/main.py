import os
import time
import datetime

import ref
import torch
import torch.utils.data
from opts import opts
from model.Pose3D import Pose3D

from datahelpers.dataloaders.oldfusedDataLoader import FusionDataset
from datahelpers.dataloaders.h36mLoader import h36m
from datahelpers.dataloaders.mpiiLoader import mpii
from datahelpers.dataloaders.myposetrackLoader import posetrack

from utils.utils import adjust_learning_rate
from utils.logger import Logger

from train import train,val
from inflateScript import *


def main():
	opt = opts().parse()
	now = datetime.datetime.now()
	logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

	if opt.loadModel == 'none':

		model = inflate(opt).cuda()
	elif opt.loadModel == 'scratch':
		model = HourglassNet3D(opt.nChannels, opt.nStack, opt.nModules, opt.numReductions, ref.nJoints).cuda()
	else :
		model = torch.load(opt.loadModel).cuda()


	val_loader1 = torch.utils.data.DataLoader(
		h36m('val',opt),
		batch_size = 1,
		shuffle = False,
		num_workers = int(ref.nThreads)
	)

	val_loader2 = torch.utils.data.DataLoader(
		mpii('val',opt),
		batch_size = 1,
		shuffle = False,
		num_workers = int(ref.nThreads)
	)

	if (opt.test):
		val(0, opt, val_loader2, model)
		val(0, opt, val_loader1, model)
		pass


	train_loader = torch.utils.data.DataLoader(
		#h36m('train',opt),
		FusionDataset('train',opt),
		#posetrack('train', opt),
		batch_size = opt.dataloaderSize,
		shuffle = True,
		num_workers = int(ref.nThreads)
	)

	optimizer = torch.optim.RMSprop(
		[{'params': model.parameters(), 'lr': opt.LRhg}], 
		alpha = ref.alpha, 
		eps = ref.epsilon, 
		weight_decay = ref.weightDecay, 
		momentum = ref.momentum
	)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = opt.dropMag, patience = opt.patience, verbose = True, threshold = opt.threshold)

	for epoch in range(1, opt.nEpochs + 1):
		loss_train, acc_train = train(epoch, opt, train_loader, model, optimizer)
		logger.scalar_summary('loss_train', loss_train, epoch)
		logger.scalar_summary('acc_train', acc_train, epoch)
		#logger.scalar_summary('mpjpe_train', mpjpe_train, epoch)
		#logger.scalar_summary('loss3d_train', loss3d_train, epoch)
		if epoch % opt.valIntervals == 0:
			loss_val, acc_val = val(epoch, opt, val_loader1, model)
			logger.scalar_summary('loss_val', loss_val, epoch)
			logger.scalar_summary('acc_val', acc_val, epoch)
			logger.write('{:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, acc_train, loss_val, acc_val))
			loss_val, acc_val = val(epoch, opt, val_loader2, model)
			logger.scalar_summary('loss_val', loss_val, epoch)
			logger.scalar_summary('acc_val', acc_val, epoch)
			logger.write('{:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, acc_train, loss_val, acc_val))
			#logger.scalar_summary('mpjpe_val', mpjpe_val, epoch)
			#logger.scalar_summary('loss3d_val', loss3d_val, epoch)
			torch.save(model, os.path.join(opt.saveDir, 'model_{}.pth'.format(epoch)))
		else:
			logger.write('{:8f} {:8f} \n'.format(loss_train, acc_train))
		#adjust_learning_rate(optimizer, epoch, opt.dropLR, opt.LR)
		if opt.scheduler == 1:
			scheduler.step(int(loss_train))


				
	logger.close()

if __name__ == '__main__':
	#torch.set_default_tensor_type('torch.DoubleTensor')
	main()
