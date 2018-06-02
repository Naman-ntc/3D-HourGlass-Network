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
		model = Pose3D(opt.nChannels, opt.nStack, opt.nModules, opt.numReductions, opt.nRegModules, opt.nRegFrames, ref.nJoints).cuda()
	else :
		model = torch.load(opt.loadModel).cuda()

	train_loader = torch.utils.data.DataLoader(
		h36m('train',opt),
		batch_size = opt.dataloaderSize,
		shuffle = False,
		num_workers = int(ref.nThreads)
	)

	optimizer = torch.optim.RMSprop(
		[{'params': model.parameters(), 'lr': opt.LRhg}], 
		alpha = ref.alpha, 
		eps = ref.epsilon, 
		weight_decay = ref.weightDecay, 
		momentum = ref.momentum
	)

	
	for epoch in range(1, opt.nEpochs + 1):
		loss_train, acc_train = train(epoch, opt, train_loader, model, optimizer)
		logger.scalar_summary('loss_train', loss_train, epoch)
		logger.scalar_summary('acc_train', acc_train, epoch)
		logger.write('{:8f} {:8f} \n'.format(loss_train, acc_train))

	logger.close()

if __name__ == '__main__':
	main()
