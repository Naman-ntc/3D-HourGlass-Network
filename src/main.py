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
from datahelpers.dataloaders.ntuLoader import ntu

from utils.utils import adjust_learning_rate
from utils.logger import Logger




def main():
	opt = opts().parse()
	now = datetime.datetime.now()
	logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

	if opt.loadModel != 'none':
		model = torch.load(opt.loadModel).cuda()
	else :
		model = Pose3D(opt.nChannels, opt.nStack, opt.nModules, opt.numReductions, opt.nRegModules, opt.nRegFrames, ref.nJoints)


	val_loader = torch.utils.data.DataLoader(
		h36m('val', opt),
		batch_size = 1,
		shuffle = False,
		num_workers = int(ref.nThreads)
	)

	if (opt.test):
		# Validate!!!
		pass

		
	train_loader = torch.utils.data.DataLoader(
		h36m('train', opt),
		batch_size = opt.trainBatch,
		shuffle = True if opt.DEBUG == 0 else False,
		num_workers = int(ref.nThreads)
	)

	optimizer = torch.optim.RMSprop(
		model.parameters(), opt.LR, 
		alpha = ref.alpha, 
		eps = ref.epsilon, 
		weight_decay = ref.weightDecay, 
		momentum = ref.momentum
	)





if __name__ == '__main__':
	main()
