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
	opt.DEBUG = 2

	if opt.loadModel == 'none':
		model = inflate(opt).cuda()
	elif opt.loadModel == 'scratch':
		model = Pose3D(opt.nChannels, opt.nStack, opt.nModules, opt.numReductions, opt.nRegModules, opt.nRegFrames, ref.nJoints).cuda()
	else :
		model = torch.load(opt.loadModel).cuda()


	val_loader1 = torch.utils.data.DataLoader(
		h36m('val',opt),
		batch_size = 1,
		shuffle = False,
		num_workers = int(ref.nThreads)
	)

	val_loader2 = torch.utils.data.DataLoader(
		posetrack('val',opt),
		batch_size = 1,
		shuffle = True,
		num_workers = int(ref.nThreads)
	)



	val(0, opt, val_loader2, model)
	val(0, opt, val_loader2, model)



	
if __name__ == '__main__':
	#torch.set_default_tensor_type('torch.DoubleTensor')
	main()
