import torch
import random
import numpy as np
import torch.utils.data as data
import pickle
from datahelpers.dataloaders.h36mLoader import *
from datahelpers.dataloaders.myposetrackLoader import posetrack

class FusionDataset(data.Dataset):
	"""docstring for FusionDataset"""
	def __init__(self, split, opts):
		super(FusionDataset, self).__init__()
		self.split = split

		self.dataset_h36m = h36m(split, opts)
		self.dataset_posetrack = posetrack(split, opts)

		self.nVideos_h36m = len(self.dataset_h36m)
		self.nVideos_posetrack = min(len(self.dataset_posetrack), self.nVideos_h36m//opts.ratioHM)


		print("Built h36m and posetrack dataset containing %d and %d samples" %(self.nVideos_h36m, self.nVideos_posetrack))

	def __getitem__(self, index):
		if (index < self.nVideos_h36m):
			return self.dataset_h36m[index]
		else :
			return self.dataset_posetrack[index - self.nVideos_h36m]

	def __len__(self):
		return self.nVideos_h36m + self.nVideos_posetrack
