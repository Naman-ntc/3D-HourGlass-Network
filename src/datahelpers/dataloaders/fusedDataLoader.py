import torch
import random
import numpy as np
import torch.utils.data as data
import pickle
							
class FusionDataset(data.Dataset):
	"""docstring for FusionDataset"""
	def __init__(self, split, opts):
		super(FusionDataset, self).__init__()
		self.split = split
		
		self.dataset_h36m = h36m(split, opts)
		self.dataset_ntu = ntu(split, opts)

		self.nVideos_h36m = len(self.dataset_h36m)
		self.nVideos_ntu = len(self.dataset_ntu)

		print("Built h36m and mpii dataset containing %d and %d samples")

	def __getitem__(self, index):
		if (index < self.nImagesh36m):
			return self.dataset_h36m[index]
		else :
			return self.dataset_ntu[index - self.nVideos_h36m]

	def __len__(self):
		return self.nVideos_h36m + self.nVideos_mpii