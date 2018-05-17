import torch
import random
import numpy as np
import torch.utils.data as data
import pickle
							
class FusionDataset(data.Dataset):
	"""docstring for FusionDataset"""
	def __init__(self, opts, split):
		super(FusionDataset, self).__init__()
		self.opts = opts
		self.split = split

		self.dataseth36m = h36m(opts, split)
		self.datasetmpii = mpii(opts, split, True)

		self.nImagesh36m = len(self.dataseth36m)
		self.nImagesmpii = len(self.datasetmpii)

		print("Built h36m and mpii dataset containing %d and %d samples")

	def __getitem__(self, index):
		if (index < self.nImagesh36m):
			return self.dataseth36m[index]
		else :
			return self.datasetmpii[index - self.nImagesh36m]

	def __len__(self):
		return self.nImagesh36m + self.nImagesmpii