import torch.utils.data as data
import numpy as np



class h36m(data.dataset):
	"""docstring for h36m"""
	def __init__(self, opts, split):
		super(h36m, self).__init__()
		print("Initializing 3D %s data" %(split))
		self.opts = opts
		self.split = split
		annot = {}
    	tags = ['action', 'bbox', 'camera', 'id', 'joint_2d', 'joint_3d_mono', 'subaction', 'subject', 'istrain']
    	f = File('../data/h36m/annotSampleTest.h5', 'r')
    	for tag in tags:
    		annot[tag] = np.asarray(f[tag]).copy()
    	f.close()

		ids = np.arange(annot['id'].shape[0])[annot['istrain'] == (1 if split == 'train' else 0)]
    	for tag in tags:
    		annot[tag] = annot[tag][ids]    	


    	self.root = 7
    	self.split = split
    	self.opts = opts
    	self.annot = annot

    def 	

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


