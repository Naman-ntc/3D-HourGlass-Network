import ref
import cv2
import torch
import numpy as np
from h5py import File
import torch.utils.data as data
from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform

class mpii(data.Dataset):
	def __init__(self, split, opts, returnMeta = True):
		print('==> initializing 2D {} data.'.format(split))
		annot = {}
		tags = ['imgname','part','center','scale']
		f = File('{}/{}.h5'.format(ref.mpiiDataDir, split), 'r')
		for tag in tags:
			annot[tag] = np.asarray(f[tag]).copy()
		f.close()

		self.nVideos = len(annot['scale'])

		print('Loaded 2D {} {} static videos for mpii 2D data'.format(split, self.nVideos))

		self.split = split
		self.opts = opts
		self.nFramesLoad = opts.nFramesLoad
		self.annot = annot
		self.returnMeta = returnMeta

	def LoadImage(self, index):
		path = '{}/{}'.format(ref.mpiiDataDir, str(self.annot['imgname'][index])[2:-1])
		img = cv2.imread(path)
		return img

	def GetPartInfo(self, index):
		pts = self.annot['part'][index].copy()
		c = self.annot['center'][index].copy()
		s = self.annot['scale'][index]
		s = s * 200
		return pts, c, s

	def getitem(self, index):
		img = self.LoadImage(index)
		pts, c, s = self.GetPartInfo(index)
		r = 0

		if self.split == 'train':
			s = s * (2 ** Rnd(ref.scale))
			r = 0 if np.random.random() < 0.6 else Rnd(ref.rotate)
		inp = Crop(img, c, s, r, ref.inputRes) / 256.
		out = np.zeros((ref.nJoints, ref.outputRes, ref.outputRes))
		Reg = np.zeros((ref.nJoints, 3))
		for i in range(ref.nJoints):
			if pts[i][0] > 1:
				pt = Transform(pts[i], c, s, r, ref.outputRes)
				out[i] = DrawGaussian(out[i], pt, ref.hmGauss) 
				Reg[i, :2] = pt
				Reg[i, 2] = 1
		if self.split == 'train':
			if np.random.random() < 0.5:
				inp = Flip(inp)
				out = ShuffleLR(Flip(out))
				Reg[:, 1] = Reg[:, 1] * -1
				Reg = ShuffleLR(Reg)
			#print 'before', inp[0].max(), inp[0].mean()
			inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
			inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
			inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)
			#print 'after', inp[0].max(), inp[0].mean()


		meta = (np.zeros((ref.nJoints, 3)))
		if self.returnMeta:
			return inp, out, Reg, meta
		else:
			return inp, out

	def __getitem__(self, index):
		a,b,c,d = self.getitem(index)
		a = np.repeat(a[:,None,:,:],self.nFramesLoad,axis=1).astype('d')
		b = np.repeat(b[:,None,:,:],self.nFramesLoad,axis=1)
		c = np.repeat(c[:,None,2:],self.nFramesLoad,axis=1)
		d = np.repeat(d[:,None,:2],self.nFramesLoad,axis=1)
		e = -1*np.ones((ref.nJoints,self.nFramesLoad,3))
		return (a,b,d,c,e)

	def __len__(self):
		return self.nVideos
