import ref
import cv2
import torch
import pickle
import random
import numpy as np
import torch.utils.data as data
from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform


class posetrack(data.Dataset):
	"""docstring for posetrack"""
	def __init__(self, split, opts, returnMeta = True):
		super(posetrack, self).__init__()
		self.split = split
		self.nFramesLoad = opts.nFramesLoad
		self.loadConsecutive = opts.loadConsecutive
		self.annotations = pickle.load(open(ref.posetrackDataDir + '/' + split + '.pkl','rb'))

		self.nVideos = len(self.annotations)

		print("Loaded %d %s videos for posetrack data" %(self.nVideos, split))


	def getitem(self, index, meta, whichFrames, frameWiseData):
		img = cv2.imread(ref.posetrackDataDir + "/" + whichFrames[index])
		#print(ref.posetrackDataDir + whichFrames[index])
		frameData = frameWiseData[whichFrames[index]]
		bboxmean,bboxdelta, joints = frameData[meta['personIndex']]

		pts = joints
		c = np.asarray(bboxmean)
		s = bboxdelta * meta['s'] / 100.0
		r = meta['r']

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
			if meta['flip']:
				inp = Flip(inp)
				out = ShuffleLR(Flip(out))
				Reg[:, 1] = Reg[:, 1] * -1
				Reg = ShuffleLR(Reg)

			inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
			inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
			inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)


		whatever = (np.zeros((ref.nJoints, 3)))

		return inp, out, Reg, whatever

	def __getitem__(self, index):
		if self.split=='train':
			index = np.random.randint(self.nVideos)
		whichFrames,frameWiseData = self.annotations[index]

		numFrames = len(whichFrames)

		if self.loadConsecutive:

			startpt = random.randint(1, numFrames - (self.nFramesLoad - 1))
			inpFrames = np.zeros((3,self.nFramesLoad,256,256))
			outPts_2ds = np.zeros((ref.nJoints,self.nFramesLoad,2))
			outOutRegs = np.ones((ref.nJoints,self.nFramesLoad,1))
			outPts_3d_monos = -1*np.ones((ref.nJoints,self.nFramesLoad,3))
			outOutMaps = np.zeros((ref.nJoints, self.nFramesLoad, ref.outputRes, ref.outputRes))
			meta = {}
			nPersons = len(frameWiseData[whichFrames[startpt ]])

			if nPersons<1:
				return self.__getitem__(index+1)
			personIndex = random.randint(0,nPersons-1)
			meta['personIndex'] = personIndex

			scale = 1
			angle = 0
			flipOrNot = 0

			if self.split=='train':
				angle = 0 if np.random.random() < 0.5 else Rnd(ref.rotate)
				scale = (2 ** Rnd(ref.scale))
				flipOrNot  = random.random() > 0.5
			meta['s'] = scale
			meta['r'] = angle
			meta['flip'] = flipOrNot

			for i in range(self.nFramesLoad):
				a,b,c,d = self.getitem(startpt+i,meta, whichFrames, frameWiseData)
				inpFrames[:,i,:,:] = a
				outOutMaps[:,i,:,:] = b

			return (inpFrames, outOutMaps, outPts_2ds, outOutRegs, outPts_3d_monos)


	def __len__(self):
		return self.nVideos
