import cv2
import ref
import torch
import random
import numpy as np
import torch.utils.data as data
import pickle

from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform3D

class h36m(data.Dataset):
	"""docstring for h36m"""
	def __init__(self, split, opts):
		super(h36m, self).__init__()
		print("==> Initializing 3D %s data for h36m data" %(split))
		#self.split = split
		self.nFramesLoad = opts.nFramesLoad
		self.loadConsecutive = opts.loadConsecutive
		self.vidFolders = np.load(ref.h36mDataDir + "/vid_" + split + ".npy")
		self.countFrames = np.load(ref.h36mDataDir + "/cnt_" + split + ".npy")

		self.root = 7
		self.split = split

		self.nVideos = (self.vidFolders).shape[0]

		print("Loaded %d %s videos for h36m data" %(self.nVideos, split))

	def LoadFrameAndData(self, path, frameName):
		frame = cv2.imread(path+frameName)
		
		pts_2d, pts_3d, pts_3d_mono = pickle.load(open(path + "data.pkl",'rb'))[int(frameName[-10:-4])]
		

		pts_2d = pts_2d
		pts_3d = pts_3d
		pts_3d_mono = pts_3d_mono


		c = np.ones(2) * ref.h36mImgSize / 2
		s = ref.h36mImgSize * 1.0

		pts_3d = pts_3d - pts_3d[self.root]

		s2d, s3d = 0, 0
		for e in ref.edges:
			s2d += ((pts_2d[e[0]] - pts_2d[e[1]]) ** 2).sum() ** 0.5
			s3d += ((pts_3d[e[0], :2] - pts_3d[e[1], :2]) ** 2).sum() ** 0.5
		scale = s2d / s3d

		for j in range(ref.nJoints):
			pts_3d[j, 0] = pts_3d[j, 0] * scale + pts_2d[self.root, 0]
			pts_3d[j, 1] = pts_3d[j, 1] * scale + pts_2d[self.root, 1]
			pts_3d[j, 2] = pts_3d[j, 2] * scale + ref.h36mImgSize / 2

		pts_3d[7,:] = (pts_3d[12,:] + pts_3d[13,:]) / 2


		frame = Crop(frame, c, s, 0, ref.inputRes) / 256.

		outMap = np.zeros((ref.nJoints, ref.outputRes, ref.outputRes))
		outReg = np.zeros((ref.nJoints, 3))
		for i in range(ref.nJoints):
			pt = Transform3D(pts_3d[i], c, s, 0, ref.outputRes)
			if pts_2d[i][0] > 1:
				outMap[i] = DrawGaussian(outMap[i], pt[:2], ref.hmGauss)
			outReg[i, 2] = pt[2] / ref.outputRes * 2 - 1

		return frame, outMap, pts_2d, outReg, pts_3d_mono



	def __getitem__(self, index):

		if (self.split == 'train'):
			index = np.random.randint(self.nVideos)

		vidFolder = self.vidFolders[index]

		path = ref.h36mDataDir + "/" + vidFolder + "/"

		CountFramesInVid = self.countFrames[index]

		if self.loadConsecutive:

			fpsFac = 1

			startPt = random.randint(1, CountFramesInVid - fpsFac*(self.nFramesLoad + 2))
			inpFrames = np.zeros((3,self.nFramesLoad,256,256))
			outPts_2ds = np.zeros((ref.nJoints,self.nFramesLoad,2))
			outOutRegs = np.zeros((ref.nJoints,self.nFramesLoad,3))
			outPts_3d_monos = np.zeros((ref.nJoints,self.nFramesLoad,3))
			outOutMaps = np.zeros((ref.nJoints, self.nFramesLoad, ref.outputRes, ref.outputRes))

			for i in range(self.nFramesLoad):
				frameIndex = "{:06d}.jpg".format(fpsFac*(startPt//fpsFac + i) + 1)
				frame,outMap,pts_2d,outReg,pts_3d_mono = self.LoadFrameAndData(path, vidFolder + "_" + frameIndex)
				inpFrames[:,i,:,:] = frame
				outOutMaps[:,i,:,:] = outMap
				outPts_2ds[:,i,:] = pts_2d
				outOutRegs[:,i,:] = outReg
				outPts_3d_monos[:,i,:] = pts_3d_mono
		else :

			frameIndices = np.random.permutation(CountFramesInVid)
			selectedFrameIndices = frameIndices[:self.nFramesLoad]

			inpFrames = np.zeros((3,self.nFramesLoad,256,256))
			outPts_2ds = np.zeros((ref.nJoints,self.nFramesLoad,2))
			outOutRegs = np.zeros((ref.nJoints,self.nFramesLoad,3))
			outPts_3d_monos = np.zeros((ref.nJoints,self.nFramesLoad,3))
			outOutMaps = np.zeros((ref.nJoints, self.nFramesLoad, ref.outputRes, ref.outputRes))

			for i in range(self.nFramesLoad):
				ithFrameIndex = "{:06d}.jpg".format(5*selectedFrameIndices[i] - 4)
				frame,outMap,pts_2d,outReg,pts_3d_mono = self.LoadFrameAndData(path, vidFolder + "_" + frameIndex)
				inpFrames[:,i,:,:] = frame
				outOutMaps[:,i,:,:] = outMap
				outPts_2ds[:,i,:] = pts_2d
				outOutRegs[:,i,:] = outReg
				outPts_3d_monos[:,i,:] = pts_3d_mono
			
		outOutRegs = outOutRegs[:,:,2:]
		return (inpFrames, outOutMaps, outPts_2ds, outOutRegs, outPts_3d_monos)

	def __len__(self):
		return self.nVideos
