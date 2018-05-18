import ref
import torch
import random
import numpy as np
import torch.utils.data as data
import pickle


class h36m(data.dataset):
	"""docstring for h36m"""
	def __init__(self, opts, split, nFramesLoad, loadConsecutive = True):
		super(h36m, self).__init__()
		print("Initializing 3D %s data for h3.6m data" %(split))
		self.opts = opts
		self.split = split
		self.nFramesLoad = nFramesLoad
		self.loadConsecutive = loadConsecutive
		self.vidFolders = np.load(ref.h36mdir + "/vid_" + split + ".npy")
		self.countFrames = np.load(ref.h36mdir + "/cnt_" + split + ".npy")
		
		self.root = 7
		self.split = split
		self.opts = opts
		self.annot = annot

		self.nVideos = (self.vidFolders).shape[0]

	

	def LoadFrameAndData(self, path, frameName):
		frame = cv2.imread(path+frameName)
		pts_2d, pts_3d, pts_3d_mono = pickle.load(path + "data.pkl")
		
		pts_2d = pts_2d[frameName]
		pts_3d = pts_3d[frameName]
		pts_3d_mono = pts_3d_mono[frameName]


		c = np.ones(2) * ref.h36mImgSize / 2
		s = ref.h36mImgSize * 1.0
	
		pts_3d = pts_3d - pts_3d[self.root]
	
		s2d, s3d = 0, 0
		for e in ref.edges:
			s2d += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
			s3d += ((pts_3d[e[0], :2] - pts_3d[e[1], :2]) ** 2).sum() ** 0.5
		scale = s2d / s3d

		for j in range(ref.nJoints):
			pts_3d[j, 0] = pts_3d[j, 0] * scale + pts_2d[self.root, 0]
			pts_3d[j, 1] = pts_3d[j, 1] * scale + pts_2d[self.root, 1]
			pts_3d[j, 2] = pts_3d[j, 2] * scale + ref.h36mImgSize / 2
		
		pts_3d[7,:] = (pts_3d[12,:] + pts_3d[13,:]) / 2


		frame = Crop(frame, c, s, 0, ref.inputRes) / 256.

		outReg = np.zeros((ref.nJoints, 3))
		for i in range(ref.nJoints):
			pt = Transform3D(pts_3d[i], c, s, 0, ref.outputRes)
			outReg[i, 2] = pt[2] / ref.outputRes * 2 - 1

		frame = torch.from_numpy(frame)

		return frame, pts_2d, outReg, pts_3d_mono	



	def __getitem__(self, index):

		if (self.split == 'train'):
			index = np.random.randint(self.nVideos)

		vidFolder = self.vidFolders[index]

		path = ref.h36mdir + "/" + vidFolder + "/"

		CountFramesInVid = self.countFrames[index]

		if self.loadConsecutive:

			startPt = random.randint(1, CountFramesInVid - self.nFramesLoad + 2)
			inpFrames = np.zeros(3,self.nFramesLoad,:,:)		
			outPts_2ds = np.zeros(self.nFramesLoad,ref.nJoints,2)
			outOutRegs = np.zeros(self.nFramesLoad,ref.nJoints,3)
			outPts_3d_monos = np.zeros(self.nFramesLoad,ref.nJoints,3)

			for i in range(self.nFrames):
				frameIndex = "{:06d}.jpg".format(5*startPt-4)
				frame,pts_2d,outReg,pts_3d_mono = self.LoadFrameAndData(path, vidFolder + "_" + frameIndex)
				inpFrames[:,i,:,:] = frame
				outptss[i,:,:] = pts_2d
				outOutRegs[i,:,:] = outReg
				outPts_3d_monos[i,:,:] = pts_3d_mono
		else :

			frameIndices = np.random.permutation(CountFramesInVid)
			selectedFrameIndices = frameIndices[:self.nFramesLoad]

			inpFrames = np.zeros(3,self.nFramesLoad,:,:)		
			outPts_2ds = np.zeros(self.nFramesLoad,ref.nJoints,2)
			outOutRegs = np.zeros(self.nFramesLoad,ref.nJoints,3)
			outPts_3d_monos = np.zeros(self.nFramesLoad,ref.nJoints,3)
			
			for i in range(self.nFramesLoad):
				ithFrameIndex = "{:06d}.jpg".format(5*selectedFrameIndices[i] - 4)
				frame,pts_2d,outReg,pts_3d_mono = self.LoadFrameAndData(path, frameIndex)
				inpFrames[:,i,:,:] = frame
				outptss[i,:,:] = pts_2d
				outOutRegs[i,:,:] = outReg
				outPts_3d_monos[i,:,:] = pts_3d_mono
		

		return (inpFrames, outPts_2ds, outOutRegs, outPts_3d_monos)	