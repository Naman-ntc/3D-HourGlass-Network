import ref
import torch
import random
import numpy as np
import torch.utils.data as data
import pickle


class ntu(data.dataset):
	"""docstring for ntu"""
	def __init__(self, split, nFramesLoad, loadConsecutive = True):
		super(ntu, self).__init__()
		print("Initializing 3D %s data for ntu data" %(split))
		self.split = split
		self.nFramesLoad = nFramesLoad
		self.loadConsecutive = loadConsecutive
		self.vidFolders = np.load(ref.ntuDataDir + "/vid_" + split + ".npy")
		self.countFrames = np.load(ref.ntuDataDir + "/cnt_" + split + ".npy")
		
		self.root = 7
		self.split = split
		self.annot = annot

		self.nVideos = (self.vidFolders).shape[0]


	def LoadFrameAndData(self, path, frameName):
		frame = cv2.imread(path+frameName)
		dict = pickle.load(open(path + "data.pkl",'rb'))
		
		pts_2d = pts_2d['2d'][int(frameName[:-4]),:]
		pts_3d = pts_3d['3d'][int(frameName[:-4]),:]
		pts_3d_mono = pts_3d_mono['3d'][int(frameName[:-4]),:]


		c = np.ones(2) * ref.ntuImgSize / 2
		s = ref.ntuImgSize * 1.0

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

		outReg = np.zeros((ref.nJoints, 3))
		for i in range(ref.nJoints):
			pt = Transform3D(pts_3d[i], c, s, 0, ref.outputRes)
			outReg[i, 2] = pt[2] / ref.outputRes * 2 - 1

		frame = torch.from_numpy(frame)
		pts_2d = torch.from_numpy(pts_2d)
		outReg = torch.from_numpy(outReg)
		pts_3d_mono = torch.from_numpy(pts_3d_mono)
		
		return frame, pts_2d, outReg, pts_3d_mono	



	def __getitem__(self, index):

		if (self.split == 'train'):
			index = np.random.randint(self.nVideos)

		vidFolder = self.vidFolders[index]

		path = ref.ntuDataDir + "/" + vidFolder + "/"

		CountFramesInVid = self.countFrames[index]

		if self.loadConsecutive:

			startPt = random.randint(1, CountFramesInVid - self.nFramesLoad + 2)
			inpFrames = np.zeros((3,self.nFramesLoad,256,256))		
			outPts_2ds = np.zeros((self.nFramesLoad,ref.nJoints,2))
			outOutRegs = np.zeros((self.nFramesLoad,ref.nJoints,3))
			outPts_3d_monos = np.zeros((self.nFramesLoad,ref.nJoints,3))

			for i in range(self.nFramesLoad):
				frameIndex = "{:06d}.jpg".format(startPt + i)
				frame,pts_2d,outReg,pts_3d_mono = self.LoadFrameAndData(path, frameIndex)
				inpFrames[:,i,:,:] = frame
				outPts_2ds[i,:,:] = pts_2d
				outOutRegs[i,:,:] = outReg
				outPts_3d_monos[i,:,:] = pts_3d_mono
		else :

			frameIndices = np.random.permutation(CountFramesInVid)
			selectedFrameIndices = frameIndices[:self.nFramesLoad]

			inpFrames = np.zeros((3,self.nFramesLoad,256,256))		
			outPts_2ds = np.zeros((self.nFramesLoad,ref.nJoints,2))
			outOutRegs = np.zeros((self.nFramesLoad,ref.nJoints,3))
			outPts_3d_monos = np.zeros((self.nFramesLoad,ref.nJoints,3))
			
			for i in range(self.nFramesLoad):
				ithFrameIndex = "{:06d}.jpg".format(selectedFrameIndices[i])
				LoadFrameAndData(path, ithFrameIndex)
				inpFrames[:,i,:,:] = frame
				outPts_2ds[i,:,:] = pts_2d
				outOutRegs[i,:,:] = outReg
				outPts_3d_monos[i,:,:] = pts_3d_mono
		

		return (inpFrames, outPts_2ds, outOutRegs, outPts_3d_monos)	