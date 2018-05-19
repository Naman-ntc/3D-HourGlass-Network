import torch
import cv2
import numpy as np
import random
from utils import *
import pickle


def LoadFrameAndData(path, frameName):
	print(path)
	print(frameName)
	edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [6, 8], [8, 9]]
	nJoints = 16
	h36mImgSize = 224
	outputRes = 64
	inputRes = 256
	nFramesLoad = 16
	frame = cv2.imread(path+frameName)
	pts_2d, pts_3d, pts_3d_mono = pickle.load(open(path + "data.pkl", 'rb'))[int(frameName[-10:-4])]
	p = pts_2d
	pts_3d = pts_3d
	pts_3d_mono = pts_3d_mono
	c = np.ones(2) * h36mImgSize / 2
	s = h36mImgSize * 1.0
	pts_3d = pts_3d - pts_3d[7]
	s2d, s3d = 0, 0
	for e in edges:
		s2d += ((pts_2d[e[0]] - pts_2d[e[1]]) ** 2).sum() ** 0.5
		s3d += ((pts_3d[e[0], :2] - pts_3d[e[1], :2]) ** 2).sum() ** 0.5
	scale = s2d / s3d
	for j in range(nJoints):
		pts_3d[j, 0] = pts_3d[j, 0] * scale + pts_2d[7, 0]
		pts_3d[j, 1] = pts_3d[j, 1] * scale + pts_2d[7, 1]
		pts_3d[j, 2] = pts_3d[j, 2] * scale + h36mImgSize / 2
	pts_3d[7,:] = (pts_3d[12,:] + pts_3d[13,:]) / 2
	frame = Crop(frame, c, s, 0, inputRes) / 256.
	outReg = np.zeros((nJoints, 3))
	for i in range(nJoints):
		pt = Transform3D(pts_3d[i], c, s, 0, outputRes)
		outReg[i, 2] = pt[2] / outputRes * 2 - 1
	frame = torch.from_numpy(frame)
	return frame, pts_2d, outReg, pts_3d_mono



def getitem(index, loadConsecutive):
	nJoints = 16
	h36mImgSize = 224
	outputRes = 64
	inputRes = 256
	nFramesLoad = 16
	vidFolders = np.load('vid_train.npy')
	countFrames = np.load('cnt_train.npy')
	if ('train' == 'train'):
		index = np.random.randint(100)
	vidFolder = vidFolders[index]
	path = '.' + "/" + vidFolder + "/"
	CountFramesInVid = countFrames[index]
	if loadConsecutive:
		startPt = random.randint(1, CountFramesInVid - nFramesLoad + 2)
		inpFrames = np.zeros((3,nFramesLoad,256,256))		
		outPts_2ds = np.zeros((nFramesLoad,nJoints,2))
		outOutRegs = np.zeros((nFramesLoad,nJoints,3))
		outPts_3d_monos = np.zeros((nFramesLoad,nJoints,3))
		for i in range(nFramesLoad):
			frameIndex = "{:06d}.jpg".format(5*startPt-4)
			frame,pts_2d,outReg,pts_3d_mono = LoadFrameAndData(path, vidFolder + "_" + frameIndex)
			inpFrames[:,i,:,:] = frame
			outPts_2ds[i,:,:] = pts_2d
			outOutRegs[i,:,:] = outReg
			outPts_3d_monos[i,:,:] = pts_3d_mono
	else :
		frameIndices = np.random.permutation(CountFramesInVid)
		selectedFrameIndices = frameIndices[:nFramesLoad]
		inpFrames = np.zeros((3,nFramesLoad,256,256))		
		outPts_2ds = np.zeros((nFramesLoad,nJoints,2))
		outOutRegs = np.zeros((nFramesLoad,nJoints,3))
		outPts_3d_monos = np.zeros((nFramesLoad,nJoints,3))
		for i in range(nFramesLoad):
			ithFrameIndex = "{:06d}.jpg".format(5*selectedFrameIndices[i] - 4)
			frame,pts_2d,outReg,pts_3d_mono = LoadFrameAndData(path, path[2:-1] + '_' +  ithFrameIndex)
			inpFrames[:,i,:,:] = frame
			outPts_2ds[i,:,:] = pts_2d
			outOutRegs[i,:,:] = outReg
			outPts_3d_monos[i,:,:] = pts_3d_mono
	return (inpFrames, outPts_2ds, outOutRegs, outPts_3d_monos)	