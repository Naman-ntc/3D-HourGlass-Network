import os
import sys
import numpy as np
import cv2
import torch
from HourGlassNet3D import *
from Loss import *
from Softargmax import *
from DepthRegressor import *

argumentList = sys.argv[1:]

assert len(argumentList) > 0, "Give an Image Folder with -imageFolder Flag"
assert argumentList[0] == "-imageFolder", "Give an Image Folder with -imageFolder Flag"


argumentList[1] += "/"
all_frames = os.listdir(argumentList[1])

n_frames = len(all_frames)
frames_seq = np.zeros((1, 3, n_frames, 256, 256))
for idx, frame in enumerate(all_frames):
	frames_seq[0,:,idx,:,:] = cv2.imread(argumentList[1] + frame).transpose(2,0,1)

frames_seq = torch.from_numpy(frames_seq[:,:,:,:,:]).float() /256
frames_var = torch.autograd.Variable(frames_seq).float().cuda()

hg = HourglassNet3D()
hg = hg.cuda()
dr = DepthRegressor()
dr = dr.cuda()
print("Models Loaded in CUDA")


while True :
	
	#heatmaps,forDepth = hg(frames_var)
	#heatmaps = (heatmaps).float()
	#forDepth = (forDepth)
	print(len(hg(frames_var)))
	#print(hg(frames_var)[-1].shape)
	
	#heatmapsLen = len(heatmaps)
	#loss = 0
	"""
	SoftArgMaxLayer = SoftArgMax()
	for i in range(int(heatmapsLen)):
		temp = SoftArgMaxLayer(heatmaps[i])
		print(temp.size())

	zs = dr(forDepth)
	print(zs.size())
	"""
#hg = torch.load('inflatedModel.pth').cuda()
