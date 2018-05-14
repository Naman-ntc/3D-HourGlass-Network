import os
import sys
import numpy as np
import cv2
import torch
from HourGlassNet3D import *
from Loss import *
from Softargmax import *

argumentList = sys.argv[1:]

assert len(argumentList) > 0, "Give an Image Folder with -imageFolder Flag"
assert argumentList[0] == "-imageFolder", "Give an Image Folder with -imageFolder Flag"


argumentList[1] += "/"
all_frames = os.listdir(argumentList[1])


n_frames = len(all_frames)
frames_seq = np.zeros((1, 3, n_frames, 256, 256))
for idx, frame in enumerate(all_frames):
	frames_seq[0,:,idx,:,:] = cv2.imread(argumentList[1] + frame).transpose(2,0,1)


frames_seq = torch.from_numpy(frames_seq[:,:,:16,:,:]).float() /256
print("Frames Developed\n")

frames_seq = frames_seq.cuda()
print("Frames in CUDA\n")


hg = torch.load('inflatedModel.pth').cuda()
print("Model Loaded in CUDA")


while True :
	heatmaps = hg(frames_seq)
	heatmapsLen = len(heatmaps)
	loss = 0

	SoftArgMaxLayer = SoftArgMax()
	for i in range(heatmapsLen):
		temp = SoftArgMaxLayer(heatmaps[i])
		print(temp)
        #temp1 = (Variable(torch.randn(temp.size())).cuda() + 1)*256
		#loss  += JointSquaredError(temp, temp1)

