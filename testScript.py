import os
import sys
import torch
import numpy as np
import cv2
from HourGlassNet3D import *
import time

argumentList = sys.argv[1:]

assert argumentList[0] == "-imageFolder", "Give an Image Folder with -imageFolder Flag"

all_frames = os.listdir(argumentList[1])

n_frames = len(all_frames)
frames_seq = np.zeros((1, 3, n_frames, 256, 256))
for idx, frame in enumerate(all_frames):
	frames_seq[0,:,idx,:,:] = cv2.imread(argumentList[1] + frame).transpose(2,0,1)

frames_seq = torch.from_numpy(frames_seq).float() /256
frames_var = torch.autograd.Variable(frames_seq).float().cuda()

hg = HourglassNet3D(64,2,2,4)
hg = hg.cuda()

start_time = time.time()

while(True):
	print(hg(frames_var)[-1].shape)
	print("--- %s seconds ---" % (time.time() - start_time))
	start_time = time.time()


"""
Can Improve it!!
"""
