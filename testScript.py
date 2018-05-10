import os
import torch
import numpy as np
import cv2
from HourGlassNet3D import *

all_frames = os.listdir('../train_frames/train927')
n_frames = len(all_frames)
frames_seq = np.zeros((1, 3, n_frames, 256, 256))
for idx, frame in enumerate(all_frames):
	frames_seq[0,:,idx,:,:] = cv2.imread('../train_frames/train927/'+ frame).transpose(2,0,1)

frames_seq = torch.from_numpy(frames_seq).float() /256
frames_var = torch.autograd.Variable(frames_seq).float().cuda()

hg = HourglassNet3D(64,2,1,4)
hg = hg.cuda()
while(True):
	print(hg(frames_var)[-1].shape)


"""
Can Improve it!!
"""