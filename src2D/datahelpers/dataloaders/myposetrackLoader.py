import ref
import cv2
import torch
import pickle
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms.functional as F 
from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform
import matplotlib.pyplot as plt

class posetrack(data.Dataset):
	"""docstring for posetrack"""
	def __init__(self, split, returnMeta = True, augmentation = False):
		super(posetrack, self).__init__()
		self.split = split
		self.nFramesLoad = 16#opts.nFramesLoad
		self.loadConsecutive = 1#opts.loadConsecutive
		self.augmentation = augmentation
		self.annotations = pickle.load(open(ref.posetrackDataDir + '/' + split + '.pkl','rb'))
		
		self.nVideos = len(self.annotations)

		print("Loaded %d %s videos for posetrack data" %(self.nVideos, split))

	def __getitem__(self, index):
		# if self.split=='train':
		# 	index = np.random.randint(self.nVideos)
		whichFrames,frameWiseData = self.annotations[index]
		
		numFrames = len(whichFrames)

		if self.loadConsecutive:
			
			startpt = random.randint(1, numFrames - (self.nFramesLoad))
			inpFrames = np.zeros((3,self.nFramesLoad,256,256))
			outPts_2ds = np.zeros((ref.nJoints,self.nFramesLoad,2))
			outOutRegs = np.ones((ref.nJoints,self.nFramesLoad,1))
			outPts_3d_monos = np.zeros((ref.nJoints,self.nFramesLoad,3))
			outOutMaps = np.zeros((ref.nJoints, self.nFramesLoad, ref.outputRes, ref.outputRes))

			result = set(frameWiseData[whichFrames[startpt]].keys())
			for i in range(self.nFramesLoad):
				try:
					result.intersection_update(frameWiseData[whichFrames[startpt+i]].keys())
				except:
					print(whichFrames[startpt+i])
					assert 1!=1
			if (len(result) == 0):
				return self.__getitem__(index+1)
			personIndex = random.choice(list(result))

			if self.augmentation:
				angle = random.randint(0,10)
				scale = 1 + (random.random()-0.5)*0.4
				flipOrNot  = random.random() > 0
			else:
				angle = 0
				scale = 1
				flipOrNot = 0	

			for i in range(self.nFramesLoad):
				tempFrame = cv2.imread(ref.posetrackDataDir + '/' + whichFrames[startpt + i])
				plt.imshow(tempFrame)
				plt.show()
				frameData = frameWiseData[whichFrames[startpt + i]]
				bboxmean,bboxdelta, joints = frameData[personIndex]
				print(bboxmean,bboxdelta)				
				tempFrame = F.to_pil_image(tempFrame)
				cropedAndResized = F.resized_crop(tempFrame, bboxmean[0] + bboxdelta, bboxmean[1] - bboxdelta, bboxdelta, bboxdelta, (224,224))
				for k in range(joints.shape[0]):
					if (joints[k][0]<0):
						pass
					else :
						joints[k][0] += 16
						joints[k][1] += 16
				padded = F.pad(cropedAndResized, 16)
				
				rotated = F.affine(padded,angle,(0,0),scale,0)

				if flipOrNot:
					flipped = hflip(rotated)
					thisFrame = flipped
				else:
					thisFrame = rotated

					
				inpFrames[:,i,:,:] = F.to_tensor(thisFrame)
				outPts_2ds[:,i,:] = joints
				for j in range(ref.nJoints):
					if (joints[j,:] == -1).all():
						continue
					else:
						joints[j,0] += bboxdelta
						joints[j,1] += bboxdelta
						outOutMaps[j,i,:,:] = DrawGaussian(outOutMaps[j,i,:,:], joints[j,:], ref.hmGauss)
		return (inpFrames, outOutMaps, outPts_2ds, outOutRegs, outPts_3d_monos)			

	def __len__(self):
		return self.nVideos