import ref
import cv2
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms.functional as F 


class posetrack(data.Dataset):
	"""docstring for posetrack"""
	def __init__(self, split, opts, returnMeta = True, augmentation = False):
		super(posetrack, self).__init__()
		self.split = split
		self.nFramesLoad = opts.nFramesLoad
		self.loadConsecutive = opts.loadConsecutive
		self.augmentation = augmentation
		self.annotations = pickle.load(open(ref.posetrackDataDir + '/annotations.pkl','rb'))
		
		self.nVideos = len(self.annotations)

		print("Loaded %d %s videos for posetrack data" %(self.nVideos, split))

	def __getitem__(self, index):
		if split=='train':
			index = np.random.randint(self.nVideos)
		whichFrames,frameWiseData = self.annotations[index]
		
		numFrames = len(whichFrames)

		if self.loadConsecutive:
			
			startpt = random.randint(1, numFrames - (self.nFramesLoad - 1))
			inpFrames = np.zeros((3,self.nFramesLoad,256,256))
			outPts_2ds = np.zeros((ref.nJoints,self.nFramesLoad,2))
			outOutRegs = np.ones((ref.nJoints,self.nFramesLoad,1))
			outPts_3d_monos = np.zeros((ref.nJoints,self.nFramesLoad,3))
			outOutMaps = np.zeros((ref.nJoints, self.nFramesLoad, ref.outputRes, ref.outputRes))

			nPersons = len(frameWiseData[whichFrames[0]])
			personIndex = random.randint(0,nPersons-1)

			if self.augmentation:
				angle = random.randint(0,10)
				scale = 1 + (random.random()-0.5)*0.4
				flipOrNot  = random.random() > 0
			else:
				angle = 0
				scale = 1
				flipOrNot = 0	

			for i in range(self.nFramesLoad):
				tempFrame = cv2.imread(ref.posetrackDataDir + whichFrames[startpt + i])
				frameData = frameWiseData[whichFrames[startpt + i]]
				bbox, joints = frameData[personIndex]
				
				cropedAndResized = F.resized_crop(tempFrame, bbox.mean[0] + bbox.delta, bbox.mean[1] - bbox.delta, bbox.delta, bbox.delta, 224)
				
				padded = F.pad(cropedAndResized, 256 - 224)
				
				rotated = F.affine(padded,angle,scale)

				if flipOrNot:
					flipped = hflip(rotated)
					thisFrame = flipped
				else:
					thisFrame = rotated

				inpFrames[:,i,:,:] = thisFrame
				outPts_2ds[:,i,:] = joints
				for j in range(ref.nJoints):
					if (joints[j,:] == -1).all():
						continue
					else:
						outOutMaps[j,i,:,:] = DrawGaussian(outOutMaps[j,i,:,:], pt, ref.hmGauss)