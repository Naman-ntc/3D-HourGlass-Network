import os
import torch
import numpy as np 
import scipy.io as sio
import numpy as np
from helperFunctions import *
import pickle

class Bbox:
	def __init__(self):
		self.mean = None
		self.delta = None

def makeBoundingBox(joints, slack = 0.2):
	#slack is the percent of the extra area that we keep in the crop, ie, right now the total width will be 1.2 times the maximum distance between the
	#joints that are farthest apart along the x axis
	minX = 20000
	maxX = 0
	minY = 20000
	maxY = 0
	for joint in joints:
		if joint[0]<=0 or joint[1]<=0:
			continue
		minX = min(minX, joint[0])
		maxX = max(maxX, joint[0])
		minY = min(minY, joint[1])
		maxY = max(maxY, joint[1])
	box = Bbox()
	box.mean = ((minX + maxX)/2.0, (minY + maxY)/2.0)
	box.delta = (0.5*(maxX-minX+5)*(1+slack), 0.5*(maxY-minY+5)*(1+slack))
	print(box.mean, box.delta)
	return box



split = 'train'
os.system("ls annotations/%s > output" %(split))
file = open("output", 'r')
matList = ""
for chunk in file:
	matList += chunk
file.close()
matList = matList.split()
matList = [x for x in matList if (x[-3:] == 'mat')]

##### MADE A LIST OF THE FILES IN THE DIRECTORY


finalData = []

for mat in matList:
	file = sio.loadmat("annotations/" + split + "/" + mat)
	file = file['annolist']
	size = file.shape[1]
	whichFrames = []
	frameWiseData = {}
	for i in range(size):
		thisFrame = file[0,i]
		isAnnotated = thisFrame[3][0][0]
		thisframeData = {}
		if isAnnotated:
			imageName = thisFrame[0][0][0][0][0]
			whichFrames.append(imageName)
			
			if (len(thisFrame[1]) == 0):
				whichFrames.remove(imageName)
				continue
			try :
				countPeople = thisFrame[1][0].shape[0]
			except:
				print(thisFrame[1])
			
			personWiseData = {}
			for j in range(countPeople):
				person = thisFrame[1][0][j]
				personID = person[6][0][0]
				if person[7].shape == (0,0):
					continue
				crazyJoints = person[7][0][0][0][0]
				cluttered,joints = stabilize(crazyJoints)
				"""
				COMPLETE HERE!!!
				given set of n Joints (here probably 13 but better make invariant to it)
				give me a bounding box, no cropping required (as memory pains ho skta thoda)
				that is we'll runtime on crop, give boundingBox + updated pixels

				also better to give a bounding box in the following manney :
					1) give me the center root relative krke (average of joints[2,:] and joints[3,:])
					2) number of pixels to crop from centre on each sidex (delta/2 on each side give me the delta)
				also give the changed pixel coors as a numpy array of shape 
							Nx2 (N is number of joints)
				return them as a tuple of bbox,newJoints
				"""

				bbox = makeBoundingBox(joints) ##Bbox is a class, with bbox.mean a tuple of x, y and bbox.delta
				personWiseData[personID] = ([bbox.mean,bbox.delta,joints])
			frameWiseData[imageName] = personWiseData
	finalData.append((whichFrames,frameWiseData))


pickle.dump(finalData, open(split + '.pkl','wb'))
