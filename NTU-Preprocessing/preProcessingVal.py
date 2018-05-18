import numpy as np
import cv2
import os
import pickle

from functions import resizeAndPad, playVideoFromArray, playVideoFromAVI

from ntu_read_skeleton import read_skeleton

class box:
	def __init__(self):
		self.x1 = 1080
		self.x2 = 0
		self.y1 = 1920
		self.y2 = 0
		self.delta = 0

	def __str__(self):
		return "x1:" + str(self.x1) + " x2:" + str(self.x2) + " y1:" + str(self.y1) + " y2:" + str(self.y2) 

	def makeInt(self):
		self.x1 = int(self.x1)
		self.x2 = int(self.x2)
		self.y1 = int(self.y1)
		self.y2 = int(self.y2)

	def extend(self, percentage  = 40):
		fraction = percentage/100.0
		delX = self.x2 - self.x1
		delY = self.y2 - self.y1
		meanX = (self.x1 + self.x2)/2.0
		meanY = (self.y1 + self.y2)/2.0
		delta = max(delX, delY)
		delta += fraction*delta
		self.delta = int(delta)
		self.x1 = max(meanX - delta/2.0, 0)
		self.x2 = min(meanX + delta/2.0, 1080)
		self.y1 = max(meanY - delta/2.0, 0)
		self.y2 = min(meanY + delta/2.0, 1920)
		self.makeInt()
		return int(delta)

os.system("ls > output")
file = open("output", 'r')
videoList = ""
for chunk in file:
	videoList += chunk
file.close()
videoList = videoList.split()
videoList = [x for x in videoList if (x[-3:] == 'avi' and len(x) == 28)]

##### MADE A LIST OF THE FILES IN THE DIRECTORY

file = open("labels", 'w')
count = 0

total = len(videoList)


for video in videoList:
	try:
		label = video[video.find('A')+1:video.find('A') + 4]
		label = int(label) - 1
		if label >= 49:
			#print(label)
			continue
		cap = cv2.VideoCapture(video)
		skeletonFileName = "../../skeletonData/originalData/" + video[:-8] + ".skeleton"
		skeleton = read_skeleton(skeletonFileName)
		bboxes = []
		currentDict = {}
		currentDict["action"] = label
		d2d = []
		d3d = []
		for frameNo in range(skeleton['numFrame']):
			bbox = box()
			current2d = []
			current3d = []
			for jointNo in range(skeleton['frameInfo'][frameNo]['bodyInfo'][0]['numJoint']):
				bbox.x1 = min(bbox.x1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorY'])
				bbox.x2 = max(bbox.x2, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorY'])
				bbox.y1 = min(bbox.y1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorX'])
				bbox.y2 = max(bbox.y2, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorX'])
				current2d.append((bbox.x1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorX'],\
				 bbox.x1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorY']))
				current3d.append((bbox.x1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorX'],\
				 bbox.x1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorY'], \
				  bbox.x1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['z']))
				#print(bbox)

			sideLength = bbox.extend()
			bboxes.append(bbox)
			d2d.append(current2d)
			d3d.append(current3d)

		name = "val" + str(count) + ".avi"
		#outFile = cv2.VideoWriter(name,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (sideLength, sideLength))

		dirName = name[:-4]
		os.system("mkdir val/" + dirName)
		i = 0
		currentDict["2d"] = d2d
		currentDict["3d"] = d3d
		pickle.dump(currentDict, open("val/" + dirName + "/data.pkl", 'wb'))
		while (cap.isOpened()):
			
			ret, frame = cap.read()
			if ret:
				#print(type(bboxes[i].x1))
				crop = resizeAndPad(frame[bboxes[i].x1:bboxes[i].x2, bboxes[i].y1:bboxes[i].y2, :], (bboxes[i].delta, bboxes[i].delta))
				finalCrop = cv2.resize(crop, (224, 224))
				#currentVideo.append(crop)
				#outFile.write(crop)
				#print(crop.shape)
				cv2.imwrite("val/" + dirName + "/" + str(i) + ".jpg", finalCrop)
				#print("came here", dirName + "/" + str(i) + ".jpg")
				i+=1
			else:
				break

		#np.save(name,currentVideo)

		file.write(str(count)+","+str(label)+"\n")

		if count%10 == 0:
			print(str(count) + " of " + str(total) + " done - " + str(count*100.0/total) + " %")
		
		count = count + 1
	except:
		pass
	#playVideoFromAVI(name)
#playVideo("S001C001P001R001A001_rgb.avi")
