import os
import cv2
import numpy as np

from utils import *


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
		cap = cv2.VideoCapture(video)
		bbox = box()
		skeletonFileName = "skeletons/" + video[:-8] + ".skeleton"
		skeleton = read_skeleton(skeletonFileName)
		
		jointsPixels = np.zeros(skeleton['numFrame'],16,2)  ## stores x,y
		
		for frameNo in range(skeleton['numFrame']):
			for jointNo in range(skeleton['frameInfo'][frameNo]['bodyInfo'][0]['numJoint']):
				bbox.x1 = min(bbox.x1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorY'])
				bbox.x2 = max(bbox.x2, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorY'])
				bbox.y1 = min(bbox.y1, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorX'])
				bbox.y2 = max(bbox.y2, skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorX'])
				jointsPixels[i,j,0] = skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorX']
				jointsPixels[i,j,1] = skeleton['frameInfo'][frameNo]['bodyInfo'][0]['jointInfo'][jointNo]['colorY']
		
		
		sideLength = bbox.extend()

		name = "val_" + str(count) 
		
		while (cap.isOpened()):
			ret, frame = cap.read()
			if ret:
				crop = resizeAndPad(frame[bbox.x1:bbox.x2, bbox.y1:bbox.y2, :], (sideLength, sideLength))
				outFile.write(crop)
			else:
				break

		outFile.release()

		#np.save(name,currentVideo)

		file.write(str(count)+","+str(label)+"\n")

		if count%10 == 0:
			print(str(count) + " of " + str(total) + " done - " + str(count*100.0/total) + " %")
		
		count = count + 1
	except:
		pass