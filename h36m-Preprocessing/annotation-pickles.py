import torch
import pickle
import numpy as np 
from h5py import File

import re


tags = ['action', 'bbox', 'camera', 'id', 'joint_2d', 'joint_3d_mono', 'subaction', 'subject', 'istrain']

annot = {}

f = File('annotSampleTest.h5', 'r')

for tag in tags:
	annot[tag] = np.asarray(f[tag]).copy()
f.close()


data = {}

length = annot['id'].shape[0]

train_vids = []
val_vids = []

train_cnt = []
val_cnt = []

for i in range(length):
	subject = annot['subject'][i]
	action = annot['action'][i]
	subaction = annot['subaction'][i]
	camera = annot['camera'][i]
	name = annot['id'][i]
	joint_2d = annot['joint_2d'][i]
	joint_3d_mono = annot['joint_3d_mono'][i]
	istrain = annot['istrain'][i]

	if subject in data:
		if action in data[subject]:
			if subaction in data[subject][action]:
				if camera in data[subject][action][subaction]:
					if name in data[subject][action][subaction][camera]:
						assert 1 != 1, "Name already seen :/ strange"
					else :
						data[subject][action][subaction][camera][name] = (joint_2d, joint_3d_mono, joint_3d_mono)
				else :
					data[subject][action][subaction][camera] = {name : (joint_2d, joint_3d_mono, joint_3d_mono)}
			else :
				data[subject][action][subaction] = {camera : {name : (joint_2d, joint_3d_mono, joint_3d_mono)}}
		else :
			data[subject][action] = {subaction : {camera : {name : (joint_2d, joint_3d_mono, joint_3d_mono)}}}
	else :
		data[subject] = {action : {subaction : {camera : {name : (joint_2d, joint_3d_mono, joint_3d_mono)}}}}	

	if istrain :
		vidFolder = "s_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}".format(subject,action,subaction,camera)
		if vidFolder in train_vids:
			pass
		else :
			train_vids.append(vidFolder)
	else :
		vidFolder = "s_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}".format(subject,action,subaction,camera)
		if vidFolder in val_vids:
			pass
		else :
			val_vids.append(vidFolder)


for i in train_vids:
	parsed = list(map(int, re.findall(r'\d+', i)))
	train_cnt.append(len(data[parsed[0]][parsed[1]][parsed[2]][parsed[3]]))
	pickle.dump(data[parsed[0]][parsed[1]][parsed[2]][parsed[3]], open('i/data.pkl','wb'))

for i in val_vids:
	parsed = list(map(int, re.findall(r'\d+', i)))
	val_cnt.append(len(data[parsed[0]][parsed[1]][parsed[2]][parsed[3]]))
	pickle.dump(data[parsed[0]][parsed[1]][parsed[2]][parsed[3]], open('i/data.pkl', 'wb'))



np.save(open('vid_train.npy','wb'), train_vids)
np.save(open('vid_val.npy','wb'), val_vids)
np.save(open('cnt_train.npy','wb'), train_cnt)
np.save(open('cnt_val.npy','wb'), val_cnt)

