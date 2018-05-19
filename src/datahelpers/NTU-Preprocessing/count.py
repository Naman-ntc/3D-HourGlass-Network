import os
import pickle

folders = os.listdir('.')


for folder in folders:
	if (folder[:5] != 'train'):
		continue
	n = len(pickle.load(open(folder + '/data.pkl', 'rb'))['joint_2d'])