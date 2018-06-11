import os
import pickle

folders = os.listdir('.')

vids = []
cnts = []

for folder in folders:
	if (folder[:5] != 'train'):
		continue
	data = (pickle.load(open(folder + '/data.pkl', 'rb'))
	n = len(data['2d'])
	vids.append(folder)
	cnts.append(n)

	2d = np.asarray(data['2d'])
	3d = np.asarray(data['3d'])

	data['2d'] = 2d
	data['3d'] = 3d




