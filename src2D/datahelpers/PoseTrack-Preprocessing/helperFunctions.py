import numpy as np
mapping = {0:0,1:1,2:2,3:3,4:4,5:5,6:10,7:11,8:12,9:13,10:14,11:15,12:8,14:9}


def stabilize(crazy):
	numVisible = crazy.shape[0]
	joints = -1*np.ones((16,2))
	cluttered = 13
	for i in range(numVisible):
		ithJoint = crazy[i]
		id = ithJoint[0][0][0]
		x = ithJoint[1][0][0]
		y = ithJoint[2][0][0]
		if (id == 13):
			continue
		cluttered -= 1
		joints[mapping[id]][0] = x
		joints[mapping[id]][1] = y
	if ((joints[2,0] == -1)  or (joints[3,0] == -1)):
		pass
	else :	
		joints[6,:] = (joints[2,:] + joints[3,:])/2
	
	if ((joints[6,0] == -1)  or (joints[8,0] == -1)):
		pass
	else :	
		joints[7,:] = (joints[8,:] + joints[6,:])/2

	return (cluttered, joints)	
