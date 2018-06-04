import sys
import h5py
import numpy as np
import cv2

def getData(tmpFile):
	data = h5py.File(tmpFile, 'r')
	d = {}
	for k, v in data.items():
		d[k] = np.asarray(data[k])
	data.close()
	return d

def ShowImg(data):
	img = data['img']
	img[0], img[2] = img[2].copy(), img[0].copy()
	if img.shape[0] == 3:
		img = img.transpose(1, 2, 0)
	cv2.imshow('img', img)

J = 16
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
				 [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
				 [6, 8], [8, 9]]

def show_3d(ax, points, c = (255, 0, 0)):
	points = points.reshape(J, 3)
	points[7] = points[8]
	x, y, z = np.zeros((3, J))
	for j in range(J):
		x[j] = points[j, 0] 
		y[j] = - points[j, 1] 
		z[j] = - points[j, 2] 
	ax.scatter(z, x, y, c = c)
	for e in edges:
		ax.plot(z[e], x[e], y[e], c =c)
		
		
def show_2d(img, points, c = 'g'):
	points = points.reshape(J, 2)
	for j in range(J):
		cv2.circle(img, (int(points[j, 0]), int(points[j, 1])), 3, c, -1)
	for e in edges:
		cv2.line(img, (int(points[e[0], 0]), int(points[e[0], 1])), 
									(int(points[e[1], 0]), int(points[e[1], 1])), c, 2)

def Show3d(datas,frame,gt_ind,dir_path,epoch):
	
	if(type(datas)!=list):
		datas = [datas]
	#print(len(datas))

	import matplotlib.pyplot as plt
	import mpl_toolkits.mplot3d
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot((111),projection='3d')
	ax.set_xlabel('z') 
	ax.set_ylabel('x') 
	ax.set_zlabel('y')
#  ax.set_xlim([-3000 , 3000])
#  ax.set_xlim([-3000 , 3000])
#  ax.set_zlim([-1500 , 1500])
#  
	
	
	
	for data in datas:
			joint = data['joint']
			oo = max(joint.max(), 2) / 8  
			xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
			if 'gt' in data:
				show_3d(ax, data['gt'], 'r')
			show_3d(ax, joint, 'b')
			max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
			Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
			Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
			Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
			for xb, yb, zb in zip(Xb, Yb, Zb):
				ax.plot([zb], [xb], [yb], 'w')
			if 'img' in data:
				img = data['img'].copy()
#        print(img.shape)
				stepX, stepY = 100. / img.shape[0], 100. / img.shape[1]
				X1 = np.arange(-3000, 3000, stepX)
				Y1 = np.arange(-1500, 1500, stepY)
#        img = img.transpose(1,2,0)
#        print('here')
				X1, Y1 = np.meshgrid(X1, Y1)
#        print('here after mesh grid')
				ax.plot_surface(X1, -1500,Y1, rstride=1, cstride=1, facecolors=img)
				
	fig.set_dpi(100)
	fig.savefig(dir_path+'epoch_'+str(epoch)+'_video'+str(video)+'_frame'+str(frame)+'show_3d.png')
	plt.show()
	fig.savefig(dir_path+str(frame)+'.jpg')
	plt.close()

if __name__ == '__main__':
	tmpFile = sys.argv[2]
	data = getData(tmpFile)
	eval(sys.argv[1] + '(data)') 
