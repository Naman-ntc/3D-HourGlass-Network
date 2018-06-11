
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/project/multipose/src_3d/')
import torch.utils.data
#from datasets.cmu import CMU
from random import randint

import numpy as np
import matplotlib.pyplot as plt
import cv2
#from datasets.h36m import H36M
#from visualise_maps import visualise_map
js=['rank','rkne','rhip','lhip','lknee','lank','pelvi','spin','neck','head','rwr','relb','rshou','lshou','lelb','lwri']
ls=['rloleg','rupleg','rhip','lhip','lupleg','lloleg','rloarm','ruparm','rshou','lshou','luparm','lloarm','spine','head']






def test_heatmaps(heatmaps,img,i):
    heatmaps=heatmaps.numpy()
    #heatmaps=np.squeeze(heatmaps)
    heatmaps=heatmaps[:,:64,:]
    heatmaps=heatmaps.transpose(1,2,0)
    print('heatmap inside shape is',heatmaps.shape)
##    print('----------------here')
##    print(heatmaps.shape)
    img=img.numpy()
    #img=np.squeeze(img)
    img=img.transpose(1,2,0)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    print('heatmaps',heatmaps.shape)
    heatmaps = cv2.resize(heatmaps,(0,0), fx=4,fy=4)
#    print('heatmapsafter',heatmaps.shape)
    for j in range(0, 16):
        heatmap = heatmaps[:,:,j]
        heatmap = heatmap.reshape((256,256,1))
        heatmapimg = np.array(heatmap * 255, dtype = np.uint8)
        heatmap = cv2.applyColorMap(heatmapimg, cv2.COLORMAP_JET)
        heatmap = heatmap/255
        plt.imshow(img)
        plt.imshow(heatmap, alpha=0.5)
        plt.show()
        #plt.savefig('hmtestpadh36'+str(i)+js[j]+'.png')
        


def test_vecmaps(vecmaps,img,i):
    vecmaps = vecmaps.numpy()
    vecmaps=np.squeeze(vecmaps)
    vecmaps=vecmaps[:,:64,:]
    vecmaps=vecmaps.transpose(1,2,0)
    print('vecmapisss',vecmaps.shape)
    img=img.numpy()
    img=np.squeeze(img)
    img=img.transpose(1,2,0)
    img=img[:256,:,:]
    img=img.astype(np.uint8)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vecmaps = cv2.resize(vecmaps, (0,0), fx=4,fy=4)
    for k in range(0, 28, 2):
        vec = np.abs(vecmaps[:,:,k])
        vec += np.abs(vecmaps[:,:,k + 1])
        vec[vec > 1] = 1
        vec = np.reshape(vec,(256,256,1))
        vecimg = np.array(vec * 255, dtype = np.uint8)
        vec = cv2.applyColorMap(vecimg, cv2.COLORMAP_JET)
        vec=vec/255;
        print('vecmap shapeeeee',vec.shape)
#        vec = vec.reshape((256,256))

        plt.imshow(img)
        plt.imshow(vec,alpha=0.5)
        plt.savefig('vmh36'+str(i)+ls[k//2]+'.png')
        plt.close()


def func1(k=None):
    if not k:
        k=randint(0, 20)
    print('image is',k)
    for i, (img, heatmap,vecmap,depthmap,kpt_3d) in enumerate(train_loader):
        if i==k:
#            test_heatmaps(heatmap,img,i)
#            test_vecmaps(vecmap,img,i)
#            edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [6, 8], [8, 9]]
#            ppl=kpt_3d.shape[0]
#            for i in range(ppl):
#                for edge in edges:
#                    cv2.line(img,(int(kpt_3d[i][edge[0]][0]),int(kpt_3d[i][edge[0]][1])),(int(kpt_3d[i][edge[1]][0]),int(kpt_3d[i][edge[1]][1])),(0,255,0))
#            cv2.imwrite('outside3dfinal.png',img)
       
            return img,heatmap,vecmap,depthmap,kpt_3d

