# -*- coding: utf-8 -*-


import sys
from  utils.pyTools import Show3d
import ref
import numpy as np



def visualise3d(pred,gt,epoch,iterindex,frame):
    
    pred_root_rel = pred[:,:3] - pred[ref.root,:3]
#    gt_root_rel   = gt[:,:3]   - gt[ref.root, :3]
    
    
    gt_length=0
    len_pred=0
    tot_cnt=0
    for e in ref.edges:
        if pred_root_rel[e[0]][0]!=0 and pred_root_rel[e[0]][1]!=0 and pred_root_rel[e[1]][1]!=0 and pred_root_rel[e[1]][1]!=0:
            len_pred += ((pred_root_rel[e[0]][:2] - pred_root_rel[e[1]][:2]) ** 2).sum() ** 0.5
            gt_length += ((gt[e[0]][:2] - gt[e[1]][:2]) ** 2).sum() ** 0.5
        else:
            tot_cnt=tot_cnt+1

    gt_root   =  gt[ref.root]
    for j in range(ref.nJoints):
        pred_root_rel[j] = ((pred_root_rel[j]) / len_pred) * gt_length + gt_root
    data={}
    data['joint']=pred_root_rel
    data['gt']=gt
    Show3d(data,'./Plots/', epoch, iterindex, frame)