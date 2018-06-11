import ref
import torch
import numpy as np 
from utils.eval import *
from numpy.random import randn

def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
             
def Rnd(x):
  return max(-2 * x, min(2 * x, randn() * x))
  
def Flip(img):
  return img[:, :, ::-1].copy()  
  
def ShuffleLR(x):
  for e in ref.shuffleRef:
    x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
  return x


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

def csvFrame(output2D, output3D, meta):
  output2D = output2D.transpose(1,2).reshape(-1,output2D.shape[1],output2D.shape[3],output2D.shape[4])
  output3D = output2D.transpose(1,2).reshape(-1,output3D.shape[1])
  meta = meta.transpose(1,2).reshape(-1,meta.shape[1],meta.shape[3])
  meta = meta.numpy()
  p = np.zeros((output2D.shape[0], ref.nJoints, 3))
  p[:, :, :2] = getPreds(output2D).copy()
  
  hm = output2D.reshape(output2D.shape[0], output2D.shape[1], ref.outputRes, ref.outputRes)
  for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
      pX, pY = int(p[i, j, 0]), int(p[i, j, 1])
      scores = hm[i, j, pX, pY]
      if pX > 0 and pX < ref.outputRes - 1 and pY > 0 and pY < ref.outputRes - 1:
        diffY = hm[i, j, pX, pY + 1] - hm[i, j, pX, pY - 1]
        diffX = hm[i, j, pX + 1, pY] - hm[i, j, pX - 1, pY]
        p[i, j, 0] = p[i, j, 0] + 0.25 * (1 if diffX >=0 else -1)
        p[i, j, 1] = p[i, j, 1] + 0.25 * (1 if diffY >=0 else -1)
  p = p + 0.5
  
  p[:, :, 2] = (output3D.copy() + 1) / 2 * ref.outputRes
  h36mSumLen = 4296.99233013
  root = 6
  err = 0
  num3D = 0
  for i in range(p.shape[0]):
    s = meta[i].sum()
    if not (s > - ref.eps and s < ref.eps):
      num3D += 1
      lenPred = 0
      for e in ref.edges:
        lenPred += ((p[i, e[0]] - p[i, e[1]]) ** 2).sum() ** 0.5 
      pRoot = p[i, root].copy()
      for j in range(ref.nJoints):
        p[i, j] = (p[i, j] - pRoot) / lenPred * h36mSumLen + meta[i, root]
      p[i, 7] = (p[i, 6] + p[i, 8]) / 2
  return p


def writeCSV(filename,array):
  f=open(filename,'ab')
  np.savetxt(f,array,'%.4e',',')