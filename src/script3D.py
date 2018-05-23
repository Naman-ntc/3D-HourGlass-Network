import ref
import cv2
import torch
import numpy as np

from utils.img import Crop, DrawGaussian, Transform3D

c = np.ones(2) * ref.h36mImgSize / 2
s = ref.h36mImgSize * 1.0

img = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000111.jpg')

img = Crop(img, c, s, 0, ref.inputRes) / 256.
img.shape


img = torch.from_numpy(img).unsqueeze(0).cuda()



out = img.expand(32,3,256,256).cuda()
import pickle
from functools import partial

pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
pickle.load = partial(pickle.load, encoding="latin1")

model = torch.load('models/hgreg-3d.pth').cuda()








print("Script3D")

"""
out = model(out)
print(out[0][0,:,:,:])
print("")
print(out[1][0,:,:,:])
print("")
#print(out[2][0,:,:,:])
#print("")

"""
out = model.conv1_(out)
print(out[0,:,:,:])
print("")

out = model.bn1(out)
print(out[0,:,:,:])
print("")

out = model.relu(out)
print(out[0,:,:,:])
print("")

out = model.r1(out)
print(out[0,:,:,:])
print("")

out = model.maxpool(out)
print(out[0,:,:,:])
print("")

out = model.r4(out)
print(out[0,:,:,:])
print("")

out = model.r5(out)
print(out[0,:,:,:])
print("")

out = model.hourglass[0](out)
print(out[0,:,:,:])
print("")

out = model.Residual[0](out)
out = model.Residual[1](out)
print(out[0,:,:,:])
print("")

out = model.lin_[0](out)
print(out[0,:,:,:])
print("")

out = model.ll_[0](out)
print(out[0,:,:,:])
print("")