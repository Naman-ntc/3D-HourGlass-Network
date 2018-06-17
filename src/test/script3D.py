import ref
import cv2
import torch
import numpy as np

import pickle
from functools import partial
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
pickle.load = partial(pickle.load, encoding="latin1")
from utils.img import Crop, DrawGaussian, Transform3D

c = np.ones(2) * ref.h36mImgSize / 2
s = ref.h36mImgSize * 1.0


img1 = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000111.jpg')
img1 = Crop(img, c, s, 0, ref.inputRes) / 256.
img1 = img1.unsqeeze(0)
img2 = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000112.jpg')
img2 = Crop(img, c, s, 0, ref.inputRes) / 256.
img2 = img2.unsqeeze(0)
img3 = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000113.jpg')
img3 = Crop(img, c, s, 0, ref.inputRes) / 256.
img3 = img3.unsqeeze(0)

img = torch.cat((img1,img2,img3),0).contiguous()


out = img
model = torch.load('models/xingy.pth').cuda()



out = model(model)
out[2][0,:]

"""
x = model.conv1_(x)
x = model.bn1(x)
x = model.relu(x)
x = model.r1(x)
x = model.maxpool(x)
x = model.r4(x)
x = model.r5(x)

out = []

for i in range(model.nStack):
	hg = model.hourglass[i](x)
	ll = hg
	for j in range(model.nModules):
		ll = model.Residual[i * model.nModules + j](ll)
	ll = model.lin_[i](ll)
	tmpOut = model.tmpOut[i](ll)
	out.append(tmpOut)
	
	ll_ = model.ll_[i](ll)
	tmpOut_ = model.tmpOut_[i](tmpOut)
	x = x + ll_ + tmpOut_

print(x[0,:,:,:])
print("")

for i in range(4):
	for j in range(model.nRegModules):
		x = model.reg_[i * model.nRegModules + j](x)
	x = model.maxpool(x)

print(x[0,:,:,:])
print("")	

x = x.view(x.size(0), -1)
reg = model.reg(x)
out.append(reg)


"""
