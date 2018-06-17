import ref
import cv2
import torch
import numpy as np
torch.set_printoptions(precision=5)

import pickle
from functools import partial
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
pickle.load = partial(pickle.load, encoding="latin1")
from utils.img import Crop, DrawGaussian, Transform3D

c = np.ones(2) * ref.h36mImgSize / 2
s = ref.h36mImgSize * 1.0


img1 = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000111.jpg')
img1 = Crop(img1, c, s, 0, ref.inputRes) / 256.

img2 = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000112.jpg')
img2 = Crop(img2, c, s, 0, ref.inputRes) / 256.

img3 = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000113.jpg')
img3 = Crop(img3, c, s, 0, ref.inputRes) / 256.

img1 = torch.from_numpy(img1).cuda().float()
img2 = torch.from_numpy(img2).cuda().float()
img3 = torch.from_numpy(img3).cuda().float()

img1.unsqueeze_(0)
img2.unsqueeze_(0)
img3.unsqueeze_(0)


img = torch.cat((img1,img2,img3),0).contiguous()


x = torch.autograd.Variable(img)
model = torch.load('models/xingy.pth').cuda().float()



x = model(x)
print(x[2][:,:].t())

"""





x = model.conv1_(x)
x = model.bn1(x)
x = model.relu(x)
x = model.r1(x)
print("Res1 Done")

x = model.maxpool(x)
x = model.r4(x)
print("Res2 Done")

x = model.r5(x)
print("Res3 Done")

out = []

for i in range(model.nStack):
	hg = model.hourglass[i](x)
	print("Hourglass Done", i)

	ll = hg
	for j in range(model.nModules):
		ll = model.Residual[i * model.nModules + j](ll)
	print("Res j Done", i)

	ll = model.lin_[i](ll)
	tmpOut = model.tmpOut[i](ll)
	out.append(tmpOut)
	print("out append Done")
	ll_ = model.ll_[i](ll)
	print("ll_ Done")
	tmpOut_ = model.tmpOut_[i](tmpOut)
	print("tmpout_ Done")
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
