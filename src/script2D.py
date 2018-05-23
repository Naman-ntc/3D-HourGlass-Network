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



out = img[:,:,None,:,:].expand(1,3,32,256,256).cuda()
model3d = torch.load('inflatedModel.pth').cuda()










out = model3d.hg.convStart(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.bnStart(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.reluStart(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr1.bn(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr1.relu(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr1.conv(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr2.bn(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr2.relu(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr2.padLayer(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr2.conv(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr3.bn(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr3.relu(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1.cb.cbr3.conv(out)
print(out[0,:,0,:,:])
print("")