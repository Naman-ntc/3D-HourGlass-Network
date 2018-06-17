import ref
import cv2
import torch
import numpy as np

from utils.img import Crop, DrawGaussian, Transform3D

c = np.ones(2) * ref.h36mImgSize / 2
s = ref.h36mImgSize * 1.0

img1 = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000111.jpg')
img1 = Crop(img, c, s, 0, ref.inputRes) / 256.

img2 = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000112.jpg')
img2 = Crop(img, c, s, 0, ref.inputRes) / 256.

img3 = cv2.imread('../data/h36m/s_01_act_02_subact_01_ca_03/s_01_act_02_subact_01_ca_03_000113.jpg')
img3 = Crop(img, c, s, 0, ref.inputRes) / 256.


img = torch.cat((img1,img2,img3),2).contiguous()

out = img
model3d = torch.load('inflatedModel.pth').cuda()



print("Script2D")

out = model3d(out)[2]
print(out[0,:,0,:])
print("")


"""
out = model3d.hg.convStart(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.bnStart(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.reluStart(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res1(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.mp(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res2(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.res3(out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.hourglass[0](out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.Residual[0](out)
print(out[0,:,0,:,:])
print("")

out = model3d.hg.lin1[0](out)
print(out[0,:,0,:,:])
print("")
print("Script2D")
"""
