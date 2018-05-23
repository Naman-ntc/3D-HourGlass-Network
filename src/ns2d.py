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



out3d = img[:,:,None,:,:].expand(1,3,32,256,256).cuda()
model3d = torch.load('inflatedModel.pth').cuda()


out2d = img.expand(32,3,256,256).cuda()
import pickle
from functools import partial
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
pickle.load = partial(pickle.load, encoding="latin1")
model = torch.load('models/hgreg-3d.pth').cuda()

out2d = model.conv1_(out2d)
out2d = model.bn1(out2d)
out2d = model.relu(out2d)
out2d = model.r1(out2d)
out2d = model.maxpool(out2d)
out2d = model.r4(out2d)
out2d = model.r5(out2d)

out2d = model.hourglass[0](out2d)
out3d = out2d


out3d = model3d.hg.Residual[0](out3d)
print(out3d[0,:,0,:,:])
print("Residual model")

out3d = model3d.hg.lin1[0](out3d)
print(out3d[0,:,0,:,:])
print("lin1 model")

out3d = model3d.hg.lin2[0](out3d)
print(out3d[0,:,0,:,:])
print("lin2 model")