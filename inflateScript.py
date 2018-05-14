import pickle
from functools import partial
from models.hg_3d import *
from HourGlassNet3D import *
from HourGlass3D import *
from Layers3D import *
from Inflate import *
import torch


model3d = HourglassNet3D(256,2,2,4,16)
torch.save(model3d, oepn('models/model3d.pth', 'wb'))

pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
pickle.load = partial(pickle.load, encoding="latin1")
model = torch.load('models/hgreg-3d.pth') #, map_location=lambda storage, loc: storage)

inflateHourglassNet(model3d, model)

torch.save(model3d,open('inflatedModel.pth','wb'))

