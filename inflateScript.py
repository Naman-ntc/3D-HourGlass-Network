import pickle
from functools import partial
from models.hg_3d import *
from HourGlassNet3D import *
from inflate import *
import torch


model3d = torch.load('model3d.pth')

pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
pickle.load = partial(pickle.load, encoding="latin1")
model = torch.load('hgreg-3d.pth')

inflateHourglassNet(model3d, model)