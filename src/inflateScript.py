import sys
#sys.path.insert(0,'..')
import pickle
from functools import partial
#from twod.hg_3d import *
from model.Pose3D import *
from model.DepthRegressor3D import *
from model.HourGlassNet3D import *
from model.HourGlass3D import *
from model.Layers3D import *
import inflation.Inflate as Inflate
import torch
import ref
from opts import opts



def inflate(opt = None):
	if opt is not None:
		model3d = Pose3D(opt.nChannels, opt.nStack, opt.nModules, opt.numReductions, opt.nRegModules, opt.nRegFrames, ref.nJoints, ref.temporal)
		Inflate.nChannels = opt.nChannels
		Inflate.nStack = opt.nStack
		Inflate.nModules = opt.nModules
		Inflate.nRegFrames = opt.nRegFrames
		Inflate.nJoints = ref.nJoints
		Inflate.scheme = opt.scheme
		Inflate.mult = opt.mult
	else :
		opt = opts().parse()
		Inflate.nChannels = opt.nChannels
		Inflate.nStack = opt.nStack
		Inflate.nModules = opt.nModules
		Inflate.nRegFrames = opt.nRegFrames
		Inflate.nJoints = ref.nJoints
		Inflate.scheme = opt.scheme
		model3d = Pose3D(opt.nChannels, opt.nStack, opt.nModules, opt.numReductions, opt.nRegModules, opt.nRegFrames, ref.nJoints, ref.temporal)
	pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
	pickle.load = partial(pickle.load, encoding="latin1")
	if opt is not None:
		model = torch.load(opt.Model2D)
	else:
		model = torch.load('models/xingy.pth') #, map_location=lambda storage, loc: storage)

	Inflate.inflatePose3D(model3d, model)

	torch.save(model3d,open('inflatedModel.pth','wb'))

	return model3d


#inflate()
