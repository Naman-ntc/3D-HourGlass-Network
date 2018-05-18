import torch
import torch.nn as nn

from HourGlassNet3D import *
from HourGlass3D import *
from Layers3D import *
from DepthRegressor3D import *

from SoftArgMax import *
from Losses import *

from dataLoading.FusedDataLoader import *

""" ##############################################################################################
	###############################################################################################
""" ##############################################################################################

hg = HourglassNet3D().cuda()
dr = DepthRegressor3D().cuda()

opt = opts().parse()
now = datetime.datetime.now()
logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))


train_loader = torch.utils.data.DataLoader(
		FusionDataset(_,'train',32),
		batch_size = opt.trainBatch,
		shuffle = True if opt.DEBUG == 0 else False,
		num_workers = int(ref.nThreads)
	)

optimizer = torch.optim.RMSprop(
	model.parameters(), opt.LR, 
	alpha = ref.alpha, 
	eps = ref.epsilon, 
	weight_decay = ref.weightDecay, 
	momentum = ref.momentum
)

