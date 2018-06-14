import argparse
import os
import ref

class opts():
	def __init__(self):
		self.parser = argparse.ArgumentParser()

	def init(self):
		self.parser.add_argument('-expID', default = 'default', help = 'Experiment ID')
		self.parser.add_argument('-test', action = 'store_true', help = 'test')
		self.parser.add_argument('-DEBUG', type = int, default = 0, help = 'DEBUG level')
		self.parser.add_argument('-demo', default = '', help = 'path/to/demo/image')
		self.parser.add_argument('-mode', default = 'pose', help = 'either pose or action')

		self.parser.add_argument('-loadModel', default = 'none', help = 'Provide full path to a previously trained model')
		self.parser.add_argument('-Model2D', default = 'models/xingy.pth', help = 'Provide full path to a model to inflate')
		self.parser.add_argument('-scheme', type = int, default = 1, help = 'inflation scheme')
		self.parser.add_argument('-mult', type = float, default = 0.1, help = 'inflation scheme1  mult factor')
		self.parser.add_argument('-isStateDict', default = 0, help = 'Whether the model to be loaded is stateDict')

		self.parser.add_argument('-nChannels', type = int, default = 256, help = '# features in the hourglass')
		self.parser.add_argument('-nStack', type = int, default = 2, help = '# hourglasses to stack')
		self.parser.add_argument('-nModules', type = int, default = 2, help = '# residual modules at each hourglass')
		self.parser.add_argument('-numReductions', type = int, default = 4, help = '# recursions in hourglass')
		self.parser.add_argument('-nRegModules', type = int, default = 2, help = '# depth regression modules')
		self.parser.add_argument('-nRegFrames', type = int, default = 2, help = '# number of frames temporally for regressor module')

		self.parser.add_argument('-freezefac', type = float, default = 0, help = '# freeze the central network by what factor')
		self.parser.add_argument('-freezeBN', type = int, default = 1, help = '# freeze the BatchNorm Layers')

		self.parser.add_argument('-nEpochs', type = int, default = 300, help = '# training epochs')
		self.parser.add_argument('-valIntervals', type = int, default = 4, help = '# valid intervel')
		self.parser.add_argument('-trainBatch', type = int, default = 1, help = '# mini-batch size')
		self.parser.add_argument('-dataloaderSize', type = int, default = 1, help = '# How many videos to load')

		self.parser.add_argument('-nFramesLoad', type = int, default = 6, help = '# Frames per video to consider furing training')
		self.parser.add_argument('-loadConsecutive', default=1, type = int, help = '# Load frames consecutively or sampling')

		self.parser.add_argument('-LRhg', type = float, default = 2.5e-5, help = '# Learning Rate for the Hourglass')
		self.parser.add_argument('-LRdr', type = float, default = 2.5e-5, help = '# Learning Rate for the depth regressor')
		self.parser.add_argument('-patience', type = int, default = 16 = '# patience for LR scheduler, kind of drop LR after \'patience\' number of frames' )
		self.parser.add_argument('-threshold', type = float, default = 0.0005, help = 'threshold for LR scheduler')
		self.parser.add_argument('-dropMag', type = float, default = 0.15, help = 'factor for LR scheduler, decrease LR by this factor')
		self.parser.add_argument('-scheduler', type = int, default = 3, help = 'drop LR')

		self.parser.add_argument('-ratioHM', type = int, default = 1, help = 'weak label data ratio')
		self.parser.add_argument('-loadMpii', action = 'store_true', help = 'test')
		self.parser.add_argument('-regWeight', type = float, default = 0.2, help = 'Depth regression loss weight')
		self.parser.add_argument('-hmWeight', type = float, default = 1, help = 'HeatMap loss weight')
		self.parser.add_argument('-varWeight', type = float, default = 0, help = 'Variance loss weight')
		self.parser.add_argument('-tempWeight', type = float, default = 0, help = 'Acceleration loss weight')

		self.parser.add_argument('-completeTest', type = int, default = 0, help = 'if you want to test on complete data')
		self.parser.add_argument('-startVal', type = int, default = 660, help = 'which frame in all videos to consider')
		self.parser.add_argument('-nVal', type = int, default = 120, help = 'number of frames to load from each video during validation')
		self.parser.add_argument('-gpu_id', type = int, default = 0, help = 'GPU ID for setting device')

	def parse(self):
		self.init()
		self.opt = self.parser.parse_args()
		self.opt.saveDir = os.path.join(ref.expDir, self.opt.expID)
		if self.opt.DEBUG > 0:
			ref.nThreads = 1

		args = dict((name, getattr(self.opt, name)) for name in dir(self.opt) if not name.startswith('_'))
		refs = dict((name, getattr(ref, name)) for name in dir(ref) if not name.startswith('_'))

		if not os.path.exists(self.opt.saveDir):
			os.makedirs(self.opt.saveDir)

		file_name = os.path.join(self.opt.saveDir, 'opt.txt')

		with open(file_name, 'wt') as opt_file:
			opt_file.write('==> Args:\n')
			for k, v in sorted(args.items()):
				opt_file.write('  %s: %s\n' % (str(k), str(v)))
			opt_file.write('==> Args:\n')
			for k, v in sorted(refs.items()):
				opt_file.write('  %s: %s\n' % (str(k), str(v)))
		return self.opt
