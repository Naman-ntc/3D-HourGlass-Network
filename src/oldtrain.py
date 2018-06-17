import cv2
import ref
import torch
import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt


from my import *
from Losses import *
from utils.eval import *
from visualise import *
from utils.utils import *
from myutils import *



def step(split, epoch, opt, dataLoader, model, optimizer = None):
	if split == 'train':
		model.train()
		if opt.freezeBN:
			model.apply(set_bn_eval)
	else:
		model.eval()
	Loss2D, Loss3D, LossTemp, Mpjpe, Acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

	nIters = len(dataLoader)
	bar = Bar('==>', max=nIters)
	totalFrames = 0
	for i, (input, targetMaps, target2D, target3D, meta) in enumerate(dataLoader):
		if (split == 'val'):
			if (input == -1).all():
				continue
			input_var = torch.autograd.Variable(input,volatile=True).float().cuda()
			targetMaps = torch.autograd.Variable(targetMaps,volatile=True).float().cuda()
			target2D_var = torch.autograd.Variable(target2D,volatile=True).float().cuda()
			target3D_var = torch.autograd.Variable(target3D,volatile=True).float().cuda()
			meta = torch.autograd.Variable(meta,volatile=True).float().cuda()	
		else:
			input_var = torch.autograd.Variable(input).float().cuda()
			targetMaps = torch.autograd.Variable(targetMaps).float().cuda()
			target2D_var = torch.autograd.Variable(target2D).float().cuda()
			target3D_var = torch.autograd.Variable(target3D).float().cuda()
			meta = torch.autograd.Variable(meta).float().cuda()
		
		totalFrames += input_var.shape[0]*input_var.shape[2]
		model = model.float()
		output = model(input_var)
		reg = output[opt.nStack]
		if opt.DEBUG == 2:
			writeCSV('../CSV/%d.csv'%(i),csvFrame((output[opt.nStack - 1].data).cpu().numpy(), (reg.data).cpu().numpy(), meta))
			# for j in range(input_var.shape[2]):
				#plt.imshow(input_var.data[0,:,j,:,:].transpose(0,1).transpose(1,2).cpu().numpy())
				#test_heatmaps(targetMaps[0,:,j,:,:].cpu(),input_var[0,:,j,:,:].cpu(),6)
				# a = np.zeros((16,3))
				# b = np.zeros((16,3))
				# a[:,:2] = getPreds(targetMaps[:1,:,j,:,:].cpu().numpy())
				# b[:,:2] = getPreds(output[opt.nStack - 1][:1,:,j,:,:].data.cpu().numpy())
				# a[:,2] = target3D[0,:,j,0].cpu().numpy()
				# b[:,2] = reg[0,:,j,0].data.cpu().numpy()
				# visualise3d(b,a,"3D",i,j,input_var.data[0,:,j,:,:].transpose(0,1).transpose(1,2).cpu())

		"""
		Depth Squared Error loss here
		"""

		if ((meta == 0).all()):
			"""
			USE BATCH SIZE ONE ONLY!!!
			"""
			loss = torch.autograd.Variable(torch.Tensor([0])).cuda()
			oldloss = 0
		else:
			loss = opt.regWeight * JointsDepthSquaredError(reg,target3D_var)
			oldloss = float((loss).detach())
			Loss3D.update(float((loss).detach()), input.size(0))

		"""
		HeatMap squared Error loss here
		"""

		for k in range(opt.nStack):
			loss += opt.hmWeight * Joints2DHeatMapsSquaredError(output[k], targetMaps)

		Loss2D.update(float((loss).detach()) - oldloss, input.size(0))
		oldloss = float((loss).detach())

		tempAcc = Accuracy((output[opt.nStack - 1].data).transpose(1,2).contiguous().view(-1,ref.nJoints,ref.outputRes,ref.outputRes).contiguous().cpu().numpy(), (targetMaps.data).transpose(1,2).contiguous().view(-1,ref.nJoints,ref.outputRes,ref.outputRes).contiguous().cpu().numpy())
		Acc.update(tempAcc)

		mplist = myMPJPE((output[opt.nStack - 1].data).cpu().numpy(), (reg.data).cpu().numpy(), meta.data.cpu())

		for l in mplist:
			mpjpe, num3D = l
			if num3D > 0:
				Mpjpe.update(mpjpe, num3D)
		tempMPJPE = (sum([(x*y if y>0 else 0) for x,y in mplist]))/(1.0*sum([(y if y>0 else 0) for x,y in mplist])) if (1.0*sum([(y if y>0 else 0) for x,y in mplist])) > 0 else 0

		if opt.DEBUG == 3 and (float(tempMPJPE) > 80):
			# for j in range(input_var.shape[2]):
			# 	#test_heatmaps(targetMaps[0,:,j,:,:].cpu(),input_var[0,:,j,:,:].cpu(),6)
			# 	a = np.zeros((16,3))
			# 	b = np.zeros((16,3))
			# 	a[:,:2] = getPreds(targetMaps[:,:,j,:,:].cpu().numpy())
			# 	b[:,:2] = getPreds(output[opt.nStack - 1][:,:,j,:,:].data.cpu().numpy())
			# 	b[:,2] = reg[0,:,j,0].detach().cpu().numpy()
			# 	a[:,2] = target3D_var[0,:,j,0].cpu().numpy()
			# 	visualise3d(b,a,'val-errors-great',i,j,input_var.data[0,:,j,:,:].transpose(0,1).transpose(1,2).cpu().numpy())
			pass


		"""
		Temporal Losses here
		"""

		#rootRelative = give3D(output[opt.nStack -1], reg, meta)
		#loss += opt.tempWeight * AccelerationMatchingError(rootRelative, meta)
		#LossTemp.update(float((loss).detach()) - oldloss, input.size(0))

		if split == 'train':
			loss = loss/opt.trainBatch
			loss.backward()
			if ((i+1)%(opt.trainBatch/opt.dataloaderSize) == 0):
				optimizer.step()
				optimizer.zero_grad()


		Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss2D {loss.avg:.6f} | Loss3D {loss3d.avg:.6f} | LossTemp {losstemp.avg:.6f} | ACC {PCKh.avg:.3f} | Mpjpe {Mpjpe.avg:.3f} ({Mpjpe.val:.3f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss2D, split=split, loss3d=Loss3D, losstemp=LossTemp, Mpjpe=Mpjpe, PCKh=Acc)
		bar.next()

	bar.finish()
	if (opt.completeTest):
		print("Num Frames : %d"%(totalFrames))
		return Mpjpe.avg, totalFrames
	return Loss2D.avg, Loss3D.avg, Mpjpe.avg, Acc.avg


def train(epoch, opt, train_loader, model, optimizer):
	return step('train', epoch, opt, train_loader, model, optimizer)

def val(epoch, opt, val_loader, model):
	#with torch.no_grad():
	return step('val', epoch, opt, val_loader, model)
