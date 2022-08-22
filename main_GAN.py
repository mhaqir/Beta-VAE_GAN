#!/bin/python3

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from time import time

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from model import BetaVAE_H, Discriminator, Generator
from utils import str2bool, CustomTensorDataset, cuda, save_checkpoint, get_zdist, load_checkpoint, save_images
from train import Trainer
from checkpoints import CheckpointIO
from eval import Evaluator


batch_size = int(sys.argv[7])
num_workers = 0
z_dim = int(sys.argv[1])
nc = 1
lr = float(sys.argv[2]) #1e-4
beta1 = 0.9
beta2 = 0.999
num_epochs = int(sys.argv[3]) #25000
beta = float(sys.argv[4]) #0.4
ckpt_dir_vae = 'VAE-models'
ckpt_dir_gan = 'GAN_models'
# summary_dir_gan = 'GAN_summary'
data_dir = 'dsprites-dataset'

pos_thr = int(sys.argv[5])
inference_mode = str2bool(sys.argv[6])
w_info = 0.0001
reg_param = 10
is_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if is_cuda else 'cpu')
# writer = SummaryWriter(summary_dir_gan)


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# Importing datatset
data = np.load(os.path.join(data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), encoding='bytes')

imgs = data['imgs']

# large dataset casting to float on Carbonate interactive session crashes.
data = torch.from_numpy(imgs).unsqueeze(1).float() 

train_data = CustomTensorDataset(data)


if inference_mode == False:

	train_loader = DataLoader(train_data,
	                          batch_size=batch_size,
	                          shuffle=True,
	                          num_workers=num_workers,
	                          pin_memory=True,
	                          drop_last=True)


	vae_net = BetaVAE_H(z_dim, nc)
	disc_net = Discriminator(z_dim, size=64, nc = nc)
	gen_net = Generator(z_dim, size=64, nc = nc)

	vae_net = cuda(vae_net, is_cuda)
	disc_net = cuda(disc_net, is_cuda)
	gen_net = cuda(gen_net, is_cuda)

	model_name = 'model_{ne}_z_{z}_b_{b}_lr_{l}_pos_{t}_bs_{bs}'.format(ne = 100, \
		z = z_dim, b = beta, l = lr, t = pos_thr, bs=batch_size)
	ckpt = load_checkpoint(ckpt_dir_vae, model_name)
	vae_net.load_state_dict(ckpt)

	vae_optim = optim.Adam(vae_net.parameters(), lr=lr,
	                    betas=(beta1, beta2))
	disc_optim = optim.Adam(disc_net.parameters(), lr=lr,
	                    betas=(beta1, beta2))
	gen_optim = optim.Adam(gen_net.parameters(), lr=lr,
	                    betas=(beta1, beta2))

	############## To run the model on multiple gpu in parallel ############
	# vae_net = torch.nn.DataParallel(vae_net)
	# disc_net = torch.nn.DataParallel(disc_net)
	# gen_net = torch.nn.DataParallel(gen_net)
	########################################################################

	checkpoint_io = CheckpointIO(checkpoint_dir = ckpt_dir_gan)
	checkpoint_io.register_modules(gen_net = gen_net, disc_net = disc_net, \
									gen_optim = gen_optim, disc_optim = disc_optim)

	trainer = Trainer(vae_net, gen_net, disc_net, gen_optim, disc_optim, reg_param, w_info)

	zdist = get_zdist('gauss', 0, device)  # this is a guassian distribution, sample from it
	st = time()
	for i in tqdm(range(num_epochs)):
		# print(next(vae_net.parameters()).is_cuda)
		# print(next(disc_net.parameters()).is_cuda)
		# print(next(gen_net.parameters()).is_cuda)
		for x in train_loader:
			x = Variable(cuda(x, is_cuda))
			z = zdist.sample((batch_size,))
			dloss, reg, cs = trainer.discriminator_trainstep(x, z)

			z = zdist.sample((batch_size,))
			gloss, encloss = trainer.generator_trainstep(z, cs)

	print('dur:', time() - st)
	checkpoint_io.save(i, os.path.join(ckpt_dir_gan, \
		'model_gan_{ne}_z_{z}_b_{b}_lr_{l}_pos_{t}_bs_{bs}'.format(ne = 100, \
		z = z_dim, b = beta, l = lr, t = pos_thr, bs=batch_size)))
	# images = iter(train_loader).next()
	# writer.add_graph(model, images)
	# writer.close()
else:

	disc_net = Discriminator(z_dim, size=64, nc = nc)
	gen_net = Generator(z_dim, size=64, nc = nc)

	disc_net = cuda(disc_net, is_cuda)
	gen_net = cuda(gen_net, is_cuda)

	checkpoint_io = CheckpointIO(checkpoint_dir = ckpt_dir_gan)
	checkpoint_io.register_modules(gen_net = gen_net, disc_net = disc_net)

	zdist = get_zdist('gauss', 3, device)
	evaluator = Evaluator(gen_net, zdist, batch_size = batch_size, device = device)
	model_name = 'model_gan_{ne}_z_{z}_b_{b}_lr_{l}_pos_{t}_bs_{bs}'.format(ne = 100, \
		z = z_dim, b = beta, l = lr, t = pos_thr, bs=batch_size)
	model = checkpoint_io.load(os.path.join(ckpt_dir_gan, model_name))


###################### For z_dim = 3 ##########################
	# ztest_np = np.zeros((729, 3), dtype = np.float32)
	# idx = 0
	# for i in np.arange(-2, 2.5, 0.5):
	# 	for j in np.arange(-2, 2.5, 0.5):
	# 		for k in np.arange(-2, 2.5, 0.5):
	# 			ztest_np[idx, 0] = i
	# 			ztest_np[idx, 1] = j
	# 			ztest_np[idx, 2] = k
	# 			idx += 1

	# ztest = torch.from_numpy(ztest_np).to(device)

	# x = evaluator.create_samples(ztest)
	# save_images(x, '{}.pdf'.format(model_name), nrow = 27)
###############################################################
###################### For z_dim = 5 ##########################
	ztest_np = np.zeros((1024, 5), dtype = np.float32)
	idx = 0
	for i in np.arange(-2, 2, 1):
		for j in np.arange(-2, 2, 1):
			for k in np.arange(-2, 2, 1):
				for l in np.arange(-2, 2, 1):
					for o in np.arange(-2, 2, 1):
						ztest_np[idx, 0] = i
						ztest_np[idx, 1] = j
						ztest_np[idx, 2] = k
						ztest_np[idx, 3] = l
						ztest_np[idx, 4] = o
						idx += 1

	ztest = torch.from_numpy(ztest_np).to(device)

	x = evaluator.create_samples(ztest)
	save_images(x, '{}.pdf'.format(model_name), nrow = 32)
###############################################################


