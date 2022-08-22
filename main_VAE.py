#!/bin/python3

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from model import BetaVAE_H
from utils import str2bool, CustomTensorDataset, cuda, save_checkpoint

batch_size = 128
num_workers = 0
z_dim = int(sys.argv[1])
nc = 1
lr = float(sys.argv[2]) #1e-4
beta1 = 0.9
beta2 = 0.999
num_epochs = int(sys.argv[3]) #25000
beta = float(sys.argv[4]) #0.4
ckpt_dir = 'VAE-models_whole_dsprite'
data_dir = 'dsprites-dataset'
pos_thr = int(sys.argv[5])
inference_mode = str2bool(sys.argv[6])


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


# getting a specific shape and size and limit the movement on x-axis 
# latent classes in order of columns: color, shape, scale, rotation, position
#################### To filter the dataset uncomment below ###########################
# latents_classes = data['latents_classes']
# y  = imgs[latents_classes[:, 4] <= pos_thr] # images (filtering out some positions)
# z = latents_classes[latents_classes[:, 4] <= pos_thr, :] # images
# y = y[z[:, 1]==0] # images (filtering out some of the shapes)
# w = z[z[:, 1]==0] # latent
# t = y[w[:, 2] == 5]  # (select the largest scale)
######################################################################################


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


	net = BetaVAE_H
	net = cuda(net(z_dim, nc), torch.cuda.is_available())

	optim = optim.Adam(net.parameters(), lr=lr,
	                    betas=(beta1, beta2))


	for i in tqdm(range(num_epochs)):
		for x in train_loader:
			x = Variable(cuda(x, torch.cuda.is_available()))
			x_recon, mu, logvar = net(x)
			recon_loss = reconstruction_loss(x, x_recon, 'bernoulli')
			total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
			beta_vae_loss = recon_loss + beta*total_kld
			optim.zero_grad()
			beta_vae_loss.backward()
			optim.step()
	save_checkpoint(ckpt_dir, 'model_{ne}_z_{z}_b_{b}_lr_{l}_pos_{t}_bs_{bs}'.format(ne = num_epochs, \
		z = z_dim, b = beta, l = lr, t = pos_thr, bs=batch_size), net)
else:

	test_loader = DataLoader(train_data,
	                          batch_size=1,
	                          shuffle=True,
	                          num_workers=num_workers,
	                          pin_memory=True,
	                          drop_last=True)

	images1 = iter(test_loader).next()
	# print(type(images1))
	# print(images1.size())
	images1 = Variable(cuda(images1, torch.cuda.is_available()))
	# recon = net._decode(net._encode(images1))
	net = BetaVAE_H
	net = cuda(net(z_dim, nc), torch.cuda.is_available())

	# optim = optim.Adam(net.parameters(), lr=lr,
	#                     betas=(beta1, beta2))
	model_name = 'model_{ne}_z_{z}_b_{b}_lr_{l}_pos_{t}_bs_{bs}'.format(ne = num_epochs, \
		z = z_dim, b = beta, l = lr, t = pos_thr, bs=batch_size)
	ckpt = load_checkpoint(ckpt_dir, model_name)
	net.load_state_dict(ckpt)
	recon , mu, logvar= net.forward(images1)
	# print(type(recon))
	# print(type(recon.detach().cpu().numpy()))
	recon = recon.detach().cpu().numpy()
	images1 = images1.detach().cpu().numpy()
	f, ax = plt.subplots(nrows =1, ncols = 2, figsize=(10, 7), sharex=False)
	# fig = plt.figure('hh')
	ax[0].imshow(np.squeeze(images1), cmap = 'gray')
	ax[1].imshow(np.squeeze(recon), cmap = 'gray')
	# ax[1, 0].imshow(np.squeeze(upad_frame), cmap = 'gray')
	# ax[1, 1].imshow(np.squeeze(upad_rec), cmap = 'gray')

	f.savefig('rec_{ne}_z_{z}_b_{b}_lr_{l}_pos_{t}_bs_{bs}.pdf'.format(ne = num_epochs, \
		z = z_dim, b = beta, l = lr, t = pos_thr, bs=batch_size), bbox_inches='tight')

