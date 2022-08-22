
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import CustomTensorDataset, cuda, load_checkpoint
from model import BetaVAE_H

ckpt_dir = '/l/vision/joltik_hdd/mhaghir/VAE-models_whole_dsprite'
data_dir = '/l/vision/joltik_hdd/mhaghir/dsprites-dataset'
pos_thr = 32
num_workers = 0


rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)

# Importing datatset
data = np.load(os.path.join(data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), encoding='bytes')

# getting a specific shape and size and limit the movement on x-axis 
# latent classes in order of columns: color, shape, scale, rotation, position
imgs = data['imgs']
# print(np.shape(imgs))

latents_classes = data['latents_classes']
y  = imgs[latents_classes[:, 4] <= pos_thr] # images (filtering out some positions)
z = latents_classes[latents_classes[:, 4] <= pos_thr, :] # images
y = y[z[:, 1] == 0] # images (filtering out some of the shapes)
w = z[z[:, 1] == 0] # latent
t = y[w[:, 2] == 5]  # (select the largest scale)

# print(np.shape(y))
# t = imgs

# large dataset casting to float on Carbonate interactive session crashes.
data = torch.from_numpy(imgs).unsqueeze(1).float()

test_data = CustomTensorDataset(data)

test_loader = DataLoader(test_data,
                          batch_size=1,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True,
                          drop_last=True)

images1 = iter(test_loader).next()
images1 = Variable(cuda(images1, torch.cuda.is_available()))


models_dir = glob(ckpt_dir + '/*')

# print(len(models_dir))
model_names = [item[item.rfind('/') + 1:] for item in models_dir]
# model_params = [item.split('_') for item in model_names]
# print(model_params)

# To get the experiments with 100 epochs of training
# model_names = [model for model in model_names if model.split('_')[1] == '100' \
# 										and model.split('_')[11] == '128']
# model_params = [item.split('_') for item in model_names]


model_names = ['model_100_z_10_b_0.0_lr_0.0005_pos_32_bs_128']
model_params = [item.split('_') for item in model_names]

# print(model_params)
# print(len(model_params))


recons = []
nc = 1
print(torch.cuda.current_device())
print(torch.cuda.is_available())
for i in range(len(model_names)):
	z_dim = int(model_params[i][3])
	net = BetaVAE_H
	net = cuda(net(z_dim, nc), torch.cuda.is_available())

	### extract z_dim, append rec to the list and plot

	# model_name = 'model_{ne}_z_{z}_b_{b}_lr_{l}_pos_{t}'.format(ne = num_epochs, \
	# 	z = z_dim, b = beta, l = lr, t = pos_thr)
	ckpt = load_checkpoint(ckpt_dir, model_names[i])
	net.load_state_dict(ckpt)
	recon , mu, logvar= net.forward(images1)

	recon = recon.detach().cphtou().numpy()
	recons.append(recon)

images1 = images1.detach().cpu().numpy()
fig, axs = plt.subplots(nrows =1, ncols = 1, figsize=(7, 7), sharex=False)
# axs = axs.ravel()
for i in range(len(model_names)):
	axs.imshow(np.squeeze(recons[i]), cmap = 'gray')
	title = model_params[i][3] + '_' + model_params[i][5] + \
			'_' + model_params[i][7] + '_' + model_params[i][9]
	axs.set_title(title)
fig.savefig('models_100_bs_128_b_0.pdf', bbox_inches='tight')
plt.close()

