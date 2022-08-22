
import os

import torch
from torch.utils.data import Dataset
from torch import distributions
import torchvision

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def load_checkpoint(ckpt_dir, filename):
    return torch.load(os.path.join(ckpt_dir, filename))


def save_checkpoint(ckpt_dir, filename, net):
	torch.save(net.state_dict(), os.path.join(ckpt_dir, filename))

def get_zdist(dist_name, dim, device=None):
    # Get distribution
    if dist_name == 'uniform':
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gauss':
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist

def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5     # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)