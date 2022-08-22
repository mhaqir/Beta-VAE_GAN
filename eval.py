# from https://github.com/LMescheder/GAN_stability/blob/master/gan_training/eval.py
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from math import sqrt
from inception_score import inception_score

from scipy.stats import truncnorm

def truncated_z_sample(batch_size, z_dim, truncation=1., seed=None):
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim))
    return torch.from_numpy(truncation * values)


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0

def generator_postprocess(x):
    return x.add(1).div(2).clamp(0, 1)

def decoder_postprocess(x):
    return torch.sigmoid(x)

gp = generator_postprocess
dp = decoder_postprocess

class Evaluator(object):
    def __init__(self, generator, zdist, batch_size=64,
                 inception_nsamples=60000, device=None, dvae=None):
        self.generator = generator
        self.zdist = zdist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device
        self.dvae = dvae

    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            ztest = self.zdist.sample((self.batch_size,))

            samples = self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )

        return score, score_std

    def create_samples(self, z):
        self.generator.eval()
        batch_size = z.size(0)
        # print(z)
        # Sample x
        with torch.no_grad():
            x = self.generator(z)
        return x


