from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json

import models.dcgan as dcgan
import models.mlp as mlp

if __name__=="__main__":

    config = './samples/generator_config.json'
    weights = './samples/netG_epoch_99.pth'
    output_dir = './img'
    nimages = 100

    with open(config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())
    
    imageSize = generator_config["imageSize"]
    nz = generator_config["nz"]
    nc = generator_config["nc"]
    ngf = generator_config["ngf"]
    noBN = generator_config["noBN"]
    ngpu = generator_config["ngpu"]
    mlp_G = generator_config["mlp_G"]
    n_extra_layers = generator_config["n_extra_layers"]

    if noBN:
        netG = dcgan.DCGAN_G_nobn(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif mlp_G:
        netG = mlp.MLP_G(imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # load weights
    netG.load_state_dict(torch.load(weights))

    # initialize noise
    fixed_noise = torch.FloatTensor(nimages, nz, 1, 1).normal_(0, 1)

    netG.cuda()
    fixed_noise = fixed_noise.cuda()

    fake = netG(fixed_noise)
    fake.data = fake.data.mul(0.5).add(0.5)

    for i in range(nimages):
        vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(output_dir, "%d.png"%(i+1)))
