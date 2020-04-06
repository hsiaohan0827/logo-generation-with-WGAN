from __future__ import print_function
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



# model of Discriminator (DCGAN)
class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        output = self.main(input)
            
        output = output.mean(0)
        return output.view(1)



# model of Generator (DCGAN)
class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        
        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output 

if __name__=="__main__":

    dataroot = '../LLD'
    path_G = ''
    path_D = ''
    path_res = './samples'
    workers = 2
    batch_s = 64
    noise_s = 100
    epoch_s = 100
    ngf = 64
    ndf = 64
    ngpu = 1    
    g_lr = 0.00005
    d_lr = 0.00005
    clamp_lower = -0.01
    clamp_upper = 0.01
    Diter = 5
    
    if path_res is None:
        path_res = 'samples'
    os.system('mkdir {0}'.format(path_res))

    manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True

    
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                               transforms.Scale(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_s,
                                            shuffle=True, num_workers=int(workers))

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": 32, "nz": noise_s, "nc": 3, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": 0, "noBN": False, "mlp_G": False}
    with open(os.path.join(path_res, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = DCGAN_G(32, noise_s, 3, ngf, ngpu)
    netG.apply(weights_init)

    if path_G != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(path_G))
    print(netG)

    
    netD = DCGAN_D(32, noise_s, 3, ndf, ngpu)
    netD.apply(weights_init)

    if path_D != '':
        netD.load_state_dict(torch.load(path_D))
    print(netD)

    input = torch.FloatTensor(batch_s, 3, 32, 32)
    noise = torch.FloatTensor(batch_s, noise_s, 1, 1)
    fixed_noise = torch.FloatTensor(batch_s, noise_s, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * (-1)

    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    
    optimizerD = optim.RMSprop(netD.parameters(), lr = d_lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr = g_lr)

    gen_iterations = 0
    for epoch in range(0, epoch_s):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            
            # DISCRIMINATOR #
            
            # set discriminator gradient to True
            for p in netD.parameters():
                p.requires_grad = True 

            # set Diters number
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = Diter


            # train the discriminator Diters times
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)

                data = data_iter.next()
                i += 1

                # train with real                
                real_cpu, _ = data
                netD.zero_grad()
                batch_size = real_cpu.size(0)

                real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)
                
                # real img >> D >> D_real loss 
                errD_real = netD(inputv)
                errD_real.backward(one)

                # train with fake

                # generate noise
                noise.resize_(batch_s, noise_s, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile = True) # totally freeze netG

                # noise >> G >> fake img
                fake = Variable(netG(noisev).data)
                inputv = fake

                # fake img >> D >> D_fake loss (try to predict 0)
                errD_fake = netD(inputv)
                errD_fake.backward(mone)

                # D loss = min(real_loss - fake_loss)
                errD = errD_real - errD_fake
                optimizerD.step()

            
            # GENERATOR #

            # set discriminator gradient to False
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()

            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(batch_s, noise_s, 1, 1).normal_(0, 1)
            noisev = Variable(noise)

            # noise >> G >> fake img
            fake = netG(noisev)

            # fake img >> D >> D_fake loss (G loss) (try to predict 1)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()

            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, epoch_s, i, len(dataloader), gen_iterations,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            if gen_iterations % 500 == 0:
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{0}/real_samples.png'.format(path_res))
                fake = netG(Variable(fixed_noise, volatile=True))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(path_res, gen_iterations))

        # do checkpointing
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(path_res, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(path_res, epoch))
