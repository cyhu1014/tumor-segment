# import library
import nibabel as nib
import numpy as np
import os
import torch
import datetime
#pytorch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

#torcvision
import torchvision.models as models
import torchvision.transforms as transforms

#import python file
from dataset import tumor_dataset
from train   import *

from models  import UNet3d ,UNet3d_vae,Generator
from loss    import loss_3d_crossentropy ,F1_Loss ,vae_loss

train_path = '../brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']

train_index = np.load('train.npy')
abs_path = '/tmp2/ryhu1014/'
abs_path = '../'
model_name = 'infogan'
os.makedirs(abs_path+model_name,exist_ok=True)
checkpoint_path =abs_path+model_name+'/'+model_name
start_epoch = 0
b_size = 8
workers = 2
classes = 5
x = 64 ; y = 64 ; z = 64
times = 4
num_epochs = 100 
nz = 1000
ngf = 64
ndf = 64
nc = 4
lr = 0.0002  # Learning rate for optimizers
beta1 = 0.5
beta2 = 0.999

train_set = tumor_dataset(path = train_path,out_index=train_index)
train_loader = DataLoader(train_set, batch_size=b_size,shuffle=False, num_workers=workers)
dataloader = train_loader


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv3d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()    
        )
    def forward(self, input):
        return self.main(input)

netG = Generator()
print(netG)
netD = Discriminator()
print(netD)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


criterion = nn.BCELoss()
real_label  = 1
fake_label = 0
G_losses = []
D_losses = []

netG.cuda()
netD.cuda()
netG.train()
netD.train()

def train () :
    best_G_loss = np.inf
    best_D_loss = np.inf
    best_g_epoch = 0
    best_d_epoch = 0
    for epoch in range(num_epochs):
        
        # For each batch in the dataloader
        D_loss = 0
        G_loss = 0
        for i, data in enumerate(dataloader, 0):
            print(i,end='\r')
            netD.zero_grad()
            real_cpu = data[0].cuda()
            real_cpu = cut_feat(real_cpu,64,64,64)
            label = torch.full((b_size,), real_label ).cuda()
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            noise = np.linspace(-2, 2, b_size*nz).reshape(1, -1)
            noise = torch.from_numpy(noise).float()
            noise = noise.view(b_size,-1, 1, 1, 1).cuda()
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            #-----netG
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            G_loss+=errG.item()
            D_loss+=errD.item()
        G_loss/=len(dataloader)
        D_loss/=len(dataloader)
        if(D_loss<best_D_loss):
            best_D_loss = D_loss
            best_d_epoch = epoch
            print('best_D: %.4f'%D_loss)
            save_checkpoint('best_D.pth',netD,optimizerD)
        if(G_loss<best_G_loss):
            best_G_loss = G_loss
            best_g_epoch = epoch
            print('best_G: ',G_loss)
            save_checkpoint('best_G.pth',netG,optimizerG)
        print('--best-- %d epoch ,G : %.4f  ; %d epoch ,D : %.4f'%(best_g_epoch,best_G_loss,best_d_epoch,best_D_loss))        
        print('epoch %d , G:%.4f,D: %.4f'%(epoch,G_loss,D_loss))
train()
