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

#torcvision
import torchvision.models as models
import torchvision.transforms as transforms

#import python file
from dataset import tumor_dataset
from train   import *
from models  import UNet3d ,UNet3d_vae
from loss    import loss_3d_crossentropy ,F1_Loss ,vae_loss

train_path = 'brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']

train_index = np.load('train.npy')
valid_index = np.load('valid.npy')

batch_size = 1
workers = 2
classes = 5
x = 120 ; y = 120 ; z = 152
start_epoch = 0
n_epochs    = 100
times = 4

train_set = tumor_dataset(path = train_path,out_index=train_index)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False, num_workers=workers)
dataloader =train_loader

model = UNet3d_vae(4,classes)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.5, 0.999))
model.train()

for epoch in range (start_epoch , n_epochs):
    n_loss = 0
    for i ,(feat,gt) in enumerate (dataloader):        
        # print(feat.shape,gt.shape)
        for idx in range (times):
            feat_cut ,_ = cut_feat_gt(feat,gt,x,y,z)                
            feat_cut = feat_cut.cuda()
            # gt_cut   = gt_cut.cuda()
            model.zero_grad()
            recon_img , mu , logvar= model(feat_cut)
            loss = F.binary_cross_entropy(recon_img,feat_cut , reduction='sum')
            loss.backward()
            optimizer.step()
            n_loss+=loss.item()
            print("[%d/%d],[%d/%d],[%d/%d],loss :%.4f"%(epoch+1,n_epochs,i+1,len(dataloader),idx+1,times,loss.item()),end = "\r")
    n_loss/=(len(dataloader)*times)
    print("[%d/%d],loss : %.4f"%(epoch+1,n_epochs,n_loss))
    # if(n_loss <best_loss):
        # best_loss = n_loss
    save_checkpoint('%s_epoch%d.pth'%(checkpoint_path,epoch+1) ,model ,optimizer )
        # print("save best at epoch:" ,epoch+1)
save_checkpoint('final_%s.pth'%checkpoint_path ,model ,optimizer )zs