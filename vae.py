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
# train_index = train_index[:256]

abs_path = '/tmp2/ryhu1014/'
model_name = 'vae_64^3_b4'
os.makedirs(abs_path+model_name,exist_ok=True)
checkpoint_path =abs_path+model_name+'/'+model_name
start_epoch = 56
batch_size = 4
workers = 2
classes = 5
x = 64 ; y = 64 ; z = 64
n_epochs    = 100
times = 4

train_set = tumor_dataset(path = train_path,out_index=train_index)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False, num_workers=workers)
dataloader = train_loader

model =UNet3d_vae(batch_size,4,classes,x,y,z)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001,betas=(0.5, 0.999))

if(start_epoch!=0):
    load_path = checkpoint_path+'_epoch%d.pth'%start_epoch
    load_checkpoint(load_path,model,optimizer)

model.train()
def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

criterion  = nn.MSELoss()
criterion2 = loss_3d_crossentropy(classes,x,y,z)
for epoch in range (start_epoch , n_epochs):
    n_loss = 0
    for i ,(feat,gt) in enumerate (dataloader):        
        # print(feat.shape,gt.shape)
        for idx in range (times):
            feat_cut ,gt_cut = cut_feat_gt(feat,gt,x,y,z)                
            feat_cut = feat_cut.cuda()
            gt_cut   = gt_cut.cuda()
            model.zero_grad()
            rec , pred ,mu , sigma= model(feat_cut)
            rec_i = rec.contiguous().view(-1)
            img = feat_cut.contiguous().view(-1) 
            loss = 0.1*criterion(rec_i , img)+criterion2(pred.double(),gt_cut).float()
            
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
save_checkpoint('final_%s.pth'%checkpoint_path ,model ,optimizer )
