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

train_path = '../brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']

train_index = np.load('train1.npy')
train_index = train_index[:25]
valid_index = np.load('valid1.npy')
valid_index = valid_index[:25]

start_epoch = 0
batch_size = 4
workers = 2
classes = 5
x = 64 ; y = 64 ; z = 64
n_epochs    = 100
abs_path = '/tmp2/ryhu1014/'
abs_path = '../'
model_name = '64^3_b4_cv1'
os.makedirs(abs_path+model_name,exist_ok=True)
checkpoint_path =abs_path+model_name+'/'+model_name
start_epoch = 0
batch_size = 4
workers = 2
classes = 5
x = 64 ; y = 64 ; z = 64
n_epochs    = 100

train_set = tumor_dataset(path = train_path,out_index=train_index)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=workers)
valid_set = tumor_dataset(path = train_path,out_index=valid_index)
valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=True, num_workers=workers)


model =UNet3d(4,5)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001,betas=(0.5, 0.999))
criterion = loss_3d_crossentropy(classes,x,y,z)
if(start_epoch!=0):
    load_path = checkpoint_path+'_epoch%d.pth'%start_epoch
    load_checkpoint(load_path,model,optimizer)

train(model,optimizer, train_loader,valid_loader ,criterion,checkpoint_path,x,y,z,n_epochs = 100 , times = 1,start_epoch = start_epoch )
