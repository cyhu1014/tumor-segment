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
# train_index = train_index[:25]
valid_index = np.load('valid1.npy')
# valid_index = valid_index[:25]

start_epoch = 0
workers = 2
classes = 5
n_epochs    = 100
abs_path = '/tmp2/ryhu1014/'
abs_path = ''
model_name = '120120152_b1_shuf_t4_cv1'
os.makedirs(abs_path+model_name,exist_ok=True)
checkpoint_path =abs_path+model_name+'/'+model_name
restart = False
batch_size = 1
workers = 2
classes = 5
x = 120 ; y = 120 ; z = 152
n_epochs    = 100

train_set = tumor_dataset(path = train_path,out_index=train_index)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=workers)
valid_set = tumor_dataset(path = train_path,out_index=valid_index)
valid_loader = DataLoader(valid_set, batch_size=1,shuffle=False, num_workers=workers)

for i , (img,label,)