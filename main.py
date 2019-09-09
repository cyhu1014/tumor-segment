# import library
import nibabel as nib
import numpy as np
import os
import torch
import datetime
import pandas as pd
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

train_index = np.load('trainval_cut/train5.npy')

valid_index = np.load('trainval_cut/valid5.npy')

# train_index = train_index[:2]
# valid_index = valid_index[:2]

workers = 2
classes = 5
n_epochs    = 100
abs_path = '/tmp2/ryhu1014/'
abs_path = ''
model_name = '128^3_b1_shuf_t1_bbox_cv5'
model_name = 'testeeeee'

os.makedirs(abs_path+model_name,exist_ok=True)
checkpoint_path =abs_path+model_name+'/'+model_name
restart = False
batch_size = 1
workers = 2
classes = 5
x = 128 ; y = 128 ; z = 128
n_epochs    = 500

bbox_csv_path = 'tumor_analysis.csv'
bbox_csv = pd.read_csv(bbox_csv_path )
bbox_csv = bbox_csv.set_index('file_name')

train_set = tumor_dataset(path = train_path,out_index=train_index)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=workers)
valid_set = tumor_dataset(path = train_path,out_index=valid_index)
valid_loader = DataLoader(valid_set, batch_size=1,shuffle=False, num_workers=workers)


model =UNet3d(4,5)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001,betas=(0.5, 0.999))
# criterion = loss_3d_crossentropy(classes,x,y,z)
criterion = F1_Loss()
if(restart==True):
    load_path = checkpoint_path+'_final.pth'
    load_checkpoint(load_path,model,optimizer)

train(model,optimizer, train_loader,valid_loader ,criterion,checkpoint_path,x,y,z,n_epochs = 100 , times = 1  )
# train2(model,optimizer, train_loader,valid_loader ,criterion,checkpoint_path,bbox_csv,x,y,z,n_epochs = n_epochs , times = 1  )
