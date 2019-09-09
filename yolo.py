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

class tumor_dataset(Dataset):
    def __init__(self,path,out_index,transform = None,read_label=True,read_image=True):
        self.path = path
        self.list = sorted(os.listdir(self.path))
        self.out_index=out_index
        self.len  = len(self.out_index)
#         self.transform = transform
        self.read_label = read_label
        self.read_image = read_image
        print('datalen: ',self.len)
    def __getitem__(self, index):
        ##set path
        abs_path = self.path+self.list[self.out_index[index]]+'/'+self.list[self.out_index[index]]+'_'
        ##read ground truth
        
        ##read image and normalize 
        if(self.read_image==True):
            feat     = nib.load(abs_path+type1[0]+'.nii.gz')
            feat     = feat.get_fdata()
            feat     = np.expand_dims(feat, axis=0)
            feat     = normalize(feat)
            for i in range (1,4):
                feat1    = nib.load(abs_path+type1[i]+'.nii.gz')
                feat1    = feat1.get_fdata()
                feat1    = normalize(feat1)
                feat1    = np.expand_dims(feat1, axis=0)
                feat     = np.concatenate((feat,feat1),axis=0)
            feat = torch.tensor(feat).type('torch.FloatTensor')
        get_yololabel(csv,self.list[self.out_index[index]],50)
        return feat,self.list[self.out_index[index]]
        
       
    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


box_length = 30
def get_yololabel (csv,filename,axis):
    min_x = csv.loc[(filename, axis)]['min_x'] 
    max_x = csv.loc[(filename, axis)]['max_x'] 
    min_y = csv.loc[(filename, axis)]['min_y'] 
    max_y = csv.loc[(filename, axis)]['max_y'] 
    x_size = csv.loc[(filename, axis)]['x_size'] 
    y_size = csv.loc[(filename, axis)]['y_size'] 
    new_bbox = np.zeros(8,8,4)
    

    print(min_x,min_y,max_x,max_y)
train_set = tumor_dataset(path = train_path,out_index=train_index)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=workers)
valid_set = tumor_dataset(path = train_path,out_index=valid_index)
valid_loader = DataLoader(valid_set, batch_size=1,shuffle=False, num_workers=workers)
csv = pd.read_csv('tumor_analysis_2d_xy_plane_z_axis.csv' )
csv.set_index(keys=['file_name','z'],inplace=True)   

for i , (img,filename) in enumerate(train_loader):
    print(img.shape,filename)
    break





