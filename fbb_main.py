#find bounding box 

# import library
import nibabel as nib
import numpy as np
import pandas as pd

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
from models  import UNet3d ,UNet3d_vae,model_bbox
from loss    import loss_3d_crossentropy ,F1_Loss ,vae_loss

train_path = '../brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']
csv = pd.read_csv('tumor_analysis.csv')
csv = csv.set_index('file_name')     
train_index = np.load('train.npy')
valid_index = np.load('valid.npy')
valid_index = valid_index[:24]
class tumor_dataset(Dataset):
    def __init__(self,path,out_index,csv,transform = None,read_label=True,read_image=True):
        self.path = path
        self.list = sorted(os.listdir(self.path))
        self.out_index=out_index
        self.len  = len(self.out_index)
        self.csv = csv
#         self.transform = transform
        self.read_label = read_label
        self.read_image = read_image
        print('datalen: ',self.len)
    def __getitem__(self, index):
        ##set path
        abs_path = self.path+self.list[self.out_index[index]]+'/'+self.list[self.out_index[index]]+'_'
        ##read ground truth
        if(self.read_label==True):
            gt_path  = abs_path+'seg.nii.gz'
            gt = nib.load(gt_path)
            gt = gt.get_fdata()
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
        if(self.read_image==False):
            return gt,self.list[self.out_index[index]]
        elif(self.read_label==False):
            ans = np.zeros(6)
            ans [0] = self.csv.loc[self.list[self.out_index[index]]]['min_x']
            ans [1] = self.csv.loc[self.list[self.out_index[index]]]['max_x']
            ans [2] = self.csv.loc[self.list[self.out_index[index]]]['min_y']
            ans [3] = self.csv.loc[self.list[self.out_index[index]]]['max_y']
            ans [4] = self.csv.loc[self.list[self.out_index[index]]]['min_z']
            ans [5] = self.csv.loc[self.list[self.out_index[index]]]['max_z']
            return feat,ans,self.list[self.out_index[index]]
        else:
            return feat ,gt,self.list[self.out_index[index]]
    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

def test(valid_loader):
    model.eval()
    epoch_loss  = 0 
    with torch.no_grad():
        for i,(img,label,fp) in enumerate (valid_loader):
            print(i,end='\r')
            img = img.cuda()
            label = label.float().cuda()
            pred = model(img)
            loss = criterion(pred,label)
            epoch_loss+=loss.detach().item()
        epoch_loss/=len(valid_loader) 
        print('\n')
    return epoch_loss

def main ():
    b_size = 2
    workers = 0
    n_epochs = 100

    train_set = tumor_dataset(path = train_path,out_index=train_index,csv=csv,read_label=False)
    train_loader = DataLoader(train_set, batch_size=b_size,shuffle=True, num_workers=workers)
    valid_set = tumor_dataset(path = train_path,out_index=valid_index,csv=csv,read_label=False)
    valid_loader = DataLoader(valid_set, batch_size=b_size,shuffle=False, num_workers=workers)
    dataloader = train_loader

    model = model_bbox(b_size,4)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.5, 0.999))
    criterion =nn.MSELoss()

    total_loss = []
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss  = 0
        for i ,(img,label,_) in enumerate(dataloader):
            model.zero_grad()
            img = img.cuda()
            label = label.float().cuda()
            pred = model(img)
            loss = criterion(pred,label)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.detach().item()
            print('[%d/%d],[%d/%d],loss : %.4f'%(epoch,n_epochs,i,len(dataloader),loss.detach().item()),end='\r')
        print('\n')
        epoch_loss/=len(dataloader)       
        total_loss.append(epoch_loss)
        valid_loss =test(valid_loader)
        if(best_loss >=valid_loss):
            best_loss = valid_loss
            best_epoch = epoch
            save_checkpoint('best_fbb.pth',model,optimizer)
            print('save best')
        save_checkpoint('final_fbb.pth',model,optimizer)
        print('-----------------------Epoch : %d ,train loss : %.4f ,valid_loss %.4f -------------Best : %d , loss %.4f----------------------'%(epoch,epoch_loss,valid_loss,best_epoch,best_loss))
        



