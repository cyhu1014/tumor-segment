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

train_path = '../brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']
index = np.arange(285)
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
        if(self.read_label==True):
            gt_path  = abs_path+'seg.nii.gz'
            gt = nib.load(gt_path)
            gt = gt.get_fdata()
        ##read image and normalize 
        if(self.read_image==True):
            feat     = nib.load(abs_path+type1[0]+'.nii.gz')
            feat     = feat.get_fdata()
            feat     = np.expand_dims(feat, axis=0)
            #feat     = normalize(feat)
            for i in range (1,4):
                feat1    = nib.load(abs_path+type1[i]+'.nii.gz')
                feat1    = feat1.get_fdata()
                #feat1    = normalize(feat1)
                feat1    = np.expand_dims(feat1, axis=0)
                feat     = np.concatenate((feat,feat1),axis=0)
            feat = torch.tensor(feat).type('torch.FloatTensor')
        if(self.read_image==False):
            return gt,self.list[self.out_index[index]]
        elif(self.read_label==False):
            return feat
        else:
            return feat ,gt
    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

b_size = 1
workers = 4
train_set = tumor_dataset(path = train_path,out_index=index,read_image = False)
train_loader = DataLoader(train_set, batch_size=b_size,shuffle=False, num_workers=workers)

txt_path = 'tumor_analysis_2d_xy_plane_z_axis.csv'
text_file = open(txt_path, "w")
text_file.write( 'file_name,z,label,min_x,max_x,min_y,max_y,x_size,y_size,\n')
for i,(label,abs_path) in enumerate(train_loader):
    
    for k in range (0,label.shape[3]):
        min_x = 0  ; min_y = 0 ; max_x = 0 ; max_y = 0
        with_label = 0
        for i in range (0,label.shape[1]):        
            if(label[0,i,:,k].sum()>0):
                min_x = i
                with_label = 1
                break  
        for i in range (label.shape[1]-1,-1,-1):
            if(label[0,i,:,k].sum()>0):
                max_x = i
                break
        for i in range (0,label.shape[2]):
            if(label[0,:,i,k].sum()>0):
                min_y = i
                break    
        for i in range (label.shape[2]-1,-1,-1):
            if(label[0,:,i,k].sum()>0):
                max_y = i
                break
        
        text_file.write('%s,%d,%d,%d,%d,%d,%d,%d,%d\n'%(abs_path[0],k,with_label,min_x,max_x,min_y,max_y,max_x-min_x,max_y-min_y))
text_file.close()
