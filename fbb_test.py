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
from fbb_main import *

train_path = '../brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']
csv = pd.read_csv('tumor_analysis.csv')
csv = csv.set_index('file_name')     
train_index = np.load('train.npy')
valid_index = np.load('valid.npy')
valid_index  = valid_index[:24]

b_size = 2
workers = 0

model = model_bbox(b_size,4)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.5, 0.999))
criterion =nn.MSELoss()

load_checkpoint('best_fbb.pth',model,optimizer)

train_set = tumor_dataset(path = train_path,out_index=train_index,csv=csv,read_label=False)
train_loader = DataLoader(train_set, batch_size=b_size,shuffle=True, num_workers=workers)
valid_set = tumor_dataset(path = train_path,out_index=valid_index,csv=csv,read_label=False)
valid_loader = DataLoader(valid_set, batch_size=b_size,shuffle=False, num_workers=workers)

model.eval()


txt_path = 'predict_fbb.csv'
text_file = open(txt_path, "w")
text_file.write( 'file_name,min_x,max_x,min_y,max_y,min_z,max_z,min_x,max_x,min_y,max_y,min_z,max_z,loss\n')

with torch.no_grad():
    for i,(img,label,fp) in enumerate (valid_loader):
        print(i,end='\r')
        img = img.cuda()
        label = label.float().cuda()
        pred = model(img)
        loss1 = criterion(pred[0],label[0])
        loss2 = criterion(pred[1],label[1])
        p = 0
        text_file.write('%s,%d,%d,%d,%d,%d,%d'%(fp[p],label[p][0].item(),label[p][1].item(),label[p][2].item(),label[p][3].item(),label[p][4].item(),label[p][5].item()))
        text_file.write(',%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(pred[p][0].item(),pred[p][1].item(),pred[p][2].item(),pred[p][3].item(),pred[p][4].item(),pred[p][5].item(),loss1.item()))
        p = 1
        text_file.write('%s,%d,%d,%d,%d,%d,%d'%(fp[p],label[p][0].item(),label[p][1].item(),label[p][2].item(),label[p][3].item(),label[p][4].item(),label[p][5].item()))
        text_file.write(',%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(pred[p][0].item(),pred[p][1].item(),pred[p][2].item(),pred[p][3].item(),pred[p][4].item(),pred[p][5].item(),loss1.item()))
        
        
        
        
text_file.close()

    


