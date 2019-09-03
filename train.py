import numpy as np
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
def train(model,optimizer, dataloader,valid_loader ,criterion,checkpoint_path,x,y,z,n_epochs = 100 , times = 1 ,start_epoch = 0 ):
    best_loss = np.inf
    model.train()
    best_f1 = 0
    best_epoch = 0
    for epoch in range (start_epoch , n_epochs):
        n_loss = 0
        for i ,(feat,gt) in enumerate (dataloader):        
            # print(feat.shape,gt.shape)
            for idx in range (times):
                feat_cut ,gt_cut = cut_feat_gt(feat,gt,x,y,z)                
                feat_cut = feat_cut.cuda()
                gt_cut   = gt_cut.cuda()
                # print(feat_cut.shape,gt_cut.shape)
                model.zero_grad()
                pred = model(feat_cut)
                loss = criterion(pred.double(),gt_cut)
                loss.backward()
                optimizer.step()
                n_loss+=loss.item()
                print("[%d/%d],[%d/%d],[%d/%d],loss :%.4f"%(epoch+1,n_epochs,i+1,len(dataloader),idx+1,times,loss.item()),end = "\r")
        n_loss/=(len(dataloader)*times)
        print("[%d/%d],loss : %.4f"%(epoch+1,n_epochs,n_loss),checkpoint_path)
        # if(n_loss <best_loss):
            # best_loss = n_loss
        f1_score = test2(model,valid_loader,240,240,155,x,y,z)
        print(f1_score)
        if(f1_score > best_f1):
            best_f1 = f1_score
            best_epoch = epoch
            save_checkpoint('%s_best.pth'%checkpoint_path ,model ,optimizer) 
        print('[%d] : %.4f      --BEST--[%d] : %.4f                         '%(epoch,f1_score , best_epoch,best_f1))
        txt_path = '%s.csv'%checkpoint_path
        text_file = open(txt_path, "a")
        text_file.write( '%d,%.4f\n'%(epoch,f1_score))
        text_file.close()
        save_checkpoint('%s_final.pth'%checkpoint_path ,model ,optimizer )

    
def test(model,dataloader,times= 8):
    model.eval()
    macro_f1 = 0
    with torch.no_grad():
        for i ,(feat,gt) in enumerate (dataloader):
            print(i,end='\r')
            feat  = feat.cuda()
            gt    = gt
            feat1 = feat[:,:,:120,:120,0:152]
            feat2 = feat[:,:,:120,120:,0:152]
            feat3 = feat[:,:,120:,120:,0:152]
            feat4 = feat[:,:,120:,:120,0:152]
            gt1   = gt[:,:120,:120,0:152].numpy().reshape(-1)
            gt2   = gt[:,:120,120:,0:152].numpy().reshape(-1)
            gt3   = gt[:,120:,120:,0:152].numpy().reshape(-1)
            gt4   = gt[:,120:,:120,0:152].numpy().reshape(-1)
            pred1 = model(feat1)
            pred2 = model(feat2)
            pred3 = model(feat3)
            pred4 = model(feat4)
            pred1 = torch.argmax(pred1,1).cpu().numpy().reshape(-1)
            pred2 = torch.argmax(pred2,1).cpu().numpy().reshape(-1)
            pred3 = torch.argmax(pred3,1).cpu().numpy().reshape(-1)
            pred4 = torch.argmax(pred4,1).cpu().numpy().reshape(-1)
            pred1 = np.concatenate((pred1,pred2))
            pred1 = np.concatenate((pred1,pred3))
            pred1 = np.concatenate((pred1,pred4))
            gt1 = np.concatenate((gt1,gt2))
            gt1 = np.concatenate((gt1,gt3))
            gt1 = np.concatenate((gt1,gt4))
            macro_f1+=f1_score(gt1, pred1,average='macro')
    return macro_f1/len(dataloader)

def test_rand(model,dataloader,criterion,x,y,z,times= 8):
    model.eval()
    macro_f1 = 0
    f1_loss  = 0
    with torch.no_grad():
        for i ,(feat,gt) in enumerate (dataloader):
            print(i,end='\r')
            feat  = feat.cuda()
            gt    = gt
            for idx in range (times):
                half_x_length = x//2 
                half_y_length = y//2 
                half_z_length = z//2 
                mid_x = np.random.randint(x//2,feat.shape[2]-x//2)
                mid_y = np.random.randint(y//2,feat.shape[3]-y//2)
                mid_z = np.random.randint(z//2,feat.shape[4]-z//2)
                # print(mid_x,mid_y,mid_z)
                feat_cut = feat[:,:,mid_x-half_x_length:mid_x+half_x_length,mid_y-half_y_length:mid_y+half_y_length,mid_z-half_z_length:mid_z+half_z_length]
                gt_cut   = gt[:,mid_x-half_x_length:mid_x+half_x_length,mid_y-half_y_length:mid_y+half_y_length,mid_z-half_z_length:mid_z+half_z_length]
#                     print(feat_cut.shape,gt_cut.shape)
                feat_cut = feat_cut.cuda()
                gt_cut   = gt_cut.cuda()
                # print(feat_cut.shape,gt_cut.shape)
                pred = model(feat_cut)
                loss = criterion(pred.double(),gt_cut)
                f1_loss+=loss.item()
                gt1   = gt_cut.cpu().numpy().reshape(-1)
                pred = torch.argmax(pred,1).cpu().numpy().reshape(-1)
                macro_f1+=f1_score(gt1, pred,average='macro')
        macro_f1 = macro_f1/(len(dataloader)*times)
        f1_loss  = f1_loss/(len(dataloader)*times)
    return macro_f1 , f1_loss



def save_checkpoint(checkpoint_path,model,optimizer):
    state = {'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path,model,optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def cut_train_val (total_len, train_len):
    ##cut train and val
    arr = np.arange(total_len)
    np.random.shuffle(arr)
    train_index = arr[:train_len]
    valid_index   = arr[train_len:]
    np.save('train.npy', train_index)
    np.save('valid.npy', valid_index)
    return train_index , valid_index

def normalize (ndarray):
    std = np.std(ndarray)
    mean = np.mean(ndarray)
    ndarray = (ndarray-mean)/std
    return ndarray


def cut_feat_gt (feat, gt,x,y,z):
    half_x_length = x//2 
    half_y_length = y//2 
    half_z_length = z//2 
    mid_x = np.random.randint(x//2,feat.shape[2]-x//2)
    mid_y = np.random.randint(y//2,feat.shape[3]-y//2)
    mid_z = np.random.randint(z//2,feat.shape[4]-z//2)
    # print(mid_x,mid_y,mid_z)
    feat_cut = feat[:,:,mid_x-half_x_length:mid_x+half_x_length,mid_y-half_y_length:mid_y+half_y_length,mid_z-half_z_length:mid_z+half_z_length]
    gt_cut   = gt[:,mid_x-half_x_length:mid_x+half_x_length,mid_y-half_y_length:mid_y+half_y_length,mid_z-half_z_length:mid_z+half_z_length]
    return feat_cut , gt_cut

def cut_feat (feat,x,y,z):
    half_x_length = x//2 
    half_y_length = y//2 
    half_z_length = z//2 
    mid_x = np.random.randint(x//2,feat.shape[2]-x//2)
    mid_y = np.random.randint(y//2,feat.shape[3]-y//2)
    mid_z = np.random.randint(z//2,feat.shape[4]-z//2)
    # print(mid_x,mid_y,mid_z)
    feat_cut = feat[:,:,mid_x-half_x_length:mid_x+half_x_length,mid_y-half_y_length:mid_y+half_y_length,mid_z-half_z_length:mid_z+half_z_length]
    return feat_cut 

def test2 (model , dataloader ,X,Y,Z, x, y ,z) :
    model.eval()
    macro_f1 = 0
    with torch.no_grad():
        for idx ,(feat,gt) in enumerate (dataloader):
            feat = feat.cuda()
            p = np.zeros((1,)) 
            g = np.zeros((1,))
            for i in range (0,X,x):
                for j in range (0,Y,y):
                    for k in range (0,Z,z):
                        x1 = i
                        cut_x1 = 0
                        x2 = i+x
                        if (x2>X):
                            x2 = X
                            tp_x1 = x1
                            x1 = x2-x
                            cut_x1 = tp_x1-x1
                        y1 = j
                        cut_y1 = 0
                        y2 = j+y
                        if (y2>Y):
                            y2 = Y
                            tp_y1 = y1
                            y1 = y2-y
                            cut_y1 = tp_y1-y1
                        z1 = k
                        cut_z1 = 0
                        z2 = k+z
                        if (z2>Z):
                            z2 = Z
                            tp_z1 = z1
                            z1 = z2-z 
                            cut_z1 = tp_z1-z1 
                        feat_cut = feat[:,:,x1:x2,y1:y2,z1:z2]
                        gt_cut = gt[:,x1:x2,y1:y2,z1:z2]
                        pred = model(feat_cut)
                        pred = torch.argmax(pred,1).cpu()
                        pred  = pred[:,cut_x1:,cut_y1:,cut_z1:]
                        gt_cut = gt_cut[:,cut_x1:,cut_y1:,cut_z1:]
                        pred   = pred.numpy().reshape(-1)
                        gt_cut = gt_cut.numpy().reshape(-1)
                        p = np.concatenate((p,pred))       
                        g = np.concatenate((g,gt_cut))  
            p = p[1:]                                                                                   
            g = g[1:]
            fs = f1_score(g, p,average='macro')
            macro_f1+=fs
            print(idx, p.shape ,fs,end='\r')
        macro_f1 = macro_f1/len(dataloader)
    return macro_f1