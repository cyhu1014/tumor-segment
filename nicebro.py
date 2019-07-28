X = 240
Y = 240
Z = 155
import numpy as np
def test_vae (X,Y,Z, x, y ,z) :
    feat = np.zeros((240,240,155))
    p = np.zeros((1,)) 
    num = 0
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
                num+=1
                feat1 = feat[x1:x2,y1:y2,z1:z2]
                pred  = feat1[cut_x1:,cut_y1:,cut_z1:]
                print(num,x1,x2,y1,y2,z1,z2,feat1.shape,pred.shape,pred.shape[0]*pred.shape[1]*pred.shape[2])
                print(x1%x,y1%y,z1%z)
                pred = pred.reshape(-1)
                p = np.concatenate((p,pred))    
    return p
8928000
print(test_vae(X,Y,Z,48,48,48).shape)