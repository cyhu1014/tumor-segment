import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
#         print(x.shape)
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
#         print(x1.shape)
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
#         print(x1.shape)
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
class UNet3d (nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3d, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.outc = outconv(64, n_classes)
#         self.outc = outconv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         print(x1.shape,x2.shape,x3.shape,x4.shape)
        x = self.up1(x4, x3)
#         print(x1.shape,x2.shape,x.shape)
        x = self.up2(x, x2)
#         print(x1.shape,x.shape)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = F.softmax(x , 1)
        return x

class UNet3d_vae (nn.Module):
    def __init__(self,batchsize ,n_channels, n_classes,x1,y,z):
        super(UNet3d_vae, self).__init__()
        self.b = batchsize
        self.x1 = x1
        self.y  = y
        self.z  = z
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.outc = outconv(64, n_classes)
        ###vae part
        self.vae = nn.Sequential (
                    nn.Linear(x1*y*z//2 , 1024),
                    nn.ReLU()
        )
        self._enc_mu = torch.nn.Linear(1024, 128)
        self._enc_log_sigma = torch.nn.Linear(1024, 128)
        self.encoder = nn.Sequential(
                    nn.Linear(128 , 512),
                    nn.ReLU(),
                    nn.Linear(512 , x1*y*z//512),
                    nn.ReLU()
        )
        self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                    nn.Conv3d(1, 64, 3, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(64, 64, 3, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                    nn.Conv3d(64, 32, 3, padding=1),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(32, 32, 3, padding=1),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                    nn.Conv3d(32, 16, 3, padding=1),
                    nn.BatchNorm3d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(16, 16, 3, padding=1),
                    nn.BatchNorm3d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(16, 4, 3, padding=1),
                    nn.ReLU(inplace=True),
        )
    def forward(self, x):
        #decoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #encoder segment
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = F.softmax(x , 1)
        #encoder img
        x4 = x4.view(-1,(self.x1//8)*(self.y//8)*(self.z//8)*256)
        x4 = self.vae (x4)
        mu = self._enc_mu(x4)
        log_sigma = self._enc_log_sigma(x4)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()
        z =  mu + sigma * std_z # Reparameterization trick
        rec_img = self.encoder(z)
        rec_img = rec_img.view(-1,1,(self.x1//8),(self.y//8),(self.z//8))
        rec_img = self.up(rec_img)
       
        return rec_img, x ,mu ,sigma




class UNet3d_vae_2 (nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3d_vae, self).__init__()
        self.inc = nn.Sequential(
                    nn.Conv3d(4, 32, 3, padding=1),
                    nn.ReLU(),
        )
        self.down1 = nn.Sequential (
                    nn.MaxPool3d(2),
                    nn.Conv3d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(64, 64, 3, padding=1),
                    nn.ReLU(),
        )
        self.down2 = nn.Sequential (
                    nn.MaxPool3d(2),
                    nn.Conv3d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(128, 128, 3, padding=1),
                    nn.ReLU(),
        )
        self.down3 = nn.Sequential (
                    nn.MaxPool3d(2),
                    nn.Conv3d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(256, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(256, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(256, 256, 3, padding=1),
                    nn.ReLU(),
        )
        self.up = nn.Sequential (
                    nn.ConvTranspose3d(256, 128, 2, stride=2)
        )
        self.concate_conv = nn.Sequential (
                    nn.Conv3d(256, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(128, 128, 3, padding=1),
                    nn.ReLU(),
        )
        self.up2 = nn.Sequential (
                    nn.ConvTranspose3d(128, 64, 2, stride=2)
        )
        self.concate_conv2 = nn.Sequential (
                    nn.Conv3d(128, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(64, 64, 3, padding=1),
                    nn.ReLU(),
        )
        self.up3 = nn.Sequential (
                    nn.ConvTranspose3d(64, 32, 2, stride=2)
        )
        self.concate_conv3 = nn.Sequential (
                    nn.Conv3d(64, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(32, 5, 3, padding=1),
                    nn.ReLU(),
        )
        self.down4  = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv3d(256, 16, 3,stride = 2)
        )
        self.fc1   = nn.Linear(7056,256)
        self.fc21  = nn.Linear(256,128)
        self.fc22  = nn.Linear(256,128)
        self.fc3   = nn.Linear(128,15*15*19)
        self.up4 = nn.Sequential (
                    nn.ReLU(),
                    nn.Conv3d(1, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose3d(256, 128, 2, stride=2)
        )
        self.up5 = nn.Sequential (
                    nn.ReLU(),
                    nn.Conv3d(128, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose3d(64, 64, 2, stride=2),
        )
        self.up6 = nn.Sequential (
                    nn.ReLU(),
                    nn.Conv3d(64, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(32, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose3d(32, 32, 2, stride=2),
        )
        self.f   = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv3d(32, 4, 3, padding=1),
                    nn.Sigmoid()
        )
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        #decoder part   
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #segment part
        # out = self.up(x4)
        # out = torch.cat([out, x3], dim=1)
        # out = self.concate_conv(out)
        # out = self.up2(out)
        # out = torch.cat([out, x2], dim=1)
        # out = self.concate_conv2(out)
        # out = self.up3(out)
        # out = torch.cat([out, x1], dim=1)
        # out = self.concate_conv3(out)
        #vae partsz
        img = self.down4(x4)
        img = img.view(img.size(0), -1)
        mu, logvar = self.encode(img)
        z   = self.reparameterize(mu, logvar)
        z   = self.fc3(z)
        z   = z.view(-1,1,15,15,19)
        z   = self.up4(z)
        z   = self.up5(z)
        z   = self.up6(z)
        z   = self.f(z)
        
        return z , mu, logvar



