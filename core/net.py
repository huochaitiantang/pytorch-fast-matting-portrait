import torch
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #m.weight.data.normal_(0, math.sqrt(2. / n))
        torch.nn.init.orthogonal(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class MattNet(nn.Module):
    def __init__(self):
        super(MattNet, self).__init__()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=13, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=4, dilation=4, groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=6, dilation=6, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=8, dilation=8, groups=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=2,  kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear')

        # for stage2 training
        for p in self.parameters():
            p.requires_grad=False
        
        # feather
        self.newconvF1 = nn.Conv2d(in_channels=11, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.newconvF2 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        self.ReLU = nn.ReLU(inplace=True)
        weight_init(self.conv1)
        weight_init(self.conv2)
        weight_init(self.conv3)
        weight_init(self.conv4)
        weight_init(self.conv5)
        weight_init(self.conv6)
        weight_init(self.newconvF1)
        weight_init(self.newconvF2)

    def forward(self, x):
        conv1 = self.conv1(x)    ## conv 1
        pool1 = self.maxpool1(x)
        cat1 = torch.cat((conv1, pool1), 1)   ## cat 16

        conv2 = self.ReLU(self.conv2(cat1))  ## conv2 16
        cat2 = torch.cat((cat1, conv2), 1)   ## cat 32

        conv3 = self.ReLU(self.conv3(cat2))   ## conv3 16
        cat3 = torch.cat((cat2, conv3), 1)    ## cat 48
        
        conv4 = self.ReLU(self.conv4(cat3))   ## conv4 16
        cat4 = torch.cat((cat3, conv4), 1)    ## cat 64

        conv5 = self.ReLU(self.conv5(cat4))   ## conv5 16
        cat5 = torch.cat((conv2, conv3, conv4, conv5), 1) ## cat 64

        conv6 = self.ReLU(self.conv6(cat5))  ## conv6 2x64x64
        seg = self.interp(conv6)             ## 2x128x128

        #print("Forward:", seg[0,0,:,:].mean(), seg[0,1,:,:].mean())
        
        #return seg
        
        # shape: n 1 h w
        seg = F.softmax(seg, dim=1)
        bg, fg = torch.split(seg, 1, dim=1)
        # shape: n 3 h w
        imgSqr = x * x
        imgMasked = x * (torch.cat((fg, fg, fg), 1))
        # shape: n 11 h w
        convIn = torch.cat((x, seg, imgSqr, imgMasked), 1)
        newconvF1 =  self.ReLU(self.bn1(self.newconvF1(convIn)))
        newconvF2 = self.newconvF2(newconvF1)
        
        # fethering inputs:
        a, b, c = torch.split(newconvF2, 1, dim=1)

        #print("seg: {}".format(seg))
        alpha = a * fg + b * bg + c        
        alpha = self.sigmoid(alpha)

        return seg, alpha, a, b, c
