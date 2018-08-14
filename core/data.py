import torch
import cv2
import os
import random
import numpy as np
from torchvision import transforms

class MatTransform(object):
    def __init__(self, size, flip=False):
        self.size = size
        self.flip = flip

    def __call__(self, img, msk, alpha, name):
        scale_h = float(self.size) / img.shape[0]
        scale_w = float(self.size) / img.shape[1]
        img = cv2.resize(img,(self.size, self.size),interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk,(self.size, self.size),interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha,(self.size, self.size),interpolation=cv2.INTER_LINEAR)
        if self.flip and random.random() < 0.5:
            img = cv2.flip(img, 1)
            msk = cv2.flip(msk, 1)
            alpha = cv2.flip(alpha, 1)
        return img, msk, alpha, [name, scale_h, scale_w]


class MatDataset(torch.utils.data.Dataset):
    def __init__(self, namelist, imgdir, mskdir, alphadir, normalize=None, transform=None):
        self.sample_set=[]
        self.transform=transform
        self.normalize=normalize
        with open(namelist, 'r') as f:
            names = f.readlines()
        print('\t--namelist: {}'.format(namelist))
        print('\t--names: {}'.format(len(names)))
        for name in names:
            name = name.strip('\n')
            img_path = imgdir + '/' + name + '.jpg'
            msk_path = mskdir + '/' + name + '.png'
            alpha_path = alphadir + '/' + name + '.png'
            if os.path.exists(img_path) and os.path.exists(msk_path) and os.path.exists(alpha_path):
                #size:[800, 600, 3] value:0-255 order BGR
                img = cv2.imread(img_path)
                #size:[800, 600] value:0 or 1
                msk = (cv2.imread(msk_path) / 255)[:, :, 0]
                #size:[800, 600] value: 0-255
                alpha = (cv2.imread(alpha_path))[:, :, 0]
                # check
                #cnt0 = len(np.where(alpha >= 0)[0])
                #cnt1 = len(np.where(alpha == 0)[0])
                #cnt2 = len(np.where(alpha == 255)[0])
                #print("alpha: all:{} fg:{} bg:{}".format(cnt0, cnt2, cnt1))
                #assert(cnt0 != (cnt1 + cnt2))
                #cnt0 = len(np.where(msk >= 0)[0])
                #cnt1 = len(np.where(msk == 0)[0])
                #cnt2 = len(np.where(msk == 1)[0])
                #print("mask : all:{} fg:{} bg:{}".format(cnt0, cnt2, cnt1))
                #assert(cnt0 == (cnt1 + cnt2))
                self.sample_set.append((name,img,msk,alpha))
        print('\t--samples: {}'.format(len(self.sample_set)))

    def __getitem__(self,index):
        name, img, msk, alpha = self.sample_set[index]
        # resize and flip
        if self.transform:
            new_img, new_msk, new_alpha, info=self.transform(img, msk, alpha, name)
        # to tensor
        toTensor = transforms.ToTensor()
        new_img = toTensor(new_img)
        # normalize
        new_img = self.normalize(new_img)
        fg_msk = np.array(new_msk)
        bg_msk = 1 - fg_msk
        #cv2.imwrite('fg_{}.jpg'.format(name), fg_msk*255)
        msk = torch.Tensor(np.stack([bg_msk, fg_msk]))
        new_alpha = torch.Tensor(new_alpha[np.newaxis, :, :]) / 255.
        return new_img, msk, new_alpha, info
    
    def __len__(self):
        return len(self.sample_set)
