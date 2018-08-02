import torch
import torch.nn as nn
import bgs_model
import cv2
import os
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

def gen_transform(size, img, name):
    scale_h = float(size) / img.shape[0]
    scale_w = float(size) / img.shape[1]
    img = cv2.resize(img,(size, size),interpolation=cv2.INTER_LINEAR)
    return img, [name, scale_h, scale_w]

def gen_dataset(namelist, imgdir, size, transform=True, normalize=None):
        sample_set = []
        with open(namelist, 'r') as f:
            names = f.readlines()
        print('namelist:{}'.format(namelist))
        print('names len:{}'.format(len(names)))
        for name in names:
            name = name.strip('\n')
            img_path = imgdir + '/' + name + '.jpg'
            if os.path.exists(img_path):
                #size:[800,600,3] value:0-255 order BGR
                img = cv2.imread(img_path)
                if transform:
                    new_img, info = gen_transform(size, img, name)
                # to tensor
                toTensor = transforms.ToTensor()
                new_img = toTensor(new_img)
                # normalize
                if normalize:
                    new_img = normalize(new_img)
                new_img = new_img.view(1, 3, size, size)

                sample_set.append((new_img, info))
        print('samples len:{}'.format(len(sample_set)))
        return sample_set


model = bgs_model.MattNet()
#ckpt = torch.load('model/backup-sigmoid-sgd/model_epoch_500.pth')
#ckpt = torch.load('model/cuda_sigmoid_sgd_lr0.01_128/model_epoch_500.pth')
#ckpt = torch.load('model/cuda_sigmoid_sgd_lr0.01_512/model_epoch_500.pth')
model.load_state_dict(ckpt['state_dict'], strict=True)

Normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

dataset = gen_dataset('deploy.txt', './img', 512, True, Normalize)
#dataset = gen_dataset('testlist.txt', './data/images_data_crop', 512, True, Normalize)
for img, info in dataset:
    print('Handle for {}'.format(info[0]))
    #seg,alpha = model(img)
    seg = model(img)
    # no need: res = F.sigmoid(res)
    
    seg_np = seg[0,1,:,:].data.numpy()
    origin_h = int(seg_np.shape[0] / info[1])
    origin_w = int(seg_np.shape[1] / info[2])
    seg_np = cv2.resize(seg_np,(origin_w, origin_h),interpolation=cv2.INTER_LINEAR)

    #seg_fg = seg_np * 255
    seg_fg = (seg_np >= 0.5).astype(np.float32) * 255
    #seg_fg = (seg_np >= 0.95).astype(np.float32) * 255
    #seg_fg = ((seg_np < 0.95) * (seg_np >= 0.05)).astype(np.float32) * 128 + seg_fg

    cv2.imwrite('img/cuda512_seg_fg_{}.jpg'.format(info[0]), seg_fg)
