import torch
import argparse
import torch.nn as nn
import net
import cv2
import os
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--size', type=int, required=True, help="size of input image")
    parser.add_argument('--testList', type=str, required=True, help="train image list")
    parser.add_argument('--imgDir', type=str, required=True, help="directory of image")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--resume', type=str, required=True, help="checkpoint that model resume from")
    parser.add_argument('--savePath', type=str, required=True, help="checkpoint that model save to")
    args = parser.parse_args()
    print(args)
    return args

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

def main():

    print("===> Loading args")
    args = get_args()

    print("===> Environment init")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
     
    model = net.MattNet()
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['state_dict'], strict=True)

    if args.cuda:
        model = model.cuda()

    Normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    dataset = gen_dataset(args.testList, args.imgDir, args.size, True, Normalize)

    for img, info in dataset:
        print('Inference for {}'.format(info[0]))
        #seg,alpha = model(img)
        if args.cuda:
            img = img.cuda()
        seg = model(img)
        # no need: res = F.sigmoid(res)
        if args.cuda:
            seg_np = seg[0,1,:,:].data.cpu().numpy()
        else:
            seg_np = seg[0,1,:,:].data.numpy()
        origin_h = int(seg_np.shape[0] / info[1])
        origin_w = int(seg_np.shape[1] / info[2])
        seg_np = cv2.resize(seg_np,(origin_w, origin_h),interpolation=cv2.INTER_LINEAR)
    
        #seg_fg = seg_np * 255
        seg_fg = (seg_np >= 0.5).astype(np.float32) * 255
        #seg_fg = (seg_np >= 0.95).astype(np.float32) * 255
        #seg_fg = ((seg_np < 0.95) * (seg_np >= 0.05)).astype(np.float32) * 128 + seg_fg
    
        cv2.imwrite('{}{}.jpg'.format(args.savePath, info[0]), seg_fg)

if __name__ == "__main__":
    main()
