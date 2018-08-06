import scipy.io as sci
import torch
import cv2
import numpy as np

f = open("../backup/list.txt")
names = f.readlines()

# jpg will lose some information

def mat2png():
    for name in names:
        msk_name = name.strip() + "_mask.mat"
        print("{}".format(msk_name))
        msk = sci.loadmat('../data/images_mask/{}'.format(msk_name))['mask']
        cv2.imwrite("../data/images_mask_png/{}.png".format(name.strip()), msk * 255)

if __name__ == "__main__":
    mat2png()
