import scipy.io as sci
import torch
import cv2
import numpy as np

f = open("../backup/list.txt")
names = f.readlines()

def mat2jpg():
    for name in names:
        msk_name = name.strip() + "_mask.mat"
        print("{}".format(msk_name))
        msk = sci.loadmat('images_mask/{}'.format(msk_name))['mask']
        cv2.imwrite("images_mask_jpg/{}_mask.jpg".format(name.strip()), msk * 255)

if __name__ == "__main__":
    mat2jpg()
