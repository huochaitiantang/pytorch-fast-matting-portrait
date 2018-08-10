import cv2
import numpy as np

imgDir = '/home/liuliang/Desktop/pytorch-fast-matting-portrait/data/images_data_crop'
AlaDir = '/home/liuliang/Desktop/pytorch-fast-matting-portrait/data/images_knn'

img_id = '00012'

img_name = imgDir + '/' + img_id + '.jpg'
ala_name = AlaDir + '/' + img_id + '.png'

img = cv2.imread(img_name)
ala = cv2.imread(ala_name)[:,:,0]

ala = ala[:, :, np.newaxis]

print("Image:{} Alpha:{}".format(img.shape, ala.shape))

res = np.concatenate((img, ala), axis=2)

print("Res:{}".format(res.shape))

cv2.imwrite('res.png', res)

x = cv2.imread('res.png',cv2.IMREAD_UNCHANGED)

print(x.shape)




