import cv2
import numpy as np

imgDir = '/home/liuliang/Desktop/alpha_pictures/kisspng/pictures'
savDir = '/home/liuliang/Desktop/alpha_pictures/kisspng/alpha'
sav2Dir = '/home/liuliang/Desktop/alpha_pictures/kisspng/rgb'

lists = '/home/liuliang/Desktop/alpha_pictures/kisspng/list.txt'
f = open(lists)
img_ids = f.readlines()

for img_id in img_ids:
    img_id = img_id.strip()[:-4]

    print(img_id)

    img_name = imgDir + '/' + img_id + '.png'
    sav_name = savDir + '/' + img_id + '_alpha.png'
    sav2_name = sav2Dir + '/' + img_id + '_rgb.png'
    
    #print(img_name)
    #print(sav_name)
    
    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        cv2.imwrite(sav_name, img[:,:,3])
        cv2.imwrite(sav2_name, img[:,:,:3])
    else:
        print("warning: shape: {}".format(img.shape))
