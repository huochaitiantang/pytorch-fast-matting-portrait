import random
import numpy as np
import shutil

base = '/home/liuliang/Desktop/alpha_pictures/kisspng'

f = open('{}/crop.txt'.format(base))
img_ids = f.readlines()
cnt = len(img_ids)

srcAlaDir = '{}/train/alpha'.format(base)
desAlaDir = '{}/test/alpha'.format(base)
srcRgbDir = '{}/train/rgb'.format(base)
desRgbDir = '{}/test/rgb'.format(base)

test_ind = np.random.choice(np.arange(cnt), size=100, replace=False)

print(test_ind)

for ind in test_ind:
    img_id = img_ids[ind][:5]
    src_ala_name = "{}/{}.png".format(srcAlaDir, img_id)
    des_ala_name = "{}/{}.png".format(desAlaDir, img_id)
    src_rgb_name = "{}/{}.png".format(srcRgbDir, img_id)
    des_rgb_name = "{}/{}.png".format(desRgbDir, img_id)
    shutil.move(src_ala_name, des_ala_name)
    shutil.move(src_rgb_name, des_rgb_name)
    #print(img_id)
