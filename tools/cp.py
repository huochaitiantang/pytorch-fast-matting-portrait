import os
import shutil

f = open('/home/liuliang/Desktop/alpha_pictures/kisspng/ok.txt')
ns = f.readlines()

i = 1
for n in ns:
  idd = n[:5]
  img_old = '/home/liuliang/Desktop/alpha_pictures/kisspng/pictures/{}.png'.format(idd)
  img_new = '/home/liuliang/Desktop/alpha_pictures/kisspng/ok_pictures/{}.png'.format(idd)
  rgb_old = '/home/liuliang/Desktop/alpha_pictures/kisspng/rgb/{}_rgb.png'.format(idd)
  rgb_new = '/home/liuliang/Desktop/alpha_pictures/kisspng/ok_rgb/{}.png'.format(idd)
  alpha_old = '/home/liuliang/Desktop/alpha_pictures/kisspng/alpha/{}_alpha.png'.format(idd)
  alpha_new = '/home/liuliang/Desktop/alpha_pictures/kisspng/ok_alpha/{}.png'.format(idd)
  #os.rename(n, nn)
  shutil.copyfile(img_old, img_new)
  shutil.copyfile(rgb_old, rgb_new)
  shutil.copyfile(alpha_old, alpha_new)
  i = i + 1
