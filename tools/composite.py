import os
import cv2
import numpy as np
import shutil

base = '/home/liuliang/Desktop/alpha_pictures/kisspng'

fgDir = '{}/unique_train/rgb'.format(base)
bgDir = '{}/voc_bg'.format(base)
alphaDir = '{}/unique_train/alpha'.format(base)

savImgDir = '{}/train/image'.format(base)
savAlaDir = '{}/train/alpha'.format(base)

fgs = os.listdir(fgDir)
bgs = os.listdir(bgDir)

fg_cnt = len(fgs)
bg_cnt = len(bgs)

bg_composite = 10
bg_inds = np.arange(bg_cnt)

print("Fg: {} Bg: {}".format(fg_cnt, bg_cnt))

for fg in fgs:

    fg_img = cv2.imread('{}/{}'.format(fgDir, fg), cv2.IMREAD_UNCHANGED)
    alpha = cv2.imread('{}/{}'.format(alphaDir, fg), cv2.IMREAD_UNCHANGED)
    alpha = (alpha.astype(np.float32) / 255.)[:,:,np.newaxis]
    alpha = np.concatenate((alpha, alpha, alpha), axis = 2)
    assert(alpha.shape[2] == 3)
    assert(fg_img.shape[2] == 3)
    h, w, c = fg_img.shape

    bg_selects = np.random.choice(bg_inds, size=bg_composite, replace=False)
    for select in bg_selects:
        bg = bgs[select]
        print("{} + {}".format(fg, bg))
        bg_img = cv2.imread('{}/{}'.format(bgDir, bg))[:,:,:3]
        bg_h, bg_w, bg_c = bg_img.shape
        assert(bg_c == 3)
        if bg_h < h or bg_w < w:
            upratio = max(float(h + 1)/bg_h, float(w + 1)/bg_w)
            new_w = int(bg_w * upratio)
            new_h = int(bg_h * upratio)
            bg_img = cv2.resize(bg_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            bg_h, bg_w, bg_c = bg_img.shape
        assert(bg_h >= h) 
        assert(bg_w >= w)
        cropy = np.random.randint(bg_h - h + 1, size = 1)[0] 
        cropx = np.random.randint(bg_w - w + 1, size = 1)[0] 
        crop_bg_img = bg_img[cropy : cropy + h, cropx : cropx + w]
        assert(crop_bg_img.shape == fg_img.shape)
        comp = alpha * fg_img + (1. - alpha) * crop_bg_img
        comp_id = fg[:-4] + '_' + bg[:-4] + '.png'
        cv2.imwrite('{}/{}'.format(savImgDir, comp_id), comp)
        shutil.copyfile('{}/{}'.format(alphaDir, fg), '{}/{}'.format(savAlaDir, comp_id))
