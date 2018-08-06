#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-fast-matting-portrait

python gen_trimap.py \
	--mskDir=$ROOT/data/images_mask_png \
	--saveDir=$ROOT/data/images_trimap_dilated \
	--list=$ROOT/backup/list.txt
