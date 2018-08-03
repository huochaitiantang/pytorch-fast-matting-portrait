#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-fast-matting-portrait

python core/seg2guidedfilter.py \
	--imgDir=$ROOT/data/images_data_crop \
	--predDir=$ROOT/result/test/deeplabv3+ \
	--saveDir=$ROOT/result/test/deeplabv3+_guidedfilter \
	--testList=$ROOT/list/testlist.txt
