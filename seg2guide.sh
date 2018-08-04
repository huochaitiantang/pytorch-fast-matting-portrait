#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-fast-matting-portrait

python core/seg2guidedfilter.py \
	--imgDir=$ROOT/data/images_data_crop \
	--predDir=$ROOT/result/test/cuda_softmax_sgd_lr0.001_128_batch256_e3000 \
	--saveDir=$ROOT/result/test/cuda_softmax_sgd_lr0.001_128_batch256_e3000_guidedfilter \
	--testList=$ROOT/list/testlist.txt
