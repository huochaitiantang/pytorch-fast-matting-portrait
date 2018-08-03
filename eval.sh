#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-fast-matting-portrait

python core/eval.py \
	--testList=$ROOT/list/testlist.txt \
	--maskDir=$ROOT/data/images_mask_jpg \
	--predDir=$ROOT/result/test/cuda_sigmoid_sgd_lr0.001_128_batch256_e3000_guidedfilter

#	--predDir=$ROOT/result/test/cuda_sigmoid_sgd_lr0.001_128_batch256_e3000 
