#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-fast-matting-portrait

python core/deploy.py \
	--size=128 \
	--testList=$ROOT/list/testlist.txt \
	--imgDir=$ROOT/data/images_data_crop \
	--savePath=$ROOT/result/test/tmp1/ \
	--resume=$ROOT/model/cuda_softmax_sgd_lr0.001_128_batch256_e3000_stage2_lr0.001/ckpt_e3000.pth \
	--cuda
	#--savePath=$ROOT/result/test/cuda_sigmoid_sgd_lr0.001_128_batch256_e3000_stage3/ \
	#--testList=$ROOT/list/deploy.txt \
	#--imgDir=$ROOT/result/deploy \
