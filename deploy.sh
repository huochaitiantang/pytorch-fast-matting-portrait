#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-fast-matting-portrait

python core/deploy.py \
	--size=128 \
	--testList=$ROOT/list/testlist.txt \
	--imgDir=$ROOT/data/images_data_crop \
	--savePath=$ROOT/result/test/cuda_sigmoid_sgd_lr0.001_128_batch256_e3000/ \
	--resume=$ROOT/model/cuda_sigmoid_sgd_lr0.001_128_batch256_e3000/ckpt_e3000.pth \
	--cuda
