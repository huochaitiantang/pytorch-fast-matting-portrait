#/bin/bash

ROOT=/home/liuliang/Desktop/pytorch-fast-matting-portrait

python core/train.py \
	--size=128 \
	--trainList=$ROOT/list/trainlist.txt \
	--imgDir=$ROOT/data/images_data_crop \
	--mskDir=$ROOT/data/images_mask_jpg \
	--saveDir=$ROOT/model/cuda_softmax_sgd_lr0.001_128_batch256_e3000 \
	--batchSize=256 \
	--nEpochs=3000 \
	--step=1500 \
	--lr=0.001 \
	--threads=4 \
	--printFreq=3 \
	--ckptSaveFreq=100 \
	--cuda \
	--resume=model/cuda_softmax_sgd_lr0.001_128_batch256_e3000/ckpt_e400.pth \
        #--pretrain=model/cuda_sigmoid_sgd_lr0.001_128_batch256_e3000_stage2/ckpt_e3000.pth
