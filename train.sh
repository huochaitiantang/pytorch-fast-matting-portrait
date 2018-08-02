#/bin/bash

python train.py \
	--size=512 \
	--trainList=trainlist.txt \
	--testList=testlist.txt \
	--imgDir=/home/liuliang/Desktop/FastMattingPortrait/data/images_data_crop \
	--mskDir=/home/liuliang/Desktop/FastMattingPortrait/data/images_mask \
	--batchSize=32 \
	--testBatchSize=1 \
	--nEpochs=500 \
	--step=300 \
	--lr=0.01 \
	--threads=4 \
	--cuda
	#--pretrain=model/sigmoid-sgd/model_epoch_500.pth \
	#--pretrain=model/sigmoid-sgd/model_epoch_500.pth \
	#--resume=model/sigmoid-sgd-lr0.01-epoch500/model_epoch_100.pth \
