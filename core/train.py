from __future__ import print_function
import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import bgs_model
from data import MatTransform, MatDataset
from torchvision import transforms
import time
import os


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--size', type=int, required=True, help="size of input image")
parser.add_argument('--trainList', type=str, required=True, help="train image list")
parser.add_argument('--testList', type=str, required=True, help="test image list")
parser.add_argument('--imgDir', type=str, required=True, help="directory of image")
parser.add_argument('--mskDir', type=str, required=True, help="directory of mask")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--step', type=int, default=10, help='epoch of learning decay')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--resume', type=str, help="checkpoint that model resume from")
parser.add_argument('--pretrain', type=str, help="checkpoint that model pretrain from")
args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cuda = args.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

print('===> Loading datasets')

# [-1,1]
Normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
train_transform = MatTransform(args.size, flip=True)
test_transform = MatTransform(args.size, flip=False)

train_set = MatDataset(args.trainList, args.imgDir, args.mskDir, normalize=Normalize, transform=train_transform)
test_set = MatDataset(args.testList, args.imgDir, args.mskDir, normalize=Normalize, transform=test_transform)

training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.testBatchSize, shuffle=False)

print('===> Building model')


model = bgs_model.MattNet()

start_epoch = 1
if args.pretrain and os.path.isfile(args.pretrain):
    print("=> loading pretrain '{}'".format(args.pretrain))
    ckpt = torch.load(args.pretrain)
    model.load_state_dict(ckpt['state_dict'],strict=False)
    print("=> loaded pretrain '{}' (epoch {})".format(args.pretrain, ckpt['epoch']))

if args.resume and os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict'],strict=True)
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))
    
#criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()
#alpha_criterion = nn.MSELoss()

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.99, weight_decay=0.0005)
#optimizer = optim.Adam(model.parameters(), lr = args.lr)

def adjust_learning_rate(opt, epoch):
    if epoch >= args.step:
        lr = args.lr * 0.1
        for param_group in opt.param_groups:
            param_group['lr'] = lr

def train(epoch):
    t0 = time.time()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()
        adjust_learning_rate(optimizer, epoch)
        optimizer.zero_grad()
        #seg, alpha = model(input)
        seg = model(input)

        loss = criterion(seg, target)
        N,C,H,W = target.shape
        #alpha_target = target[:,1,:,:].view(N,1,H,W)
        #loss = alpha_criterion(alpha, alpha_target)
        loss.backward()
        optimizer.step()
        t1 = time.time()
        num_iter = len(training_data_loader)
        speed = (t1 - t0) / iteration
        expt = speed * (num_iter * (args.nEpochs - epoch + 1) - iteration)
        exp_hour = int(expt / 3600)
        exp_min = int((expt % 3600) / 60)
        exp_sec = int(expt % 60)

        print("===> Epoch[{}/{}]({}/{}) Lr: {:.8f} Loss: {:.5f} Speed: {:.5f} s/iter Exa(H:M:S): {:0>2}:{:0>2}:{:0>2} ".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], loss.data[0], speed, exp_hour, exp_min, exp_sec))

def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / max(mse.data[0], 1e-10))
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model/cuda_sigmoid_sgd_lr0.01_512/model_epoch_{}.pth".format(epoch)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
    }, model_out_path )
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(start_epoch, args.nEpochs + 1):
    train(epoch)
    if epoch % 50 == 0:
        checkpoint(epoch)
