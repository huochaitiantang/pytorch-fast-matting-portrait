import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import net
from data import MatTransform, MatDataset
from torchvision import transforms
import time
import os


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--size', type=int, required=True, help="size of input image")
    parser.add_argument('--trainList', type=str, required=True, help="train image list")
    parser.add_argument('--imgDir', type=str, required=True, help="directory of image")
    parser.add_argument('--mskDir', type=str, required=True, help="directory of mask")
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--step', type=int, default=10, help='epoch of learning decay')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--resume', type=str, help="checkpoint that model resume from")
    parser.add_argument('--pretrain', type=str, help="checkpoint that model pretrain from")
    parser.add_argument('--saveDir', type=str, help="checkpoint that model save to")
    parser.add_argument('--printFreq', type=int, default=10, help="checkpoint that model save to")
    parser.add_argument('--ckptSaveFreq', type=int, default=10, help="checkpoint that model save to")
    args = parser.parse_args()
    print(args)
    return args


def get_dataset(args):
    # [-1,1]
    Normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    train_transform = MatTransform(args.size, flip=True)
    
    train_set = MatDataset(args.trainList, args.imgDir, args.mskDir, normalize=Normalize, transform=train_transform)
    train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)

    return train_loader


def build_model(args):
    model = net.MattNet()
    
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
    
    return start_epoch, model    


def adjust_learning_rate(args, opt, epoch):
    if epoch >= args.step:
        lr = args.lr * 0.1
        for param_group in opt.param_groups:
            param_group['lr'] = lr


def format_second(secs):
    h = int(secs / 3600)
    m = int((secs % 3600) / 60)
    s = int(secs % 60)
    ss = "Exa(h:m:s): {:0>2}:{:0>2}:{:0>2}".format(h,m,s)
    return ss    


# [h,w,3], [h,w,1],[h,w,1] 
def alpha_loss(img, msk_gt, alpha, eps=1e-6):
    L_alpha = torch.sqrt(torch.pow(msk_gt - alpha, 2.) + eps).mean()
    gt_msk_img = torch.cat((msk_gt, msk_gt, msk_gt), 1) * img
    alpha_img = torch.cat((alpha, alpha, alpha), 1) * img
    L_color = torch.sqrt(torch.pow(gt_msk_img - alpha_img, 2.) + eps).mean()
    return L_alpha + L_color


def train(args, model, criterion, optimizer, train_loader, epoch):
    t0 = time.time()
    for iteration, batch in enumerate(train_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            input = input.cuda()
            target = target.cuda()
        adjust_learning_rate(args, optimizer, epoch)
        optimizer.zero_grad()
        seg = model(input)
        #seg, alpha = model(input)

        N,C,H,W = target.shape

        # segmentation block loss
        #loss = criterion(seg, target)
        #print(target[0,:,:,:])
        loss = criterion(seg, target[:,1,:,:].long())

        # feathering block loss
        #alpha_target = target[:,1,:,:].view(N,1,H,W)
        #loss2 = alpha_loss(input, alpha_target, alpha)

        #loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        if iteration % args.printFreq ==  0:
            t1 = time.time()
            num_iter = len(train_loader)
            speed = (t1 - t0) / iteration
            exp_time = format_second(speed * (num_iter * (args.nEpochs - epoch + 1) - iteration))

            print("===> Epoch[{}/{}]({}/{}) Lr: {:.8f} Loss: {:.5f} Speed: {:.5f} s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], loss.data[0], speed, exp_time))

def checkpoint(epoch, save_dir, model):
    model_out_path = "{}/ckpt_e{}.pth".format(save_dir, epoch)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
    }, model_out_path )
    print("Checkpoint saved to {}".format(model_out_path))


def main():

    print("===> Loading args")
    args = get_args()

    print("===> Environment init")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    print('===> Loading datasets')
    train_loader = get_dataset(args)

    print('===> Building model')
    start_epoch, model = build_model(args)

    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.99, weight_decay=0.0005)
    #optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # training
    for epoch in range(start_epoch, args.nEpochs + 1):
        train(args, model, criterion, optimizer, train_loader, epoch)
        if epoch > 0 and epoch % args.ckptSaveFreq == 0:
            checkpoint(epoch, args.saveDir, model)


if __name__ == "__main__":
    main()
