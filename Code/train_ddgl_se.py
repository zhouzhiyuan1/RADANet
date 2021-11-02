#!/usr/bin/env python3
# coding: utf-8
import cv2
import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pyconvresnet_DDGL_se_norelu3 import ddgl
import torch.backends.cudnn as cudnn
from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz

from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
from vdc_loss import VDCLoss

from wpdc_loss_old import WPDCLoss
# from wpdc_loss_ll import WPDCLoss
from wingloss import WINGLoss

# global args (configuration)
args = None
lr = None
arch_choices = ['pyhgnet', 'lamdanet50', 'pyconvresnet50', 'scnet50', 'old_resnet34']


def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')
    parser.add_argument('-j', '--workers', default=6, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--start-epoch', default=22, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-vb', '--val-batch-size', default=64, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='', type=str)
    parser.add_argument('--filelists-train',
                        default='', type=str)
    parser.add_argument('--filelists-val',
                        default='', type=str)
    parser.add_argument('--root', default='')
    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--size-average', default='true', type=str2bool)
    parser.add_argument('--num-classes', default=62, type=int)
    parser.add_argument('--arch', default='', type=str,
                        choices=arch_choices)
    parser.add_argument('--frozen', default='false', type=str2bool)
    parser.add_argument('--milestones', default='15,25,30', type=str)
    parser.add_argument('--task', default='all', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)
    parser.add_argument('--warmup', default=-1, type=int)
    parser.add_argument('--param-fp-train',
                        default='',
                        type=str)
    parser.add_argument('--param-fp-val',
                        default='')
    parser.add_argument('--opt-style', default='resample', type=str)  # resample
    parser.add_argument('--resample-num', default=132, type=int)
    parser.add_argument('--loss', default='wpdc', type=str)

    global args
    args = parser.parse_args()

    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]  # 【0，1】
    args.milestones = [int(m) for m in args.milestones.split(',')]

    snapshot_dir = osp.split(args.snapshot)[0]  # 得到第一个切片要求字符之前的内容
    mkdir(snapshot_dir)


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""  # 【15，25，30】

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')


def jigsaw_generator(inputs, n):
    l = []
    patch_num_list = []
    # local_confidence_list = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 120 // n
    rounds = n ** 2
    jigsaws = inputs.clone()

    for i in range(inputs.size(0)):
        patch_num = random.randint(0, 4)
        patch_num_list.append(patch_num)
        # local_confidence_list.append(patch_num / rounds)

    for j in range(inputs.size(0)):
        random.shuffle(l)
        for i in range(patch_num_list[j]):
            x, y = l[i]
            temp = jigsaws[j, :, 0:block_size, 0:block_size].clone()
            jigsaws[j, :, 0:block_size, 0:block_size] = jigsaws[j, :, x * block_size:(x + 1) * block_size,
                                                        y * block_size:(y + 1) * block_size].clone()
            jigsaws[j, :, x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def un_normal(tensor, mean=127.5, std=128):
    tensor.mul_(std).add_(mean)
    return tensor


def train(train_loader, model, criterion, criterion1,criterion2, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    model.train()
    print (model)

    end = time.time()
    # loader is batch style
    # for i, (input, target) in enumerate(train_loader):
    for i, (input, target) in enumerate(train_loader):
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        temp = target
        target = torch.cat((target, temp), dim=0)
        input1 = jigsaw_generator(input,4)
        input = torch.cat((input, input1), dim=0)
        out = model(input)


        # param_p,param_s,param_e = model(input, input1)

        data_time.update(time.time() - end)

        if args.loss.lower() == 'vdc':
            loss = criterion(out, target)
        elif args.loss.lower() == 'wpdc':

            loss1 = criterion(out, target)
            loss2 = criterion1(out, target)

            loss_out = 0.5 * loss1+loss2




        elif args.loss.lower() == 'pdc':
            loss = criterion(out, target)
        else:
            raise Exception(f'Unknown loss {args.loss}')
        # print("input.size(0)*****",input.size(0))

        losses.update(loss_out.item(), input.size(0))
        optimizer.zero_grad()
        loss_out.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log
        if i % args.print_freq == 0:
            logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                         f'LR: {lr:8f}\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         f'Loss {losses.val:.4f} ({losses.avg:.4f})'
                         )
            # f'Loss_paf {losses3.val:.4f} ({losses3.avg:.4f})')


def validate(val_loader, model, criterion, criterion1, epoch):
    model.eval()
    end = time.time()
    with torch.no_grad():
        losses = []
        #
        # losses1 = []
        # losses2 = []
        # losses3 = []

        for i, (input, target) in enumerate(val_loader):
            # compute output
            target.requires_grad = False
            target = target.cuda(non_blocking=True)

            img_r, img_abcd_r, img_all_r, concate, out = model(input)

            # loss1 = criterion(img_r, target)
            # loss2 = criterion1(img_r, target)
            #
            # loss_img_r = 0.5 * loss1 + loss2
            #
            # loss3 = criterion(img_abcd_r, target)
            # loss4 = criterion1(img_abcd_r, target)
            #
            # loss_img_abcd_r = 0.5 * loss3 + loss4

            loss5 = criterion(img_all_r, target)
            loss6 = criterion1(img_all_r, target)

            loss_img_all_r = 0.5 * loss5 + loss6

            loss7 = criterion(out, target)
            loss8 = criterion1(out, target)

            loss_img_out = 0.5 * loss7 + loss8

            loss9 = criterion(concate, target)
            loss10 = criterion1(concate, target)

            loss_img_concate = 0.5 * loss9 + loss10

            # loss = 0.5 * loss_img_r + 0.5 * loss_img_abcd_r + loss_img_all_r + loss_img_out + loss_img_concate * 2
            loss = loss_img_all_r + loss_img_out + loss_img_concate * 2

        # losses.update(loss.item())
        #
        # losses1.update(loss_img.item())
        # losses2.update(loss_x.item())
        # losses3.update(loss_concate.item())

        elapse = time.time() - end
        # loss = np.mean(loss)
        # loss1 = np.mean(losses1)
        # loss2 = np.mean(losses2)
        # loss3 = np.mean(losses3)
        loss = loss.mean()
        logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
                     f'Loss {loss:.4f}\t'
                     # f'Loss_img {loss1:.4f}\t'
                     # #      f'Loss_p {loss_s:.4f}\t'
                     # #      f'Loss_p {loss_e:.4f}\t'
                     # f'Loss_x {loss2:.4f}\t'
                     #     f'Loss_concate {loss2:.4f}\t'
                     # # f'Loss {loss3:.4f}\t'
                     f'Time {elapse:.3f}'
                     )


def main():
    parse_args()  # parse global argsl

    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )
    print("*********py50+wwpdc**********")
    print_args(args)  # print args

    # step1: define the model structure
    # model = getattr(PYNet, args.arch)()
    model = ddgl()

    nparameters = sum(p.numel() for p in model.parameters())
    # print(model)
    print('Total number of parameters: %d' % nparameters)

    torch.cuda.set_device(args.devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`

    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()  # -> GPU

    # step2: optimization: loss and optimization method
    # criterion = nn.MSELoss(size_average=args.size_average).cuda()
    if args.loss.lower() == 'wpdc':
        print(args.opt_style)
        # criterion = WPDCLoss(opt_style=args.opt_style).cuda()
        criterion = WPDCLoss(opt_style=args.opt_style).cuda()
        criterion1 = WINGLoss(opt_style=args.opt_style).cuda()
        criterion2 = VDCLoss().cuda()
        logging.info('Use WPDC Loss')
    elif args.loss.lower() == 'vdc':
        criterion = VDCLoss(opt_style=args.opt_style).cuda()
        logging.info('Use VDC Loss')
    elif args.loss.lower() == 'pdc':
        criterion = nn.MSELoss(size_average=args.size_average).cuda()
        logging.info('Use PDC loss')
    else:
        raise Exception(f'Unknown Loss {args.loss}')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            # checkpoint = torch.load(args.resume)['state_dict']
            model.load_state_dict(checkpoint)
        else:
            logging.info(f'=> no checkpoint found at {args.resume}')

    # step3: data
    normalize = NormalizeGjz(mean=127.5, std=128)

    train_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )
    val_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_val,
        param_fp=args.param_fp_val,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True)

    # step4: run
    cudnn.benchmark = True
    if args.test_initial:
        logging.info('Testing from initial')
        validate(val_loader, model, criterion, criterion1, args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.milestones)

        # train for one epoch
        train(train_loader, model, criterion, criterion1, criterion2,optimizer, epoch)
        filename = f'{args.snapshot}_checkpoint_epoch_{epoch}.pth.tar'
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict()
            },
            filename
        )

        # validate(val_loader, model, criterion, criterion1, epoch)


if __name__ == '__main__':
    main()
