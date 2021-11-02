#!/usr/bin/env python3
# coding: utf-8
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import random

from pyconvresnet_DDGL_se_norelu3 import ddgl
# from DAMDNet import DAMDNet_v1
# from mobilenet_v1 import mobilenet_1
import time
import numpy as np
# from pyconvresnet import pynet
from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw import calc_nme as calc_nme_alfw
from benchmark_aflw import ana as ana_aflw

from utils.ddfa import ToTensorGjz, NormalizeGjz, DDFATestDataset, reconstruct_vertex
import argparse


def extract_param_2000(checkpoint_fp, root='', filelists=None, arch='resnet50', num_classes=62, device_ids=[1],
                       batch_size=128, num_workers=4):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    torch.cuda.set_device(device_ids[0])
    model = ddgl()
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # print(model)
    model.load_state_dict(checkpoint)

    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):
            # output, output_medium, output_fine, concat_out, attention_out = model(inputs, input1, input2)
            output = model(inputs)
            # output, output_fine, concat_out = model(inputs, input1, input2)
            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()

                outputs.append(param_prediction)
        outputs = np.array(outputs, dtype=np.float32)

    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs


def extract_param(checkpoint_fp, root='', filelists=None, arch='resnet50', num_classes=62, device_ids=[1],
                  batch_size=128, num_workers=4):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    torch.cuda.set_device(device_ids[0])
    model = ddgl()
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(checkpoint)

    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):

            # output, output_medium, output_fine, concat_out, attention_out = model(inputs, input1, input2)
            output = model(inputs)
            # output, output_fine, concat_out = model(inputs, input1, input2)
            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()

                outputs.append(param_prediction)
        outputs = np.array(outputs, dtype=np.float32)

    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs


def _benchmark_aflw(outputs):
    return ana_aflw(calc_nme_alfw(outputs))


def _benchmark_aflw2000(outputs):
    return ana_alfw2000(calc_nme_alfw2000(outputs))


def benchmark_alfw_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw(outputs)


def benchmark_aflw2000_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw2000(outputs)


def benchmark_pipeline(arch, checkpoint_fp):
    device_ids = [1]

    def aflw():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='test.data/AFLW_GT_crop',
            filelists='test.data/AFLW_GT_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=128)

        benchmark_alfw_params(params)

    def aflw2000():
        params = extract_param_2000(
            checkpoint_fp=checkpoint_fp,
            root='test.data/AFLW2000-3D_crop',
            filelists='test.data/AFLW2000-3D_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=128)

        benchmark_aflw2000_params(params)

    aflw2000()
    aflw()


def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('--arch', default='resnet34', type=str)
    parser.add_argument('-c', '--checkpoint-fp',
                        default='/data1/lab105/zhouzhiyuan/models/phase1_wpdc_checkpoint_epoch_37.pth.tar',
                        type=str)
    args = parser.parse_args()

    benchmark_pipeline(args.arch, args.checkpoint_fp)


if __name__ == '__main__':
    main()



# def main():
#     parser = argparse.ArgumentParser(description='3DDFA Benchmark')
#     parser.add_argument('--arch', default='', type=str)
#
# parser.add_argument('-c', '--checkpoint-fp',
# default='/data1/zhouzhiyuan/DFA/training/snapshot_replay/phase1_wpdc_checkpoint_epoch_37.pth.tar', type=str) #
# epoch_34: 3.665, 4.931;
# epoch_38: 3.692, 4.924;
#
#     # replay attention
#     # epoch_34: 3.661, 4.886;
#     # epoch_37: 3.623, 4.837;
#     args = parser.parse_args()
#
#     benchmark_pipeline(args.arch, args.checkpoint_fp)
