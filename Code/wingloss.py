import torch

import torch.nn as nn

from utils.io import _load, _numpy_to_cuda, _numpy_to_tensor, _load_gpu

import math

from utils.params import *

_to_tensor = _numpy_to_cuda  # gpu


def _parse_param_batch(param):
    """Work for both numpy and tensor"""

    N = param.shape[0]

    p_ = param[:, :12].view(N, 3, -1)

    p = p_[:, :, :3]

    offset = p_[:, :, -1].view(N, 3, 1)

    alpha_shp = param[:, 12:52].view(N, -1, 1)

    alpha_exp = param[:, 52:].view(N, -1, 1)

    return p, offset, alpha_shp, alpha_exp


class WINGLoss(nn.Module):

    def __init__(self, opt_style='resample', w=10, epsilon=2.0):
        super(WINGLoss, self).__init__()

        self.w = w

        self.epsilon = epsilon

        self.C = self.w - self.w * np.log(1 + self.w / self.epsilon)

        self.opt_style = opt_style

        self.u = _to_tensor(u)

        self.param_mean = _to_tensor(param_mean)

        self.param_std = _to_tensor(param_std)

        self.w_shp = _to_tensor(w_shp)

        self.w_exp = _to_tensor(w_exp)

        self.keypoints = _to_tensor(keypoints)

        self.u_base = self.u[self.keypoints]

        self.w_shp_base = self.w_shp[self.keypoints]

        self.w_exp_base = self.w_exp[self.keypoints]

        self.w_shp_length = self.w_shp.shape[0] // 3

        self.opt_style = opt_style

    def reconstruct_and_parse(self, input, target):
        # reconstruct

        param = input * self.param_std + self.param_mean

        param_gt = target * self.param_std + self.param_mean

        # parse param

        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)

        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)

        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def forward_all(self, input, target):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        N = input.shape[0]
        offset[:, -1] = offsetg[:, -1]
        gt_vertex = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        # print("*************")
        # print("gt_vertex", gt_vertex.shape)
        vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset


        return vertex,gt_vertex


    def forward_resample(self, input, target, resample_num=132):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) = self.reconstruct_and_parse(input, target)

        # resample index

        index = torch.randperm(self.w_shp_length)[:resample_num].reshape(-1, 1)

        keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1).cuda()

        keypoints_mix = torch.cat((self.keypoints, keypoints_resample))

        w_shp_base = self.w_shp[keypoints_mix]

        u_base = self.u[keypoints_mix]

        w_exp_base = self.w_exp[keypoints_mix]

        offset[:, -1] = offsetg[:, -1]

        N = input.shape[0]

        gt_vertex = pg @ (u_base + w_shp_base @ alpha_shpg + w_exp_base @ alpha_expg) .view(N, -1, 3).permute(0, 2, 1) + offsetg

        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).view(N, -1, 3).permute(0, 2, 1) + offset
        # print("*************")
        # print("gt_vertex",gt_vertex.shape)


        return vertex, gt_vertex


    def forward(self, input_, target_):
        if self.opt_style == 'resample':

            input, target = self.forward_resample(input_, target_)

            x = input - target

            absolute_x = torch.abs(x)

            loss = torch.mean(torch.where(absolute_x < self.w,

                                          self.w * torch.log(1 + absolute_x / self.epsilon),

                                          absolute_x - self.C))
            return loss




if __name__ == '__main__':
    pass