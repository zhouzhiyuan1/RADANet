""" PyConv networks for image recognition as presented in our paper:
    Duta et al. "Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"
    https://arxiv.org/pdf/2006.11538.pdf
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import attention
from attentiontransformer import AttentionTransformer

try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'pretrained')

__all__ = ['PyConvResNet', 'pyconvresnet34', 'pyconvresnet50']

model_urls = {
    'pyconvresnet50': 'https://drive.google.com/uc?export=download&id=128iMzBnHQSPNehgb8nUF5cJyKBIB7do5',
    'pyconvresnet101': 'https://drive.google.com/uc?export=download&id=1fn0eKdtGG7HA30O5SJ1XrmGR_FsQxTb1',
    'pyconvresnet152': 'https://drive.google.com/uc?export=download&id=1zR6HOTaHB0t15n6Nh12adX86AhBMo46m',
}


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PyConv2d(nn.Module):
    """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``

    Example::

        >>> # PyConv with two pyramid levels, kernels: 3x3, 5x5
        >>> m = PyConv2d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)

        >>> # PyConv with three pyramid levels, kernels: 3x3, 5x5, 7x7
        >>> m = PyConv2d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
    """

    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3] // 2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class PyConvBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = get_pyconv(planes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PyConvBasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBasicBlock1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = get_pyconv(inplanes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = get_pyconv(planes, planes, pyconv_kernels=pyconv_kernels, stride=1,
                                pyconv_groups=pyconv_groups)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PyConvBasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBasicBlock2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = get_pyconv(inplanes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes * self.expansion)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class PyConvResNet(nn.Module):

    def __init__(self, block, layers, num_classes=62, zero_init_residual=False, norm_layer=None, dropout_prob0=0.3,
                 x2_bool=0):
        super(PyConvResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.x2_bool = x2_bool
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        # self.frelu = FReLU(64)
        self.transformer1 = AttentionTransformer()
        self.transformer2 = AttentionTransformer()
        # self.nonloc = NLBlockND(in_channels=256, mode='embedded', dimension=2, bn_layer=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, norm_layer=norm_layer,
                                       pyconv_kernels=[3, 5, 7, 9], pyconv_groups=[1, 4, 8, 16])
        self.replay_attention1 = attention_replay(256, sr_guide=True, layer_num=1)
        #self.seLayer1 = SELayer(256)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer,
                                       pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        self.replay_attention2 = attention_replay(512, sr_guide=True, layer_num=2)
        #self.seLayer2 = SELayer(512)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer,
                                       pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        self.seLayer3 = SELayer(1024)
        #self.seLayer3 = attention_replay(1024, sr_guide=True, layer_num=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer,
                                       pyconv_kernels=[3], pyconv_groups=[1])
        #self.seLayer4 = attention_replay(2048, sr_guide=True, layer_num=4)
        self.seLayer4 = SELayer(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dropout_prob0 > 0.0:
            self.dp = nn.Dropout(dropout_prob0, inplace=True)
            print("Using Dropout with the prob to set to 0 of: ", dropout_prob0)
        else:
            self.dp = None
        # self.x2_fusion_conv = nn.Sequential(
        #     BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
        #     BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True),
        #     attention.SELayer(1024, 16),
        # )
        # self.fc = nn.Linear(512 * block.expansion, 512)
        self.fc1 = nn.Linear(512 * block.expansion, 512)
        self.fc2 = nn.Linear(512 * block.expansion, 512)
        self.fc3 = nn.Linear(512 * block.expansion, 512)
        self.fc11 = nn.Linear(512, 12)
        self.fc21 = nn.Linear(512, 40)
        self.fc31 = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PyConvBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, pyconv_kernels=[3], pyconv_groups=[1]):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 and self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer,
                            pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        #x = self.seLayer1(x)
        x = self.replay_attention1(x)
        #x = self.seLayer1(x)
        x_layer2 = self.layer2(x)
        #x_layer2 = self.seLayer2(x_layer2)
        x_layer2 = self.replay_attention2(x_layer2)

        x_layer2 = self.transformer1(x_layer2)
        x_layer2 = self.transformer2(x_layer2)
        x_layer3 = self.layer3(x_layer2)
        x_layer3 = self.seLayer3(x_layer3)

        x = self.layer4(x_layer3)
        x = self.seLayer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.dp is not None:
            x = self.dp(x)

        # x = self.fc(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)

        x1 = self.fc11(x1)
        x2 = self.fc21(x2)
        x3 = self.fc31(x3)

        x = torch.cat((x1, x2, x3), 1)
        return x


class DDGLNet(nn.Module):

    def __init__(self, **kwargs):
        super(DDGLNet, self).__init__()

        # self.encoder_fine = resnet34_module(pretrained=False)

        self.encoder_coarse = pyconvresnet50(pretrained=False)

    def forward(self, x):
        output = self.encoder_coarse(x)

        return output


def pyconvresnet34(pretrained=False, **kwargs):
    """Constructs a PyConvResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = PyConvResNet(PyConvBasicBlock1, [3, 4, 6, 3], **kwargs) #params=20.44M GFLOPs 3.09
    model = PyConvResNet(PyConvBasicBlock2, [3, 4, 6, 3], **kwargs)  # params=11.09M GFLOPs 1.75

    return model


def pyconvresnet50(pretrained=False, **kwargs):
    """Constructs a PyConvResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyConvResNet(PyConvBlock, [3, 4, 6, 3], **kwargs)

    return model


def ddgl(**kwargs):
    model = DDGLNet(**kwargs)
    return model


# from generate_image import jigsaw_generator
#
# def demo():
#     img = torch.randn(1, 3, 120, 120)
#     img1 = jigsaw_generator(img,2)
#     net = DDGLNet()
#     output, output_medium ,concat_out= net(img,img1)
#     print(output.size())
#
# demo()

class attention_replay(nn.Module):
    def __init__(self, out_channels, kernel_size=3, no_local=False, sr_guide=False, layer_num=1):
        super(attention_replay, self).__init__()
        self.is_sr_guide = sr_guide
        self.is_no_local = no_local
        if sr_guide:
            self.attention = CBAM(out_channels, reduction_ratio=16, no_spatial=False)
            self.guider = attention_replay_guider(out_channels=out_channels, kernel_size=kernel_size,
                                                  layer_num=layer_num)
        # if no_local:
        #     self.non_local = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels)
        #     self.non_local_guider = attention_replay_guider(out_channels=out_channels, kernel_size=kernel_size,
        #                                                     layer_num=layer_num)
        # self.BN_AC = nn.Sequential(
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        if self.is_sr_guide:
            attention_out = self.attention(x)
            # out = attention_out
            out = self.guider(attention_out, x)
        else:
            out = x
        # if self.is_no_local:
        #     out = self.BN_AC(out)
        #     nonlocal_out = self.non_local(out)
        #     out = self.non_local_guider(nonlocal_out)
        return out


class attention_replay_guider(nn.Module):
    def __init__(self, out_channels, kernel_size=3, layer_num=1):
        super(attention_replay_guider, self).__init__()
        # ResNet Example
        if layer_num <= 2:
            self.estimate_weight = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1,
                          kernel_size=kernel_size, padding=(kernel_size - 1) // 2, dilation=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.estimate_weight = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1,
                          kernel_size=kernel_size, padding=(kernel_size - 1) // 2, dilation=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.fitting = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dynamic_fusion = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 3, out_channels=out_channels, stride=1,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # FCN Example
        # if layer_num <= 5:
        #     self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # else:
        #     self.pooling = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # self.estimate_weight = nn.Sequential(
        #     # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1,
        #               kernel_size=kernel_size, padding=kernel_size - 1, padding_mode='circular', dilation=1),
        #     nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.5),
        # )
        # self.fitting = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1,
        #               kernel_size=kernel_size, padding=kernel_size - 1, padding_mode='circular', dilation=1),
        #     nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.5),
        # )
        # self.dynamic_fusion = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels * 3, out_channels=out_channels, stride=1,
        #               kernel_size=kernel_size, padding=kernel_size - 1, padding_mode='circular', dilation=1),
        #     nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.5),
        # )

    def forward(self, guider, x):
        identity = x
        attn = torch.sigmoid(F.interpolate(self.estimate_weight(guider), identity.size()[2:]))
        out = torch.mul(self.fitting(x), attn)
        out = self.dynamic_fusion(torch.cat((out, guider, identity), dim=1))
        return out


class SelfTrans(nn.Module):
    def __init__(self, n_head, n_mix, d_model, d_k, d_v,
                 norm_layer=nn.BatchNorm2d, kq_transform='conv', value_transform='conv',
                 pooling=True, concat=False, dropout=0.1):
        super(SelfTrans, self).__init__()

        self.n_head = n_head
        self.n_mix = n_mix
        self.d_k = d_k
        self.d_v = d_v

        self.pooling = pooling
        self.concat = concat

        if self.pooling:
            self.pool = nn.AvgPool2d(3, 2, 1, count_include_pad=False)
        if kq_transform == 'conv':
            self.conv_qs = nn.Conv2d(d_model, n_head * d_k, 1)
            nn.init.normal_(self.conv_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        elif kq_transform == 'ffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head * d_k, 3, padding=1, bias=False),
                norm_layer(n_head * d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head * d_k, n_head * d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        elif kq_transform == 'dffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head * d_k, 3, padding=4, dilation=4, bias=False),
                norm_layer(n_head * d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head * d_k, n_head * d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        else:
            raise NotImplemented

        self.conv_ks = self.conv_qs
        if value_transform == 'conv':
            self.conv_vs = nn.Conv2d(d_model, n_head * d_v, 1)
        else:
            raise NotImplemented

        nn.init.normal_(self.conv_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = MixtureOfSoftMax(n_mix=n_mix, d_k=d_k)

        self.conv = nn.Conv2d(n_head * d_v, d_model, 1, bias=False)
        self.norm_layer = norm_layer(d_model)

    def forward(self, x):
        residual = x
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_, c_, h_, w_ = x.size()
        if self.pooling:
            qt = self.conv_ks(x).view(b_ * n_head, d_k, h_ * w_)
            kt = self.conv_ks(self.pool(x)).view(b_ * n_head, d_k, h_ * w_ // 4)
            vt = self.conv_vs(self.pool(x)).view(b_ * n_head, d_v, h_ * w_ // 4)
        else:
            kt = self.conv_ks(x).view(b_ * n_head, d_k, h_ * w_)
            qt = kt
            vt = self.conv_vs(x).view(b_ * n_head, d_v, h_ * w_)

        output, attn = self.attention(qt, kt, vt)

        output = output.transpose(1, 2).contiguous().view(b_, n_head * d_v, h_, w_)

        output = self.conv(output)
        if self.concat:
            output = torch.cat((self.norm_layer(output), residual), 1)
        else:
            output = self.norm_layer(output) + residual
        return output


class MixtureOfSoftMax(nn.Module):
    """"https://arxiv.org/pdf/1711.03953.pdf"""

    def __init__(self, n_mix, d_k, attn_dropout=0.1):
        super(MixtureOfSoftMax, self).__init__()
        self.temperature = np.power(d_k, 0.5)
        self.n_mix = n_mix
        self.att_drop = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.d_k = d_k
        if n_mix > 1:
            self.weight = nn.Parameter(torch.Tensor(n_mix, d_k))
            std = np.power(n_mix, -0.5)
            self.weight.data.uniform_(-std, std)

    def forward(self, qt, kt, vt):
        B, d_k, N = qt.size()
        m = self.n_mix
        assert d_k == self.d_k
        d = d_k // m
        if m > 1:
            bar_qt = torch.mean(qt, 2, True)
            pi = self.softmax1(torch.matmul(self.weight, bar_qt)).view(B * m, 1, 1)

        q = qt.view(B * m, d, N).transpose(1, 2)
        N2 = kt.size(2)
        kt = kt.view(B * m, d, N2)
        v = vt.transpose(1, 2)
        attn = torch.bmm(q, kt)
        attn = attn / self.temperature
        attn = self.softmax2(attn)
        attn = self.dropout(attn)
        if m > 1:
            attn = (attn * pi).view(B, m, N, N2).sum(1)
        output = torch.bmm(attn, v)
        return output, attn


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
