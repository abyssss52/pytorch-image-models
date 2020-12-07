#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File         :   MobileNeXt.py
@Modify Time  :   2020/12/7 上午9:21
@Author       :   night
@Version      :   1.0
@License      :   (C)Copyright 2019-2020, Real2tech
@Desciption   :   None

'''

"""
Creates a MobileNeXt Model as defined in:
Zhou Daquan, Qibin Hou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan
Rethinking Bottleneck Structure for Efficient Mobile Network Design
arXiv preprint arXiv:2007.02269.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

import torch.nn as nn
import torch.nn.functional as F
import math

from .registry import register_model

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ['MobileNeXt']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'layer0.conv1', 'classifier': 'last_linear',
        **kwargs
    }


default_cfgs = {
    'mobilenext_100': _cfg(url=''),
}

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class SandGlass(nn.Module):
    def __init__(self, inp, oup, stride, reduction_ratio):
        super(SandGlass, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp // reduction_ratio)
        self.identity = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
            # dw-linear
            nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)



# 原为适配Grad_CAM,该层为独立出Final_layer
class FinalSandGlassLayer(nn.Module):
    def __init__(self, inp, oup, stride, reduction_ratio):
        super(FinalSandGlassLayer, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp // reduction_ratio)
        self.identity = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
            # # dw-linear
            # nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FinalLayer(nn.Module):
    def __init__(self, oup, stride):
        super(FinalLayer, self).__init__()

        self.conv = nn.Sequential(
            # dw-linear
            nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.conv(x)


class MobileNeXt(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., in_chans = 3, drop_rate = 0.0, act_layer = nn.ReLU6,
        SE = False, groups = 8, norm_layer = nn.BatchNorm2d, norm_kwargs = None,
        global_pool = 'avg'):
        super(MobileNeXt, self).__init__()

        self.width_mult = width_mult
        self.SE = SE                        # 暂时还没使用
        self.groups = groups
        self.norm_layer = norm_layer
        self.activation_type = act_layer
        self.activation = act_layer(inplace=True)
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # setting of sandglass blocks
        self.cfgs = [
            # t, c, n, s
            [2,   96, 1, 2],
            [6,  144, 1, 1],
            [6,  192, 3, 2],
            [6,  288, 3, 2],
            [6,  384, 4, 1],
            [6,  576, 4, 2],
            [6,  960, 3, 1],
            # [6, 1280, 1, 1],    # 原MobileNeXt模型参数,现为适配Grad_CAM,该层结构独立出Final_layer
        ]

        # building first layer
        input_channel = _make_divisible(32 * self.width_mult, self.groups)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = SandGlass
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * self.width_mult, self.groups)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        # 原为适配Grad_CAM,该层为独立出Final_layer
        layer1 = FinalSandGlassLayer(960, 1280, 1, 6)
        layers1 = [layer1]
        self.final_SandGlass_layers = nn.Sequential(*layers1)

        layer2 = FinalLayer(1280, 1)
        layers2 = [layer2]
        self.final_layers = nn.Sequential(*layers2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.final_SandGlass_layers(x)
        x = self.final_layers(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)  # 原MobileNeXt结构
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()





@register_model
def mobilenext_100(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """
    Constructs a MobileNeXt model
    """
    default_cfg = default_cfgs['mobilenext_100']
    model = MobileNeXt(num_classes=num_classes, width_mult=1., in_chans = in_chans, **kwargs)
    model.default_cfg = default_cfg
    return model