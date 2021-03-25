#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File         :   tinynet.py
@Modify Time  :   2021/3/25 上午10:21
@Author       :   night
@Version      :   1.0
@License      :   (C)Copyright 2019-2021, Real2tech
@Desciption   :   An implementation of TinyNet

'''

#            Huawei Technologies Co., Ltd. <foss@huawei.com>
from timm.models.efficientnet_builder import *
from timm.models.efficientnet_blocks import round_channels, resolve_bn_args, resolve_act_layer
from timm.models.efficientnet import EfficientNet, EfficientNetFeatures, _cfg
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _gen_tinynet(variant_cfg, channel_multiplier=1.0, depth_multiplier=1.0, depth_trunc='round', pretrained=False, **kwargs):
    """Creates a TinyNet model.
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'], ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'], ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'], ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc=depth_trunc),
        num_features=max(1280, round_channels(1280, channel_multiplier, 8, None)),
        stem_size=32,
        fix_stem=True,
        channel_multiplier=channel_multiplier,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    model = EfficientNet(**model_kwargs)
    model.default_cfg = variant_cfg
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'tinynet_050': _cfg(url=''),
    'tinynet_075': _cfg(url=''),
    'tinynet_100': _cfg(url=''),

}


@register_model
def tinynet_050(pretrained=False, **kwargs):
    """ TinyNet """
    # hw = int(224 * r)
    model = _gen_tinynet(
        default_cfgs['tinynet_050'], channel_multiplier=0.5, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model

@register_model
def tinynet_075(pretrained=False, **kwargs):
    """ TinyNet """
    # hw = int(224 * r)
    model = _gen_tinynet(
        default_cfgs['tinynet_075'], channel_multiplier=0.75, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model

@register_model
def tinynet_100(pretrained=False, **kwargs):
    """ TinyNet """
    # hw = int(224 * r)
    model = _gen_tinynet(
        default_cfgs['tinynet_100'], channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model
