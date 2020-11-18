# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20
"""
from torch.nn import init
from torch import nn


def weights_init_normal(model, std=0.02):
    for m in model.modules():

        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Parameter):
            init.normal_(m.weight.data, 0.0, std)
    return model