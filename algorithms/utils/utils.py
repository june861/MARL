# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/10/27 18:51:30
@Author  :   junewluo 
'''

import torch
import numpy as np

def check(input, func = None):
    output = torch.Tensor(input) if type(input) == np.ndarray else input


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module