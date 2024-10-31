# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/10/27 18:51:30
@Author  :   junewluo 
'''

import torch
import torch.optim as optim
import numpy as np

def check(input, func = None):
    output = torch.Tensor(input) if type(input) == np.ndarray else input


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_network_optim(optim : str, network, lr):
    if optim.lower() == "adam":
        return optim.Adam(network.parameters(), lr = lr)
    elif optim.lower() == "adamw":
        return optim.AdamW(network.parameters(), lr = lr)
    else:
        raise NotImplementedError(f"Only Implement two optimizers, you can implement other optimizer in this method")