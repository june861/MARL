# -*- encoding: utf-8 -*-
'''
@File    :   rnn.py
@Time    :   2024/10/27 19:48:07
@Author  :   junewluo 
'''

import torch
import torch.nn as nn

class RNNLayer(nn.Module):
    def __init__(self):
        super(RNNLayer, self).__init__()
        