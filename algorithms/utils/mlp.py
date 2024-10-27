# -*- encoding: utf-8 -*-
'''
@File    :   mlp.py
@Time    :   2024/10/27 19:11:27
@Author  :   junewluo 
'''

import torch
import torch.nn as nn
from .utils import init


class MLPLayer(nn.Module):
    """ In this class, we will implement MLPLayers except the last layers. """
    def __init__(self, input_dim, hidden_dims, use_orthogonal, use_ReLU, use_LayerNorm):
        """ init MLP network

        Args:
            input_dim (_int_): the input dimesion of network.
            hidden_dims (_list_): a list of hidden layers dimesions.
            use_orthogonal (_int_): a trick to init layer parameters.
            use_ReLU (_int_): use nn.ReLU as function or nn.Tanh.
            use_LayerNorm (_bool_): whether use a layernorm trick after a hidden layer.

        Returns:
            _none_: init func return nothing
        """
        super(MLPLayer, self).__init__()

        self.layers = []
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])
        assert type(hidden_dims) == list, f"hidden_dims must be a lits, but now recieve {type(hidden_dims)}"
        hidden_sizes = [input_dim] + hidden_dims

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        for i in range(len(hidden_sizes) - 1):
            in_d, out_d = hidden_sizes[i], hidden_sizes[i+1]
            if use_LayerNorm:
                self.layers.append(
                    nn.Sequential(init(nn.Linear(in_d, out_d)), active_func, nn.LayerNorm(out_d))
                )
            else:
                self.layers.append(
                    nn.Sequential(init(nn.Linear(in_d, out_d)), active_func)
                )
        self._layer_N = len(self.layers)
        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        """ network forward

        Args:
            x (_Tensor_): the input Tensor which is used to forwad calculation.

        Returns:
            _Tensor_: Hidden Layers Output of a Network.
        """
        for i in range(self._layer_N):
            x = self.layers[i](x)
        return x
