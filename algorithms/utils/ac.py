# -*- encoding: utf-8 -*-
'''
@File    :   ac.py
@Time    :   2024/10/31 15:15:32
@Author  :   junewluo 
'''

import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils.mlp import MLPLayer
from utils.utils import check


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dims, act_dim, use_recurrent_policy, 
                 use_orthogonal, use_ReLU, use_LayerNorm, continous_type = False, device = None):
        super(Actor, self).__init__()

        self.forward_layers = MLPLayer(obs_dim, hidden_dims, use_orthogonal, use_ReLU, use_LayerNorm)
        self.last_layers = nn.Linear(hidden_dims[-1], act_dim)
        self._continous = continous_type
        self._use_recurrent_policy = use_recurrent_policy
        self._device = device
        # implement ppo-continous
        if self._continous:
            pass
        else:
            pass
    

    def forwad(self, cen_obs, rnn_state = None, mask = None, deterministic = False):
        """ forward process of actor.

        Args:
            cen_obs (_numpy.ndarray or torch.tensor_): center observation
            rnn_state (_none_, optional): _description_. Defaults to None.
            mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _torch.tensor_: a series of action that will be taken by agents.
            _torch.tensor_: a series of log(action).
        """

        cen_obs = check(cen_obs).to(self._device)
        if rnn_state != None:
            rnn_state = check(rnn_state).to(self._device)
        if mask != None:
            mask = check(mask).to(self._device)
        
        hidden_out = self.forward_layers(cen_obs)
        net_out = self.last_layers(hidden_out)
        dist = Categorical(net_out)
        action = dist.sample()
        a_logprob = dist.log_prob(action)

        return action, a_logprob





class Critic(nn.Module):
    def __init__(self, 
                 obs_dim, hidden_dims, use_orthogonal, use_recurrent_policy,
                 use_ReLU, use_LayerNorm, continous_type = False, device = None,*args, **kwargs
                ):
    
        super(Critic, self).__init__(*args, **kwargs)
        self._use_recurrent_policy = use_recurrent_policy
        self.forward_layers = MLPLayer(obs_dim, hidden_dims, use_orthogonal, use_ReLU, use_LayerNorm)
        self.last_layer = nn.Linear(hidden_dims[-1], 1)
        self._device = device
    

    def forward(self, cen_obs, rnn_state, mask):
        cen_obs = check(cen_obs).to(self._device)
        if rnn_state != None:
            rnn_state = check(rnn_state).to(self._device)
        if mask != None:
            mask = check(mask).to(self._device)
        hidden_out = self.forward_layers(cen_obs)
        value = self.last_layers(hidden_out)

        return value


        
        


