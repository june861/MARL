# -*- encoding: utf-8 -*-
'''
@File    :   mappo_algo.py
@Time    :   2024/10/31 19:51:42
@Author  :   junewluo 
'''

import torch
import numpy as np
from algorithms.algos.base_algo import BaseAlgo
from algorithms.utils.ac import Actor, Critic
from algorithms.utils.utils import init_network_optim

class MappoAlgo(BaseAlgo):
    def __init__(self, args):
        super(MappoAlgo, self).__init__()
        self._obs_dim = args.obs_dim
        self._act_dim = args.act_dim
        self._policy_hidden_dims = args.policy_hidden_dims
        self._value_hidden_dims = args.value_hidden_dims
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._use_LayerNorm = args.use_LayerNorm
        self._continous_type = args.continous_type
        self._device = args.device
        self._lr_a = args.lr_a
        self._lr_c = args.lr_c
        self._optim = args.optim
        self._ppo_epoch = args.ppo_epoch

        # define AC framework for mappo
        self.actor = Actor(obs_dim = self._obs_dim, hidden_dims = self._policy_hidden_dims,
                           use_recurrent_policy = self._use_recurrent_policy,
                           use_orthogonal = self._use_orthogonal,
                           use_ReLU = self._use_ReLU, continous_type = self._continous_type,
                           use_LayerNorm = self._use_LayerNorm, device = self._device,
                        )
        self.critic = Critic(obs_dim = self._obs_dim, hidden_dims = self._policy_hidden_dims,
                            use_recurrent_policy = self._use_recurrent_policy,
                            use_orthogonal = self._use_orthogonal,
                            use_ReLU = self._use_ReLU, continous_type = self._continous_type,
                            use_LayerNorm = self._use_LayerNorm, device = self._device,
                        )
        
        self.actor_optimizer = init_network_optim(optim = self._optim, network = self.actor, lr = self._lr_a)
        self.critic_optimizer = init_network_optim(optim = self._optim, network = self.critic, lr = self._lr_c)
    
    def update(self):
        """ """
        return super().update()
    
    def learn(self, ):
        return super().learn()
    
    def save(self):
        return super().save()
    
    def load(self):
        return super().load()