# -*- encoding: utf-8 -*-
'''
@File    :   base_sharebuffer.py
@Time    :   2024/10/31 19:22:44
@Author  :   junewluo 
'''

import numpy as np


class BaseShareBuffer(object):
    def __init__(self, args):
        self._max_batch_steps = args.max_batch_steps
        self._n_rollout_thread = args.n_rollout_thread
        self._obs_dim  = args.obs_dim
        self._act_dim = args.act_dim
        self._num_agents = args.num_agents
        self._index = 0

        # share data
        self._obs_shape = (self._max_batch_steps, self._n_rollout_thread, self._num_agents, self._obs_dim)
        self._act_shape = (self._max_batch_steps, self._n_rollout_thread, self._num_agents, self._act_dim)
        self._re_done_shape = (self._max_batch_steps, self._n_rollout_thread, self._num_agents, 1)
        self._obs = np.zeros(shape = self._obs_shape)
        self._actions = np.zeros(shape = self._act_shape)
        self._next_obs = np.zeros(shape = self._obs_shape)
        self._rewards = np.zeros(shape = self._re_done_shape)
        self._dones = np.zeros_like(shape = self._re_done_shape)
    
    def add(self):
        """ add data into replay buffer, you can implement this method like:
                self._obs[index % self._max_batch_steps] = obs
                self._next_obs[index % self._max_batch_steps] = next_obs
                ....
                self.index += 1
        """
        raise NotImplementedError(f"method add() have not yet implement! please implement it before you call it! ")
    
    def sample(self):
        """ sample data from replay buffer """
        raise NotImplementedError(f"method sample() have not yet implement! please implement it before you call it! ")
    
    