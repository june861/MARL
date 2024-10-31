# -*- encoding: utf-8 -*-
'''
@File    :   ppo_sharebuffer.py
@Time    :   2024/10/31 19:34:13
@Author  :   junewluo 
'''

import numpy as np
from buffers.base_sharebuffer import BaseShareBuffer
from torch.utils.data import RandomSampler, SubsetRandomSampler

class PPOShareBuffer(BaseShareBuffer):
    def __init__(self, args):
        super(PPOShareBuffer, self).__init__(args)
        # ppo special rollout data
        self._a_logprobs = np.zeros(shape = self._re_done_shape)
    
    def add(self, obs, action, a_logprob, next_obs, done):
        self._obs[self._index % self._max_batch_steps] = obs
        self._actions[self._index % self._max_batch_steps] = action
        self._a_logprobs[self._index % self._max_batch_steps] = a_logprob
        self._next_obs[self._index % self._max_batch_steps] = next_obs
        self._dones[self._index % self._max_batch_steps] = done
        self._index += 1

    def sample(self, n_sample):
        if n_sample > self._max_batch_steps:
            n_sample = self._max_batch_steps 

        for indice in SubsetRandomSampler(RandomSampler(range(self._max_batch_steps)), n_sample, False):
            obs = self._obs[indice]
            action = self._actions[indice]
            a_logprob = self._a_logprobs[indice]
            next_obs = self._next_obs[indice]
            dones = self._dones[indice]

            yield obs, action, a_logprob, next_obs, dones 