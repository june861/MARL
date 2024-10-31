# -*- encoding: utf-8 -*-
'''
@File    :   mappo_runer.py
@Time    :   2024/10/31 19:44:22
@Author  :   junewluo 
'''

import numpy as np
import os
import datetime
from runer.base_runer import BaseRuner
from buffers.mappo_sharebuffer import PPOShareBuffer

class MappoRuner(BaseRuner):
    def __init__(self, args):
        super(MappoRuner, self).__init__(args)

        self.replay_buffer = PPOShareBuffer(args = args)


    def run(self):
        return super().run()

