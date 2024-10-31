# -*- encoding: utf-8 -*-
'''
@File    :   base_runer.py
@Time    :   2024/10/31 19:17:37
@Author  :   junewluo 
'''

import os
import datetime
import wandb
import numpy as np
from tensorboardX import SummaryWriter
from utils.utils import create_logdir
from utils.logger import Logger

class BaseRuner(object):
    def __init__(self, args):
        self._build_time = datetime.datetime().strftime("%Y-%M-%D_%H-%M-%S")
        self._ppid = os.getppid()
        
        self._log_interval = args.log_interval
        self._use_wandb = args.use_wandb
        # init log tool
        if self._use_wandb:
            wandb.init(
                project = f"mappo-{args.env_name}",
                name = f"{self._build_time}_{self._ppid}" if args.log_name == None else args.log_name,
                mode = "off-line" if args.use_wandb_offline else None
            )
        else:
            logdir = create_logdir(args = args)
            self.tb_writer = SummaryWriter(logdir = logdir)
        
        # logger
        self.logger = Logger(std_out_console = True)

    def log_to_console(self, level, message):
        """ this method is used to log interval situation to console """
        if level == "info":
            self.logger.info(message = message)
        elif level == "warning":
            self.logger.warning(message = message)
        elif level == "error":
            self.logger.error(message = message)
        else:
            raise AttributeError(f'logger only has three types, "info", "warning", "error"!')

    def run(self):
        """ thie method is used to run a eposide of PPO algorithm """
        raise NotImplementedError(f"method run() have not yet implement! please implement it before you call it! ")

    def log_run(self):
        """ this metho is used to log run() function """
        raise NotImplementedError(f"method log_run() have not yet implement! please implement it before you call it! ")