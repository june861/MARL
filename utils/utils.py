# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/10/31 21:32:00
@Author  :   junewluo 
'''

import os

def check_dir(dir):
    if not dir or os.path.exists(dir):
        return
    head, tail = os.path.split(dir)
    check_dir(head)
    os.mkdir(dir)


def create_logdir(args):
    env_name = args.env_name
    algo_name = args.algo_name
    seed = args.seed
    log_name = args.log_name

    # create a dir for log
    prefix_dir = f'./{env_name}/{algo_name}'
    check_dir(prefix_dir)
    logdir = prefix_dir + f"{seed}" +log_name

    return logdir
