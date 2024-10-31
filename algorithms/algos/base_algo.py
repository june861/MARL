# -*- encoding: utf-8 -*-
'''
@File    :   base_algo.py
@Time    :   2024/10/31 19:47:01
@Author  :   junewluo 
'''

class BaseAlgo(object):
    def __init__(self):
        print(f"BaseAlgo Class Init")
    
    def update(self):
        """ this method will be used to backward and update network parameter """
        raise NotImplementedError(f"method update() have not yet implement! please implement it before you call it! ")
    
    def learn(self):
        """ this method is used to give outside calling """
        raise NotImplementedError(f"method learn() have not yet implement! please implement it before you call it! ")

    def save(self):
        """ this method is used to save a model """
        raise NotImplementedError(f"method save() have not yet implement! please implement it before you call it! ")
    
    def load(self):
        """ this method is used to load a saved model """
        raise NotImplementedError(f"method load() have not yet implement! please implement it before you call it! ")