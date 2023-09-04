#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:51:55 2021

@author: mike_ubuntu
"""

from abc import ABC, abstractproperty

class StopCondition(ABC):
    '''
    Abstract class for generalized stop condition for the evolutionary algorithm.
    
    Methods:    
        reset():
            Reset the object. Mainly useful, if some sort of counter is implemented inside the instance. Returns None.
        check()
    
    '''
    def __init__(self, estrategy):
        pass
    
    def reset(self):
        pass
    
    def check(self):
        pass
    
class IterationLimit(StopCondition):
    def __init__(self, limit = 100):
        self.limit = limit + 1
        self.checks = 0 # shortcut^ checking a number of calls instead of the estrategy attr of some sort
    
    def reset(self, **kwargs):
        self.checks = 0
        self.limit = kwargs.get('limit', self.limit)
    
    def check(self):
        self.checks += 1
        return self.checks > self.limit
    
class FitnessStabilized(StopCondition):
    def __init__(self, estrategy):
        raise NotImplementedError
    
