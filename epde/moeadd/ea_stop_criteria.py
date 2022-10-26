#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:51:55 2021

@author: mike_ubuntu
"""

from abc import ABC, abstractproperty

class Stop_condition(ABC):
    '''
    
    Abstract class for generalized stop condition for the evolutionary algorithm.
    
    Methods:
    ----------
    
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
    
class Iteration_limit(Stop_condition):
    def __init__(self, limit = 100): # estrategy, 
        self.limit = limit + 1
#        self.estrategy = estrategy
        self.checks = 0 # shortcut^ checking a number of calls instead of the estrategy attr of some sort
        
    def reset(self):
        self.checks = 0
    
    def check(self):
        self.checks += 1
        return self.checks > self.limit
    
class Fitness_stabilized(Stop_condition):
    def __init__(self, estrategy):
        raise NotImplementedError
    