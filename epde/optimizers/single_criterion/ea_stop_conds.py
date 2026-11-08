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
    '''
    ``limit`` iterations; with ``patience`` (``early_stopping_patience``) also stops once the best
    individual's structure, reported through :meth:`observe` after every iteration, has not
    changed for ``patience`` iterations in a row.
    '''
    def __init__(self, limit = 100, patience = None):
        self.limit = limit + 1
        self.checks = 0 # shortcut^ checking a number of calls instead of the estrategy attr of some sort
        self.patience = patience
        self._key, self._still = None, 0

    def reset(self, **kwargs):
        self.checks = 0
        self.limit = kwargs.get('limit', self.limit)
        self.patience = kwargs.get('patience', None)
        self._key, self._still = None, 0

    def observe(self, population):
        """Record the structure of the best individual (the population's first, as sorted)."""
        if self.patience is None:
            return
        elements = getattr(population, 'population', population)
        key = getattr(elements[0], 'equations_labels', None) if len(elements) else None
        self._still = self._still + 1 if key is not None and key == self._key else 0
        self._key = key

    def check(self):
        self.checks += 1
        if self.patience is not None and self._still >= self.patience:
            return True
        return self.checks > self.limit
    
class FitnessStabilized(StopCondition):
    def __init__(self, estrategy):
        raise NotImplementedError
    
