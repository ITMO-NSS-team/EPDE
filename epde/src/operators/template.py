#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:55:18 2021

@author: mike_ubuntu
"""

from functools import reduce
#from ,,, import GeneticOperator

def flatten(folded_equation):
    assert type(folded_equation) == list
    return reduce(lambda x,y: x+y, folded_equation)


def try_iterable(arg):
    try:
        _ = [elem for elem in arg]
    except TypeError:
        return False
    return True


class Compound_Operator(): #(GeneticOperator)
    '''
    Universal class for operator of an arbitrary purpose
    '''
    def __init__(self, param_keys : list):
        self.param_keys = param_keys
    
    @property
    def params(self):
        return self._params #list(self._params.keys()), self._param.values()
    
    @params.setter
    def params(self, param_dict : dict):
#        print(list(param_dict.keys()), self.param_keys)
        if set(self.param_keys) != set(param_dict.keys()):
            print('self.param_keys:', set(self.param_keys), ' param_dict.keys():', set(param_dict.keys()))
            raise KeyError('Wrong keys of param dict')
        self._params = param_dict

    @property
    def suboperators(self):
        return self._suboperators
        
    @suboperators.setter
    def suboperators(self, operators : dict):
        if not all([isinstance(key, str) and isinstance(value, (Compound_Operator, list, tuple)) for key, value
                    in operators.items()]):
            raise TypeError('The suboperators of an evolutionary operator must be declared in format key : value, where key is str and value - Compound_Operator')
        self._suboperators = operators

    def apply(self, target):
        pass
    
#    def set_next(self, operators):
#        
#    
#    def apply_next()
