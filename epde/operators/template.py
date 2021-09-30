#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:55:18 2021

@author: mike_ubuntu
"""

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
        if not all([isinstance(key, str) and isinstance(value, (Compound_Operator, list, tuple, dict)) for key, value
                    in operators.items()]):
            raise TypeError('The suboperators of an evolutionary operator must be declared in format key : value, where key is str and value - Compound_Operator, list, tuple or dict')
        self._suboperators = operators

    def apply(self, target):
        pass

    @property
    def operator_tags(self):
        return set()
    
#    def set_next(self, operators):
#        
#    
#    def apply_next()
