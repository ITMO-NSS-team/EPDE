#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:53:15 2021

@author: maslyaev
"""

from epde.operators.template import Compound_Operator
    
class Truncate_worst(Compound_Operator):
    '''
    
    
    '''
    def apply(self, population):
        return population[:self.params['population_size']]
    
    @property
    def operator_tags(self):
        return {'truncation', 'population level', 'no suboperators'}