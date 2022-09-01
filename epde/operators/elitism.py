#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:17:46 2021

@author: mike_ubuntu
"""

from operator import index
from epde.operators.template import CompoundOperator

class NDLElitism(CompoundOperator):
    '''
    Simple elitism for multiobjective optimization, preserving the elements of the non-dominated level.
    '''
    def apply(self, objective):
        def set_elite_marker(marker : int, indexes : slice):
            for level in objective.levels[indexes]:
                for individual in level:
                    individual.elite = marker
            
        elite_indexes = slice(0, self.params['elite_num'])
        set_elite_marker(2, elite_indexes)

        refining_indexes = slice(self.params['elite_num'], self.params['elite_num'] + self.params['refining_num'])
        set_elite_marker(1, refining_indexes)

        return objective

    def use_default_tags(self):
        self._tags = {'elitism', 'pareto level level', 'auxilary', 'no suboperators', 'standard'}