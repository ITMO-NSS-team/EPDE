#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:17:46 2021

@author: mike_ubuntu
"""

from epde.operators.template import CompoundOperator

class NDLElitism(CompoundOperator):
    '''
    Simple elitism for multiobjective optimization, preserving the elements of the non-dominated level.
    '''
    def apply(self, pareto_levels):
        for level in pareto_levels.levels[self.params['pareto_levels_excluded']:]:
            for individual in level:
                individual.elite = True
        return pareto_levels

    def use_default_tags(self):
        self._tags = {'elitism', 'pareto level level', 'auxilary', 'no suboperators'}