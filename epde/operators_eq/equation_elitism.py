#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:17:46 2021

@author: mike_ubuntu
"""

import numpy as np
from epde.operators.template import Compound_Operator
from epde.supplementary import Population_Sort

class Fraction_elitism(Compound_Operator):
    def apply(self, population):
        population = Population_Sort(population)
        if isinstance(self.params['elite_fraction'], float):
            assert self.params['elite_fraction'] <= 1 and self.params['elite_fraction'] >= 0
            fraction = int(np.ceil(self.params['elite_fraction'] * len(population)))
            
        for idx, elem in enumerate(population):
            if idx == 0:
                setattr(elem, 'elite', 'immutable')
            if idx < fraction:
                setattr(elem, 'elite', 'elite')
            else:
                setattr(elem, 'elite', 'non-elite')
                
        return population
    
    @property
    def operator_tags(self):
        return {'elitism', 'population level', 'auxilary', 'no suboperators'}