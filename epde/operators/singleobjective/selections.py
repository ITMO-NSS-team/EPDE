#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:49:36 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import reduce

from epde.optimizers.single_criterion.optimizer import Population
from epde.optimizers.moeadd.moeadd import ParetoLevels

from epde.operators.utils.template import CompoundOperator


class RouletteWheelSelection(CompoundOperator):
    key = 'RouletteWheelSelection'
    def apply(self, objective : Population, arguments: dict):
        # TODO: add docstring
        if isinstance(objective, ParetoLevels):
            raise TypeError('Tring to call method, implemented for Population class objects, to a ParetoLevels object. Must be a wrong type of evolution.')
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        parents_number_counted = int(len(objective.population) * self.params['parents_fraction']) # TODO: Странное упрощение 
        parents_number_counted = parents_number_counted if not parents_number_counted % 2 else parents_number_counted + 1
        
        parents_number = min(max(2, parents_number_counted), len(objective.population))

        fitnesses = np.array([1/candidate.obj_fun for candidate in objective]) # Inspect for cases, when the solutions have relatively good fitness
        probas = (fitnesses/np.sum(fitnesses)).squeeze()

        # print(probas.shape)
        candidate_idxs = np.random.choice(range(len(objective.population)), size = parents_number, 
                                          replace = True, p = probas) # Experiment with roulette with replace = False
        for idx in candidate_idxs:
            objective.population[idx].incr_counter() # TODO: direct access
        
        return objective
    
    def use_default_tags(self):
        self._tags = {'selection', 'population level', 'auxilary', 'suboperators', 'standard'}