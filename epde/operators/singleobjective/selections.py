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
    """
    Implements roulette wheel selection, a fitness-proportionate selection method.
    
        This class selects individuals from a population with probabilities proportional
        to their fitness, simulating a roulette wheel where each individual occupies
        a slot sized according to its fitness.
    
        Methods:
        - apply
        - use_default_tags
    """

    key = 'RouletteWheelSelection'
    def apply(self, objective : Population, arguments: dict):
        """
        Applies a selection operator to a population based on fitness.
        
                This method selects parent candidates from a population based on their fitness
                scores, using a roulette wheel selection approach. It increments a counter for
                each selected candidate. This selection mechanism favors individuals with better
                fitness, increasing their chance to be chosen as parents for the next generation,
                driving the evolutionary process towards identifying better equation structures.
        
                Args:
                    objective (Population): The population to select from.
                    arguments (dict): A dictionary of arguments for the selection process.
        
                Returns:
                    Population: The updated population with incremented counters for selected candidates.
        """
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
        """
        Applies a predefined set of tags to the object.
        
        This method resets the object's tags to a default configuration, ensuring consistency in how the object is categorized within the evolutionary process. This is useful for standardizing the object's role and properties, especially during initialization or after modifications that might have altered its tagging.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        self._tags = {'selection', 'population level', 'auxilary', 'suboperators', 'standard'}