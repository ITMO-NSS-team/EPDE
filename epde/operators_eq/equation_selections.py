#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:49:36 2021

@author: mike_ubuntu
"""

import numpy as np
from epde.operators.template import Compound_Operator

class Tournament_selection(Compound_Operator):
    """
    Basic tournament selection, inherits properties from class ``Compound_Operator()``;

    Methods:
    ---------
    
    apply(population)
        return the indexes of the individuals, selected for procreation.
    """
    def apply(self, population):
        """
        Select a pool of pairs, between which the procreation will be later held. The population is divided into groups, and the individual 
        with highest fitness is allowed to take part in the crossover. 
        
        Parameters:
        -----------
        population : list of equation objects
            The population, among which the selection is held.
        
        Returns:
        -----------
        parent_pairs : list of lists (pairs of parents)
            The pairs of parents, chosen to take part in the crossover.
            
        """
#        raise NotImplementedError('Kek')
        for solution in population:
            solution.crossover_selected_times = 0
#        print('\n')
#        print('size of the selection:', int(len(population)*self.params['part_with_offsprings']), 'from', len(population), self.params['part_with_offsprings'])
#        print('\n')
        # print('Running selection')
        ssize = int(len(population)*self.params['part_with_offsprings']) 
        if ssize == 0: ssize += 2
        if ssize & 0x1: ssize += 1
        
        for elem_idx in range(ssize):
            selection_indexes = np.random.choice(len(population), self.params['tournament_groups'], replace = False)
            candidates = [population[idx] for idx in selection_indexes]
#            [idx for _, idx in sorted(zip(candidates, selection_indexes), key=lambda pair: pair[0].fitness_value)][-1].crossover_selected_times += 1
            [solution for solution in sorted(candidates, key=lambda sol: sol.fitness_value)][-1].crossover_selected_times += 1
        return population
    
    @property
    def operator_tags(self):
        return {'selection', 'population level', 'auxilary', 'no suboperators'}    