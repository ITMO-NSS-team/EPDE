#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:49:36 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import reduce

from epde.moeadd.moeadd_supplementary import Constraint

from epde.operators.template import CompoundOperator
from epde.operators.supplementary_operators import SimpleNeighborSelector


class MOEADDSelection(CompoundOperator):
    def __init__(self, param_keys : list = []):
        self.neighborhoods_set = False
        self.suboperators
        super().__init__(param_keys)

    def set_neighborhoods(self, weights, neighborhood_vectors):
        self.neighborhoods_set = True
        self.weights = weights
        self.neighborhood_vectors = neighborhood_vectors
    
    def apply(self, weight_idx, pareto_levels, **kwargs) -> list:
        '''
        
        The mating operator, designed to select parents for the crossover with respect 
        to the location of the point in the objective functions values space and the 
        connected weight vector.
        
        Parameters:
        ------------
        
        weight_idx : int,
            Index of the processed weight vector.
            
        weights : np.ndarray,        
            Numpy array, containing weight vectors.
            
        neighborhood_vectors : list,
            List of lists, containing indexes: i-th element is the list of 
            k - closest to the i-the weight vector weight vectors.
            
        population : list,
            List of candidate solutions.
            
        neighborhood_selector : function,
            Method of finding "close neighbors" of the vector with proximity list.
            The baseline example of the selector, presented in 
            ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
            
        delta : float
            The probability of mating selection to be limited only to the selected
            subregions (adjacent to the weight vector domain). :math:`\delta \in [0., 1.)`
            
        Returns:
        ---------
        
        parent_idxs : list
            List of the selected parents in the population pool.
        
        '''
        if not self.neighborhoods_set:
            raise AttributeError('Neighborhood properties have not been set before the selection call.')
    
    
        parents_number = int(len(pareto_levels.population) * self.params['parents_fraction']) # Странное упрощение   
        if np.random.uniform() < self.params['delta']:
            selected_regions_idxs = self.suboperators['neighborhood_selector'].apply(self.neighborhood_vectors[weight_idx], 
                                                                                     *self.neighborhood_selector_params)
            candidate_solution_domains = list(map(lambda x: x.get_domain(self.weights), [candidate for candidate in 
                                                                                         pareto_levels.population]))
    
            solution_mask = [(pareto_levels.population[solution_idx].get_domain(self.weights) in selected_regions_idxs) 
                             for solution_idx in candidate_solution_domains]
            available_in_proximity = sum(solution_mask)
            parent_idxs = np.random.choice([idx for idx in np.arange(len(pareto_levels.population)) if solution_mask[idx]], 
                                            size = min(available_in_proximity, parents_number), 
                                            replace = False)
            if available_in_proximity < parents_number:
                parent_idxs_additional = np.random.choice([idx for idx in np.arange(len(pareto_levels.population))
                                                           if not solution_mask[idx]],
                                                          size = parents_number - available_in_proximity,
                                                          replace = False)
                parent_idxs_temp = np.empty(shape = parent_idxs.size + parent_idxs_additional.size)
                parent_idxs_temp[:parent_idxs.size] = parent_idxs; parent_idxs_temp[parent_idxs.size:] = parent_idxs_additional
                parent_idxs = parent_idxs_temp
        else:
            parent_idxs = np.random.choice(np.arange(len(pareto_levels.population)), size = parents_number, replace = False)
        for idx in parent_idxs:
            pareto_levels.population[int(idx)].incr_counter()    
        return pareto_levels
    
    def use_default_tags(self):
        self._tags = {'selection', 'pareto_levels level', 'auxilary', 'suboperators'}


class MOEADDSelectionConstrained(CompoundOperator):
    def __init__(self, param_keys : list = []):
        self.neighborhoods_set = False
        self.suboperators
        super().__init__(param_keys)

    def set_neighborhoods(self, weights, neighborhood_vectors):
        self.neighborhoods_set = True
        self.weights = weights
        self.neighborhood_vectors = neighborhood_vectors
    
    def apply(self, weight_idx, pareto_levels, **kwargs) -> list:
        '''
        
        The mating operator, designed to select parents for the crossover with respect 
        to the location of the point in the objective functions values space and the 
        connected weight vector.
        
        Parameters:
        ------------
        
        weight_idx : int,
            Index of the processed weight vector.
            
        weights : np.ndarray,        
            Numpy array, containing weight vectors.
            
        neighborhood_vectors : list,
            List of lists, containing indexes: i-th element is the list of 
            k - closest to the i-the weight vector weight vectors.
            
        population : list,
            List of candidate solutions.
            
        neighborhood_selector : function,
            Method of finding "close neighbors" of the vector with proximity list.
            The baseline example of the selector, presented in 
            ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
            
        delta : float
            The probability of mating selection to be limited only to the selected
            subregions (adjacent to the weight vector domain). :math:`\delta \in [0., 1.)`
            
        Returns:
        ---------
        
        parent_idxs : list
            List of the selected parents in the population pool.
        
        '''
        if not self.neighborhoods_set:
            raise AttributeError('Neighborhood properties have not been set before the selection call.')
    
    
        if 'constraint_processer' in list(self.suboperators.keys()):
            multiplier = self.suboperators['constraint_processer'].params['group_size']
            
        parents_number = int(len(pareto_levels.population) * self.params['parents_fraction'] * multiplier)
        if np.random.uniform() < self.params['delta']:
            selected_regions_idxs = self.suboperators['neighborhood_selector'].apply(self.neighborhood_vectors[weight_idx], 
                                                                                     *self.neighborhood_selector_params)
            candidate_solution_domains = list(map(lambda x: x.get_domain(self.weights), [candidate for candidate in 
                                                                                         pareto_levels.population]))
    
            solution_mask = [(pareto_levels.population[solution_idx].get_domain(self.weights) in selected_regions_idxs) 
                             for solution_idx in candidate_solution_domains]
            available_in_proximity = sum(solution_mask)
            parent_idxs = np.random.choice([idx for idx in np.arange(len(pareto_levels.population)) if solution_mask[idx]], 
                                            size = min(available_in_proximity, parents_number), 
                                            replace = False)
            if available_in_proximity < parents_number:
                parent_idxs_additional = np.random.choice([idx for idx in np.arange(len(pareto_levels.population))
                                                           if not solution_mask[idx]],
                                                          size = parents_number - available_in_proximity,
                                                          replace = False)
                parent_idxs_temp = np.empty(shape = parent_idxs.size + parent_idxs_additional.size)
                parent_idxs_temp[:parent_idxs.size] = parent_idxs; parent_idxs_temp[parent_idxs.size:] = parent_idxs_additional
                parent_idxs = parent_idxs_temp
        else:
            parent_idxs = np.random.choice(np.arange(len(pareto_levels.population)), size = parents_number, replace = False)

        if 'constraint_processer' in list(self.suboperators.keys()):
            assert 'constraints' in self.suboperators['constraint_processer'].tags
            parent_idxs = self.suboperators['constraint_processer'].apply(parent_idxs, pareto_levels)

        for idx in parent_idxs:
            pareto_levels.population[int(idx)].incr_counter()
        return pareto_levels

    def use_default_tags(self):
        self._tags = {'selection', 'pareto_levels level', 'auxilary', 'suboperators'}


class SelectionConstraintProcesser(object):
    def __init__(self, constraints = None, param_keys = []):
        assert all(isinstance(constraints, Constraint) for constr in constraints)
        self._constraints = constraints
        super().init(param_keys)
    
    def apply(self, indexes, pareto_levels):
        def violation(individual):
            return reduce(lambda y, z: y + z(individual.vals()), self._constraints, initial = 0)

        constraint_violations = {idx : violation(pareto_levels.population[idx]) for idx in indexes}
        selected_idxs = []
        for idx in np.arange(int(indexes/self.params['group_size'])):
            first_elem_idx = idx * self.params['group_size']
            if all([constraint_violations[indexes[first_elem_idx]] == constraint_violations[indexes[first_elem_idx + incr]] 
                    for incr in range(self.params['group_size'])]):
                selected_idxs.append(np.random.choice(a = indexes[first_elem_idx : first_elem_idx + self.params['group_size']]))
        if len(selected_idxs) % 2: selected_idxs = selected_idxs[:-1]
        np.random.shuffle(selected_idxs)
        return selected_idxs
    
    def use_default_tags(self):
        self._tags = {'constraints', 'selection', 'mixed input level', 'auxilary'}