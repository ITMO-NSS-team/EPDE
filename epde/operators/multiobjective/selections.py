#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:49:36 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import reduce

from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.optimizers.moeadd.supplementary import Constraint

from epde.operators.utils.template import CompoundOperator


class MOEADDSelection(CompoundOperator):
    '''
    This class implements the selection mechanism used in the MOEA/DD algorithm.
    
        Class Methods:
        - apply: Selects parents for crossover based on weight vectors and neighborhood.
        - arguments: Returns the set of arguments used by the layer.
        - use_default_tags: Uses the default set of tags for the object.
    '''

    key = 'MOEADDSelection'
    
    def apply(self, objective : ParetoLevels, arguments : dict): # pareto_levels
        '''
        
        The mating operator, designed to select parents for the crossover with respect 
        to the location of the point in the objective functions values space and the 
        connected weight vector.
        
        Parameters:
        """
        Selects parents for crossover, prioritizing solutions within the neighborhood of a weight vector to encourage localized search and maintain diversity in the population. This approach balances exploration and exploitation by focusing on promising regions of the search space while still allowing for broader exploration.
        
                Args:
                    objective (ParetoLevels): The current Pareto front, containing the population and their objective values.
                    arguments (dict): A dictionary containing the necessary parameters for the mating selection process, including:
                        - weight_idx (int): Index of the processed weight vector.
                        - weights (np.ndarray): Numpy array, containing weight vectors.
                        - neighborhood_vectors (list): List of lists, containing indexes: i-th element is the list of k - closest to the i-the weight vector weight vectors.
                        - population (list): List of candidate solutions.
                        - neighborhood_selector (function): Method of finding "close neighbors" of the vector with proximity list. The baseline example of the selector, presented in ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
                        - delta (float): The probability of mating selection to be limited only to the selected subregions (adjacent to the weight vector domain). :math:`\delta \in [0., 1.)`
        
                Returns:
                    ParetoLevels: The updated Pareto front with incremented selection counters for the chosen parents.
        """
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
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        parents_number_counted = int(len(objective.population) * self.params['parents_fraction']) # Странное упрощение 
        parents_number_counted = parents_number_counted if not parents_number_counted % 2 else parents_number_counted + 1
        
        parents_number = min(max(2, parents_number_counted), len(objective.population))
                             
        if np.random.uniform() < self.params['delta']:
            selected_regions_idxs = self.suboperators['neighborhood_selector'].apply(self_args['neighborhood_vectors'][self_args['weight_idx']],
                                                                                     arguments = subop_args['neighborhood_selector']) #, 

            candidate_solution_domains = list(map(lambda x: x.get_domain(self_args['weights']), [candidate for candidate in 
                                                                                                 objective.population]))

            try:
                solution_mask = [(objective.population[solution_idx].get_domain(self_args['weights']) in selected_regions_idxs) 
                                 for solution_idx in candidate_solution_domains]
            except IndexError:
                print(f'Indexes are: {[solution_idx for solution_idx in candidate_solution_domains]}')
                print(len(objective.population), len(candidate_solution_domains))
                raise IndexError('list index out of range')
            available_in_proximity = sum(solution_mask)
            parent_idxs = np.random.choice([idx for idx in np.arange(len(objective.population)) if solution_mask[idx]], 
                                            size = min(available_in_proximity, parents_number),
                                            replace = False)
            if available_in_proximity < parents_number: 
                parent_idxs_additional = np.random.choice([idx for idx in np.arange(len(objective.population))
                                                           if not solution_mask[idx]],
                                                          size = parents_number - available_in_proximity,
                                                          replace = False)
                parent_idxs_temp = np.empty(shape = parent_idxs.size + parent_idxs_additional.size)
                parent_idxs_temp[:parent_idxs.size] = parent_idxs; parent_idxs_temp[parent_idxs.size:] = parent_idxs_additional
                parent_idxs = parent_idxs_temp
        else:
            parent_idxs = np.random.choice(np.arange(len(objective.population)), size = parents_number, replace = False)
        for idx in parent_idxs.reshape(-1):
            objective.population[int(idx)].incr_counter()
        return objective
    
    @property
    def arguments(self):
        """
        Returns the set of arguments required to evaluate the contribution of each candidate solution.
        
                These arguments define the information needed to assess the quality and diversity of solutions within the population.
        
                Returns:
                    set: A set containing the strings 'neighborhood_vectors',
                         'weight_idx', and 'weights', representing the arguments
                         used by the layer.
        """
        return set(['neighborhood_vectors', 'weight_idx', 'weights'])    
    
    def use_default_tags(self):
        """
        Uses a predefined set of tags to categorize and manage the selection process.
        
                This method overwrites any existing tags with a predefined set of default tags, ensuring consistency in how selection mechanisms are classified within the evolutionary process. This standardization aids in the organization and analysis of different selection strategies.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        
                Class Fields:
                    _tags (set): A set containing the tags associated with the object. Initialized to a set containing 'selection', 'population level', 'auxilary', 'suboperators', and 'standard'.
        """
        self._tags = {'selection', 'population level', 'auxilary', 'suboperators', 'standard'}


class MOEADDSelectionConstrained(CompoundOperator):
    """
    MOEADDSelectionConstrained performs the selection process in MOEA/D with constraint handling.
    
        This class implements a selection mechanism that considers both objective function values and constraint violations
        to guide the search towards feasible and optimal solutions.
    """

    key = 'MOEADDSelectionConstrained'
    
    def apply(self, objective : ParetoLevels, arguments : dict):
        """
        Selects parents for crossover, prioritizing solutions within specific regions of the objective space.
        
                This method strategically chooses parent solutions for the crossover operation.
                It focuses on selecting parents that reside in proximity to particular weight vectors,
                thereby encouraging the exploration of promising areas within the objective space.
                This targeted selection aims to enhance the discovery of equation structures that
                effectively model the underlying dynamics of the system.
        
                Args:
                    objective (ParetoLevels): The current Pareto front of solutions.
                    arguments (dict): A dictionary containing necessary parameters, including:
                        - weight_idx (int): Index of the processed weight vector.
                        - weights (np.ndarray): Numpy array of weight vectors.
                        - neighborhood_vectors (list): List of neighboring weight vector indices for each weight vector.
                        - population (list): List of candidate solutions.
                        - neighborhood_selector (function): Method for selecting neighboring weight vectors.
                        - delta (float): Probability of limiting mating selection to adjacent subregions.
                Returns:
                    objective (ParetoLevels): Updated Pareto front with incremented counters for selected parents.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
    
        if 'constraint_processer' in list(self.suboperators.keys()):
            multiplier = self.suboperators['constraint_processer'].params['group_size']
            
        parents_number = int(len(objective.population) * self.params['parents_fraction'] * multiplier)
        if np.random.uniform() < self.params['delta']:
            selected_regions_idxs = self.suboperators['neighborhood_selector'].apply(self_args['neighborhood_vectors'][self_args['weight_idx']])
            candidate_solution_domains = list(map(lambda x: x.get_domain(self_args['weights']), [candidate for candidate in 
                                                                                                 objective.population]))
    
            solution_mask = [(objective.population[solution_idx].get_domain(self_args['weights']) in selected_regions_idxs) 
                             for solution_idx in candidate_solution_domains]
            available_in_proximity = sum(solution_mask)
            parent_idxs = np.random.choice([idx for idx in np.arange(len(objective.population)) if solution_mask[idx]], 
                                            size = min(available_in_proximity, parents_number), 
                                            replace = False)
            if available_in_proximity < parents_number:
                parent_idxs_additional = np.random.choice([idx for idx in np.arange(len(objective.population))
                                                           if not solution_mask[idx]],
                                                          size = parents_number - available_in_proximity,
                                                          replace = False)
                parent_idxs_temp = np.empty(shape = parent_idxs.size + parent_idxs_additional.size)
                parent_idxs_temp[:parent_idxs.size] = parent_idxs; parent_idxs_temp[parent_idxs.size:] = parent_idxs_additional
                parent_idxs = parent_idxs_temp
        else:
            parent_idxs = np.random.choice(np.arange(len(objective.population)), size = parents_number, replace = False)

        if 'constraint_processer' in list(self.suboperators.keys()):
            assert 'constraints' in self.suboperators['constraint_processer'].tags
            parent_idxs = self.suboperators['constraint_processer'].apply(parent_idxs, objective)

        for idx in parent_idxs:
            objective.population[int(idx)].incr_counter()
        return objective
    
    @property
    def arguments(self):
        """
        Returns the set of arguments required for the selection process.
        
                These arguments define the information needed to evaluate and compare
                candidate solutions within the evolutionary algorithm. Specifically,
                'neighborhood_vectors' are used to consider solutions within a defined
                vicinity, 'weight_idx' helps in navigating the Pareto front, and
                'weights' are the weights used for multi-objective optimization.
        
                Args:
                    None
        
                Returns:
                    set: A set containing the strings 'neighborhood_vectors',
                         'weight_idx', and 'weights', representing the arguments
                         used by the selection mechanism.
        """
        return set(['neighborhood_vectors', 'weight_idx', 'weights'])    

    def use_default_tags(self):
        """
        Uses a predefined set of tags to categorize the selection operator.
        
                This method overwrites any existing tags with a default set, ensuring consistent categorization. This helps in organizing and identifying the selection operator within the broader evolutionary process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        self._tags = {'selection', 'population level', 'auxilary', 'suboperators', 'standard'}


class SelectionConstraintProcesser(object):
    """
    Processes selection constraints to filter individuals based on constraint violations.
    
        This class is designed to apply constraints during a selection process,
        typically within an optimization or evolutionary algorithm. It identifies
        and handles groups of individuals with identical constraint violations,
        ensuring diversity in the selection.
    """

    key = 'SelectionConstraintProcesser'
    
    def __init__(self, constraints = None, param_keys = []):
        """
        Initializes the ConstraintSet object.
        
        This constructor initializes the constraint set with a list of constraints
        and a list of parameter keys. These constraints will be used to guide the search
        for suitable equation structures by imposing restrictions on the possible solutions.
        It also calls the superclass's init method to initialize parameter keys.
        
        Args:
            constraints (list): A list of Constraint objects that define the restrictions on the equation search space.
            param_keys (list): A list of parameter keys associated with the constraints.
        
        Returns:
            None
        
        Class Fields:
            _constraints (list): A list of Constraint objects associated with the set.
        """
        assert all(isinstance(constraints, Constraint) for constr in constraints)
        self._constraints = constraints
        super().init(param_keys)
    
    def apply(self, indexes, pareto_levels):
        """
        Applies a selection process to enhance population diversity based on constraint violations within groups.
        
                This method selects individuals from a given set of indexes, prioritizing diversity by examining constraint violation scores within groups.
                It groups individuals and selects a random representative from groups where all members exhibit identical constraint violations.
                This approach helps maintain a diverse population by ensuring that similar solutions within a group do not dominate the selection process.
                Finally, the selected individuals are shuffled to further promote randomness in the evolutionary process.
        
                Args:
                    indexes (list): The indexes of the individuals to consider.
                    pareto_levels (ParetoLevels): The Pareto levels object containing the population data.
        
                Returns:
                    list: A list of selected individual indexes.
        """
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
        """
        Resets the tag set to a predefined default.
        
        This method is used to ensure a consistent and well-defined set of tags
        for processing selection constraints, providing a known starting point.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Attributes:
            _tags (set): A set containing the default tags: 'constraints', 'selection', 'custom level', and 'auxilary'.
        """
        self._tags = {'constraints', 'selection', 'custom level', 'auxilary'}