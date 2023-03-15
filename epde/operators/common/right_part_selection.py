#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:50:55 2021

@author: mike_ubuntu
"""
import time

import numpy as np
from copy import deepcopy
import warnings

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.decorators import HistoryExtender
from epde.structure.main_structures import Term, Equation
    
class EqRightPartSelector(CompoundOperator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the CompoundOperator class
        key - str, value - instance of a class, inhereted from the CompoundOperator. 
        Suboperators, performing tasks of equation processing. In this case, only one suboperator is present: 
        fitness_calculation, dedicated to calculation of fitness function value.

    Methods:
    -----------
    apply(equation)
        return None
        Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``

    
    '''    
    key = 'FitnessCheckingRightPartSelector'

    @HistoryExtender('\n -> The equation structure was detected: ', 'a')        
    def apply(self, objective : Equation, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        if not objective.right_part_selected:
            max_fitness = 0
            max_idx = 0
            if not objective.contains_deriv():
                objective.restore_property(deriv = True)
            if not objective.contains_family(objective.main_var_to_explain):
                objective.restore_property(mandatory_family = True)
                
            
                
            for target_idx, target_term in enumerate(objective.structure):
                if target_term.contains_family(objective.main_var_to_explain): #target_term.descr_variable_marker:
                    # target_term.contains_family(equation.main_var_to_explain)
                    if not objective.structure[target_idx].contains_deriv:
                        continue
                    objective.target_idx = target_idx
                    self.suboperators['fitness_calculation'].apply(objective, arguments = subop_args['fitness_calculation'])
                    if objective.fitness_value > max_fitness:
                        max_fitness = objective.fitness_value
                        max_idx = target_idx
                    else:
                        pass

            objective.target_idx = max_idx
            objective.reset_explaining_term(objective.target_idx)
            self.suboperators['fitness_calculation'].apply(objective, arguments = subop_args['fitness_calculation'])
            if not np.isclose(objective.fitness_value, max_fitness) and global_var.verbose.show_warnings:
                warnings.warn('Reevaluation of fitness function for equation has obtained different result. Not an error, if ANN DE solver is used.')
            objective.right_part_selected = True

    def use_default_tags(self):
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}
        
class RandomRHPSelector(CompoundOperator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the CompoundOperator class
        key - str, value - instance of a class, inhereted from the CompoundOperator. 
        Suboperators, performing tasks of equation processing. In this case, only one suboperator is present: 
        fitness_calculation, dedicated to calculation of fitness function value.

    Methods:
    -----------
    apply(equation)
        return None
        Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``

    
    '''
    key = 'RandomRightPartSelector'

    @HistoryExtender('\n -> The equation structure was detected: ', 'a')
    def apply(self, objective : Equation, arguments : dict):
        # print(f'CALLING RIGHT PART SELECTOR FOR {objective.text_form}')
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        if not objective.right_part_selected:
            term_selection = [term_idx for term_idx, term in enumerate(objective.structure)
                              if term.contains_deriv(family = objective.main_var_to_explain)]
            
            if len(term_selection) == 0:
                idx = np.random.choice([term_idx for term_idx, _ in enumerate(objective.structure)])
                prev_term = objective.structure[idx]
                while True:
                    candidate_term = Term(pool = prev_term.pool, mandatory_family = objective.main_var_to_explain,
                                          max_factors_in_term = len(prev_term.structure), 
                                          create_derivs = True)
                    if candidate_term.contains_deriv(family = objective.main_var_to_explain):
                        break
                
                objective.structure[idx] = candidate_term
            else:
                idx = np.random.choice(term_selection)

            objective.target_idx = idx
            # print('Selected right part term', objective.structure[idx].name)
            objective.reset_explaining_term(idx)
            objective.right_part_selected = True


    def use_default_tags(self):
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}
