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

        while not (objective.simplified and objective.is_correct_right_part):
            objective.reset_state(True)
            min_fitness = np.inf
            weights_internal = np.zeros(len(objective.structure) - 1)
            min_idx = 0
            while not any(term.contains_deriv(objective.main_var_to_explain) for term in objective.structure):
            # while not any(term.contains_deriv() for term in objective.structure):
                objective.restore_property(mandatory_family=False, deriv=True)
                
            for target_idx, target_term in enumerate(objective.structure):
                if not objective.structure[target_idx].contains_deriv(objective.main_var_to_explain):
                # if not objective.structure[target_idx].contains_deriv():
                    continue
                objective.target_idx = target_idx
                fitness = self.suboperators['fitness_calculation'].apply(objective, arguments = subop_args['fitness_calculation'], force_out_of_place = True)
                if fitness < min_fitness and not all(objective.weights_internal == 0):
                    min_fitness = fitness
                    min_idx = target_idx
                    weights_internal = objective.weights_internal
                else:
                    pass

            if all(weights_internal == 0) or np.isinf(min_fitness):
                objective.randomize()
                continue

            objective.weights_internal = weights_internal
            objective.weights_internal_evald = True
            objective.target_idx = min_idx
            if not self.simplify_equation(objective):
                objective.simplified = True
            if objective.structure[objective.target_idx].contains_deriv(objective.main_var_to_explain):
            # if objective.structure[objective.target_idx].contains_deriv():
                objective.is_correct_right_part = True
        else:
            objective.right_part_selected = True
            objective.reset_state(False)

    def simplify_equation(self, objective: Equation):
        # Get nonzero terms
        nonzero_terms_mask = np.array([False if weight == 0 else True for weight in objective.weights_internal], dtype=np.int32)
        nonrs_terms = [term for i, term in enumerate(objective.structure) if i != objective.target_idx]
        nonzero_terms = [item for item, keep in zip(nonrs_terms, nonzero_terms_mask) if keep]
        nonzero_terms.append(objective.structure[objective.target_idx])

        equation_terms = objective.described_variables
        # If amount nonzero terms is more than one -- get their intersection
        if len(equation_terms) > 1:
            common_factors = list(frozenset.intersection(*equation_terms))
            if len(common_factors) > 0:
                for common_factor in common_factors:
                    # Find if this intersection in the same dimension (i.e. trigonometry functions) + it's minimal order
                    min_order = np.inf
                    common_dim = []
                    for term in nonzero_terms:
                        for factor in term.structure:
                            if len(factor.params) == 1:
                                factor_label = (factor.cache_label[0])
                            else:
                                factor_label = (factor.cache_label[0], (factor.cache_label[1][-1]))
                            if factor_label == common_factor:
                                if len(factor.params) > 1:
                                    common_dim.append(factor.params[-1])
                                if factor.cache_label[1][0] < min_order:
                                    min_order = factor.cache_label[1][0]
                    if len(set(common_dim)) < 2:
                        # If dimension is the same -- reduce order of terms' factor
                        for term in nonzero_terms:
                            factors_simplified = []
                            for factor in term.structure:
                                if len(factor.params) == 1:
                                    factor_label = (factor.cache_label[0])
                                else:
                                    factor_label = (factor.cache_label[0], (factor.cache_label[1][-1]))
                                if factor_label == common_factor:
                                    for i, value in enumerate(factor.params_description):
                                        if factor.params_description[i]["name"] == "power":
                                            factor.params[i] -= min_order
                                            if factor.params[i] == 0:
                                                factors_simplified.append(factor)
                                        else:
                                            continue
                            term.structure = [factor for factor in term.structure if factor not in factors_simplified]
                            term.reset_saved_state()
                            # If term's order became zero -- replace term
                            if len(term.structure) == 0 or not term.contains_meaningful():
                                term.randomize()
                                term.reset_saved_state()
                            while len(objective.described_variables_full) != len(objective.structure):
                                term.randomize()
                                term.reset_saved_state()
                        return True
        return False

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
                              if term.contains_deriv(variable = objective.main_var_to_explain)]
            
            if len(term_selection) == 0:
                idx = np.random.choice([term_idx for term_idx, _ in enumerate(objective.structure)])
                prev_term = objective.structure[idx]
                while True:
                    candidate_term = Term(pool = prev_term.pool, mandatory_family = objective.main_var_to_explain,
                                          max_factors_in_term = len(prev_term.structure), 
                                          create_derivs = True)
                    if candidate_term.contains_deriv(variable = objective.main_var_to_explain):
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
