#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:46:31 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
from functools import partial
from typing import Union

from epde.moeadd.moeadd import ParetoLevels

from epde.structure.main_structures import Equation, SoEq, Term
from epde.structure.structure_template import check_uniqueness
from epde.supplementary import filter_powers, try_iterable
from epde.operators.template import CompoundOperator, add_param_to_operator


from epde.decorators import History_Extender, Reset_equation_status


class SystemMutation(CompoundOperator):
    def apply(self, objective : SoEq): # TODO: add setter for best_individuals & worst individuals 
        altered_objective = deepcopy(objective)
        eqs_keys = altered_objective.vals.equation_keys; params_keys = altered_objective.vals.params_keys
        affected_by_mutation = np.random.uniform(0, 1) > self.params['indiv_mutation_prob']
    
        if affected_by_mutation:
            for eq_key in eqs_keys:
                altered_objective = self.suboperators['equation_mutation'].apply(altered_objective.vals[eq_key])
                altered_objective.vals.replace_gene(gene_key = eq_key, value = altered_objective)

            for param_key in params_keys:
                altered_objective = self.suboperators['param_mutation'].apply(altered_objective.vals[param_key])
                altered_objective.vals.replace_gene(gene_key = param_key, value = altered_objective)
                altered_objective.vals.pass_parametric_gene(key = param_key, value = altered_objective)

        return altered_objective

    def use_default_tags(self):
        self._tags = {'mutation', 'chromosome level', 'contains suboperators'}
    

class EquationMutation(CompoundOperator):
    @Reset_equation_status(reset_input = True)
    @History_Extender(f'\n -> mutating equation', 'ba')
    def apply(self, objective : Equation):
        altered_objective = deepcopy(objective)
        for term_idx in range(objective.n_immutable, len(objective.structure)):
            if np.random.uniform(0, 1) <= self.params['r_mutation']:
                altered_objective.structure[term_idx] = self.suboperators['mutation'].apply(term_idx, altered_objective)
        return altered_objective

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'contains suboperators'}


class MetaparameterMutation(CompoundOperator):
    def apply(self, objective : Union[int, float]):
        altered_objective = objective + np.random.normal(loc = self.params['mean'], scale = self.params['std'])        
        return altered_objective

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'no suboperators'}

    
class TermMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term is replaced with a randomly created new one.
    """
    def apply(self, objective : tuple): #term_idx, equation):
        """
        Return a new term, randomly created to be unique from other terms of this particular equation.
        
        Parameters:
        -----------
        term_idx : integer
            The index of the mutating term in the equation.
            
        equation : Equation object
            The equation object, in which the term is present.
        
        Returns:
        ----------
        new_term : Term object
            A new, randomly created, term.
            
        """       
        new_term = Term(objective[1].pool, mandatory_family = objective[1][objective[0]].descr_variable_marker, 
                        max_factors_in_term = objective[1].max_factors_in_term)
        while not check_uniqueness(new_term, objective[1].structure[:objective[0]] + objective[1].structure[objective[0]+1:]):
            new_term = Term(objective[1].pool, mandatory_family = objective[1][objective[0]].descr_variable_marker, 
                            max_factors_in_term = objective[1].max_factors_in_term)
        new_term.use_cache()
        return new_term
    
    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploration', 'no suboperators'}    


class TermParameterMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term parameters are changed with a random increment.
    """
    def apply(self, objective : tuple): # term_idx, objective
        """ 
        Specific operator of the term mutation, where the term parameters are changed with a random increment.
        
        Parameters:
        -----------
        term_idx : integer
            The index of the mutating term in the equation.
            
        equation : Equation object
            The equation object, in which the term is present.
        
        Returns:
        ----------
        new_term : Term object
            The new, created from the previous one with random parameters increment, term.
            
        """                
        unmutable_params = {'dim', 'power'}
        altered_objective = deepcopy(objective[1])
        while True:
            term = altered_objective.structure[objective[0]] 
            for factor in term.structure:
                if objective[0] == altered_objective.target_idx:
                    continue
                if objective[0] < altered_objective.target_idx:
                    corresponding_weight = altered_objective.weights_internal[objective[0]] 
                else:
                    corresponding_weight = altered_objective.weights_internal[objective[0] - 1]
                if corresponding_weight == 0:                
                    parameter_selection = deepcopy(factor.params)
                    for param_idx, param_properties in factor.params_description.items():
                        if np.random.random() < self.params['r_param_mutation'] and param_properties['name'] not in unmutable_params:
                            interval = param_properties['bounds']
                            if interval[0] == interval[1]:
                                shift = 0
                                continue
                            if isinstance(interval[0], int):
                                shift = np.rint(np.random.normal(loc= 0, scale = self.params['multiplier']*(interval[1] - interval[0]))).astype(int) #
                            elif isinstance(interval[0], float):
                                shift = np.random.normal(loc= 0, scale = self.params['multiplier']*(interval[1] - interval[0]))
                            else:
                                raise ValueError('In current version of framework only integer and real values for parameters are supported') 
                            if self.params['strict_restrictions']:
                                parameter_selection[param_idx] = np.min((np.max((parameter_selection[param_idx] + shift, interval[0])), interval[1]))
                            else:
                                parameter_selection[param_idx] = parameter_selection[param_idx] + shift
                    factor.params = parameter_selection
            term.structure = filter_powers(term.structure)        
            if check_uniqueness(term, altered_objective.structure[:objective[0]] + 
                                altered_objective.structure[objective[0]+1:]):
                break
        term.reset_saved_state()
        return term
    
    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploitation', 'no suboperators'}


def get_basic_mutation(mutation_params):
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(add_param_to_operator, target_dict = mutation_params)    

    term_mutation = TermMutation([])
    term_param_mutation = TermParameterMutation(['r_param_mutation', 'multiplier'])
    add_kwarg_to_operator(operator = term_param_mutation, labeled_base_val = {'r_param_mutation' : 0.2, 
                                                                              'strict_restrictions' : True,
                                                                              'multiplier' : 0.1})

    equation_mutation = EquationMutation(['r_mutation', 'type_probabilities'])
    add_kwarg_to_operator(operator = equation_mutation, labeled_base_val = {'r_mutation' : 0.3, 'type_probabilities' : []})
    
    metaparameter_mutation = MetaparameterMutation(['std', 'mean'])
    add_kwarg_to_operator(operator = metaparameter_mutation, labeled_base_val = {'std' : 0, 'mean' : 0.4})

    chromosome_mutation = SystemMutation(['indiv_mutation_prob'])
    add_kwarg_to_operator(operator = chromosome_mutation, labeled_base_val = {'indiv_mutation_prob' : 0.5})
    # chromosome_mutation.params = {'indiv_mutation_prob' : 0.5} if not 'mutation_params' in kwargs.keys() else kwargs['mutation_params']

    equation_mutation.set_suboperators(operators = {'mutation' : [term_param_mutation, term_mutation]},
                                       probas = {'equation_crossover' : [0.9, 0.1]})

    chromosome_mutation.set_suboperators(operators = {'equation_mutation' : equation_mutation, 
                                                      'param_mutation' : metaparameter_mutation})
    return chromosome_mutation