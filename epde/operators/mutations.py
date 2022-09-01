#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:46:31 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
from functools import partial

from epde.moeadd.moeadd import ParetoLevels

from epde.structure.main_structures import Equation, SoEq, Term
from epde.structure.structure_template import check_uniqueness
from epde.supplementary import filter_powers, try_iterable
from epde.operators.template import CompoundOperator, add_param_to_operator


from epde.decorators import History_Extender, Reset_equation_status

# Pareto level and chromosome processers shall be declare as mappings of 
# equation-level operators onto the corresponding types.  


def validate_indiviual_suitability(individual, best_inidividual, worst_individual):
    '''
    We assert, that the problem is the problem of minimization
    '''
    elite_fraction = 0.2 # TODO: set better initialization
    return individual < best_inidividual + elite_fraction * (worst_individual - best_inidividual)


class ChromosomeMutation(CompoundOperator):
    def apply(self, objective : SoEq, best_inidividuals, worst_individuals): # TODO: add setter for best_individuals & worst individuals 
        altered_objective = deepcopy(objective)
        eqs_keys = altered_objective.vals.equation_keys; params_keys = altered_objective.vals.params_keys
        affected_by_mutation = np.random.uniform(0, 1) > self.params['indiv_mutation_prob']
    
        if affected_by_mutation and altered_objective.elite == 0:
            for eq_key in eqs_keys:
                altered_objective = self.suboperators['equation_mutation'].apply(altered_objective.vals[eq_key])
                altered_objective.vals.replace_gene(gene_key = eq_key, value = altered_objective)
                
        if affected_by_mutation > self.params['indiv_mutation_prob'] and altered_objective.elite == 1:          
            for eq_key in eqs_keys:
                if validate_indiviual_suitability(altered_objective[altered_objective.vals[eq_key]].evaluate(False, True, True), # Call by index?
                                                  best_inidividuals[eq_key],
                                                  worst_individuals[eq_key]):
                    altered_objective = self.suboperators['refining_equation_mutation'].apply(altered_objective.vals[eq_key])
                    altered_objective.vals.replace_gene(gene_key = eq_key, value = altered_objective)

        if affected_by_mutation and (altered_objective.elite == 0 or altered_objective.elite == 1):
            for param_key in params_keys:
                altered_objective = self.suboperators['param_mutation'].apply(altered_objective.vals[param_key])
                altered_objective.vals.replace_gene(gene_key = param_key, value = altered_objective)

        return altered_objective

    def use_default_tags(self):
        self._tags = {'mutation', 'chromosome level', 'contains suboperators'}


class RefiningEquationMutation(CompoundOperator):
    @property
    def elitist(self):
        return True
    
    @Reset_equation_status(reset_input = True)
    @History_Extender(f'\n -> refining mutating equation', 'ba')
    def apply(self, objective : Equation):
        altered_objective = deepcopy(objective)
        for term_idx in range(altered_objective.n_immutable, len(altered_objective.structure)):
            if term_idx == altered_objective.target_idx:
                continue
            if term_idx < altered_objective.target_idx:
                corresponding_weight = altered_objective.weights_internal[term_idx] 
            else:
                corresponding_weight = altered_objective.weights_internal[term_idx - 1]
            if corresponding_weight == 0 and np.random.uniform(0, 1) <= self.params['r_mutation']:
                self.params['type_probabilities'] = [1 - 1/pow(altered_objective.structure[term_idx].total_params, 2), 1/pow(altered_objective.structure[term_idx].total_params, 2)]
                if try_iterable(self.suboperators['mutation']):
                    mut_operator = np.random.choice(self.suboperators['mutation'], p=self.params['type_probabilities'])
                else:
                    mut_operator = self.suboperators['mutation']
                altered_objective.structure[term_idx] = mut_operator.apply(term_idx, altered_objective)
        return altered_objective

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'elitist', 'contains suboperators'}
    

class EquationMutation(CompoundOperator):
    @property
    def elitist(self):
        return True

    @Reset_equation_status(reset_input = True)
    @History_Extender(f'\n -> mutating equation', 'ba')
    def apply(self, objective : Equation):
        altered_objective = deepcopy(objective)
        for term_idx in range(objective.n_immutable, len(objective.structure)):
            if np.random.uniform(0, 1) <= self.params['r_mutation']:
                # self.params['type_probabilities'] = [1 - 1/pow(objective.structure[term_idx].total_params, 2),
                #                                      1/pow(objective.structure[term_idx].total_params, 2)]
                # if try_iterable(self.suboperators['mutation']):
                #     mut_operator = np.random.choice(self.suboperators['mutation'], p=self.params['type_probabilities'])
                # else:
                #     mut_operator = self.suboperators['mutation']
                altered_objective.structure[term_idx] = self.suboperators['mutation'].apply(term_idx, altered_objective)
        return altered_objective
    
    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'contains suboperators'}        


class MetaparameterMutation(CompoundOperator):
    def apply(self, objective):
        altered_objective = objective + np.random.normal(loc = self.params['mean'], scale = self.params['std'])        
        return altered_objective

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'no suboperators'}

    
class TermMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term is replaced with a randomly created new one.
    """
    def apply(self, term_idx, equation):
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
        new_term = Term(equation.pool, mandatory_family = equation[term_idx].descr_variable_marker, 
                        max_factors_in_term = equation.max_factors_in_term)
        while not check_uniqueness(new_term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
            new_term = Term(equation.pool, mandatory_family = equation[term_idx].descr_variable_marker, 
                            max_factors_in_term = equation.max_factors_in_term)
        new_term.use_cache()
        return new_term
    
    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploration', 'no suboperators'}    


class TermParameterMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term parameters are changed with a random increment.
    """
    def apply(self, term_idx, objective):
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
        altered_objective = deepcopy(objective)
        while True:
            term = altered_objective.structure[term_idx] 
            for factor in term.structure:
                if term_idx == altered_objective.target_idx:
                    continue
                if term_idx < altered_objective.target_idx:
                    corresponding_weight = altered_objective.weights_internal[term_idx] 
                else:
                    corresponding_weight = altered_objective.weights_internal[term_idx - 1]
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
            if check_uniqueness(term, altered_objective.structure[:term_idx] + 
                                altered_objective.structure[term_idx+1:]):
                break
        term.reset_saved_state()
        return term
    
    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploitation', 'no suboperators'}


def get_basic_mutation(**kwargs):
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(func = add_param_to_operator, target_dict = kwargs)    

    term_mutation = TermMutation([])
    term_param_mutation = TermParameterMutation(['r_param_mutation', 'multiplier'])
    add_kwarg_to_operator(term_param_mutation, {'r_param_mutation' : 0.2, 'strict_restrictions' : True,
                                                'multiplier' : 0.1})

    equation_mutation = EquationMutation(['r_mutation', 'type_probabilities'])
    add_kwarg_to_operator(equation_mutation, {'r_mutation' : 0.3, 'type_probabilities' : []})
    refining_equation_mutation = RefiningEquationMutation([])
    add_kwarg_to_operator(refining_equation_mutation, {'r_mutation' : 0.5, 'type_probabilities' : []})
    
    metaparameter_mutation = MetaparameterMutation(['std', 'mean'])
    add_kwarg_to_operator(metaparameter_mutation, {'std' : 0, 'mean' : 0.4})

    chromosome_mutation = ChromosomeMutation(['indiv_mutation_prob'])
    add_kwarg_to_operator(chromosome_mutation, {'indiv_mutation_prob' : 0.5})
    chromosome_mutation.params = {'indiv_mutation_prob' : 0.5} if not 'mutation_params' in kwargs.keys() else kwargs['mutation_params']

    equation_mutation.set_suboperators(operators = {'mutation' : [term_param_mutation, term_mutation]},
                                       probas = {'equation_crossover' : [0.9, 0.1]})
    refining_equation_mutation.set_suboperators(operators = {'mutation' : [term_param_mutation, term_mutation]},
                                                probas = {'equation_crossover' : [0.9, 0.1]})

    chromosome_mutation.set_suboperators(operators = {'equation_mutation' : equation_mutation, 
                                                      'refining_equation_mutation' : refining_equation_mutation, 
                                                      'param_mutation' : metaparameter_mutation})
    return chromosome_mutation