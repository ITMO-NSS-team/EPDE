#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:46:31 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy

from epde.structure.main_structures import Equation, SoEq, Term
from epde.structure.structure_template import check_uniqueness
from epde.supplementary import filter_powers, try_iterable
from epde.operators.template import CompoundOperator


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
    def apply(self, objective : SoEq): # TODO: add setter for best_individuals & worst individuals 
        eqs_keys = objective.vals.equation_keys; params_keys = objective.vals.params_keys
        if np.random.uniform(0, 1) > self.params['indiv_mutation_prob'] and objective.elite == 0:
            for eq_key in eqs_keys:
                objective = self.suboperators['equation_mutation'].apply(objective.vals[eq_key])
                objective.vals.replace_gene(gene_key = eq_key, value = objective)
                
            for param_key in params_keys:
                objective = self.suboperators['param_mutation'].apply(objective.vals[param_key])
                objective.vals.replace_gene(gene_key = param_key, value = objective)

        if np.random.uniform(0, 1) > self.params['indiv_mutation_prob'] and objective.elite == 1:            
            for eq_key in eqs_keys:
                if validate_indiviual_suitability(objective[objective.vals[eq_key]].evaluate(False, True, True), # Call by index?
                                                  self.params['best_individuals'][eq_key],
                                                  self.params['worst_individuals'][eq_key]):
                    objective = self.suboperators['refining_equation_mutation'].apply(objective.vals[eq_key])
                    objective.vals.replace_gene(gene_key = eq_key, value = objective)

            for param_key in params_keys:
                objective = self.suboperators['refining_param_mutation'].apply(objective.vals[param_key])
                objective.vals.replace_gene(gene_key = param_key, value = objective)


    def use_default_tags(self):
        self._tags = {'mutation', 'chromosome level', 'contains suboperators'}


class RefiningEquationMutation(CompoundOperator):
    @property
    def elitist(self):
        return True
    
    @Reset_equation_status(reset_input = True)
    @History_Extender(f'\n -> refining mutating equation', 'ba')
    def apply(self, objective : Equation):
        for term_idx in range(objective.n_immutable, len(objective.structure)):
            if term_idx == objective.target_idx:
                continue
            corresponding_weight = objective.weights_internal[term_idx] if term_idx < objective.target_idx else objective.weights_internal[term_idx - 1]
            if corresponding_weight == 0 and np.random.uniform(0, 1) <= self.params['r_mutation']:
                self.params['type_probabilities'] = [1 - 1/pow(objective.structure[term_idx].total_params, 2), 1/pow(objective.structure[term_idx].total_params, 2)]
                if try_iterable(self.suboperators['Mutation']):
                    mut_operator = np.random.choice(self.suboperators['Mutation'], p=self.params['type_probabilities'])
                else:
                    mut_operator = self.suboperators['Mutation']
                objective.structure[term_idx] = mut_operator.apply(term_idx, objective)

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'elitist', 'contains suboperators'}
    

class EquationMutation(CompoundOperator):
    @property
    def elitist(self):
        return True

    @Reset_equation_status(reset_input = True)
    @History_Extender(f'\n -> mutating equation', 'ba')
    def apply(self, equation):
        for term_idx in range(equation.n_immutable, len(equation.structure)):
            if np.random.uniform(0, 1) <= self.params['r_mutation']:
                self.params['type_probabilities'] = [1 - 1/pow(equation.structure[term_idx].total_params, 2), 1/pow(equation.structure[term_idx].total_params, 2)]
                if try_iterable(self.suboperators['Mutation']):
                    mut_operator = np.random.choice(self.suboperators['Mutation'], p=self.params['type_probabilities'])
                else:
                    mut_operator = self.suboperators['Mutation']
                equation.structure[term_idx] = mut_operator.apply(term_idx, equation)

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'contains suboperators'}        
        
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
        if term_idx == equation.target_idx: # TODO: implement mutation, that preserves meaningful family?
            new_term = Term(equation.pool, max_factors_in_term = equation.max_factors_in_term)
            while not check_uniqueness(new_term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
                new_term = Term(equation.pool, max_factors_in_term = equation.max_factors_in_term)
            new_term.use_cache()
            return new_term
        else:            
            new_term = Term(equation.pool, max_factors_in_term = equation.max_factors_in_term)        #) #
            while not check_uniqueness(new_term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
                new_term = Term(equation.pool, max_factors_in_term = equation.max_factors_in_term)
            new_term.use_cache()
            return new_term

    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploration', 'no suboperators'}    


class TermParameterMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term parameters are changed with a random increment.
    """
    def apply(self, term_idx, equation):
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
        while True:
            term = equation.structure[term_idx] 
            for factor in term.structure:
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
            if check_uniqueness(term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
                break
        term.reset_saved_state()
        return term
    
    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploitation', 'no suboperators'}