#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:43:19 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy

from epde.structure.structure_template import check_uniqueness
from epde.supplementary import detect_similar_terms, flatten
from epde.decorators import History_Extender, Reset_equation_status

from epde.operators.template import CompoundOperator
from epde.operators.supplementary_operators import PopulationUpdaterConstrained

class ParetoLevelCrossover(CompoundOperator):
    """
    The crossover operator, combining parameter crossover for terms with same 
    factors but different parameters & full exchange of terms between the 
    completely different ones.
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the Specific_Operator class. 
        Suboperators, performing tasks of parent selection, parameter crossover, full terms crossover, calculation of weights for each terms & 
        fitness function calculation. Dictionary: keys - strings from 'Selection', 'Param_crossover', 'Term_crossover', 'Coeff_calc', 'Fitness_eval'.
        values - corresponding operators (objects of Specific_Operator class).

    Methods:
    -----------
    apply(population)
        return the new population, created with the noted operators and containing both parent individuals and their offsprings.    
    
    """
    def apply(self, objective, separate_vars):
        """
        Method to obtain a new population by selection of parent individuals (equations) and performing a crossover between them to get the offsprings.
        
        Attributes:
        -----------
        population : list of Equation objects
            the population, to that the operator is applied;
            
        Returns:
        -----------
        population : list of Equation objects
            the new population, containing both parents and offsprings;
        
        """
        crossover_pool = []
        for solution in objective.population:
            crossover_pool.extend([solution,] * solution.crossover_times())

        if len(crossover_pool) == 0:
            raise ValueError('crossover pool not created, probably solution.crossover_selected_times error')
        np.random.shuffle(crossover_pool)
        crossover_pool = np.array(crossover_pool, dtype = object).reshape((-1,2))

        offsprings = []
        for pair_idx in np.arange(crossover_pool.shape[0]):
            if len(crossover_pool[pair_idx, 0].structure) != len(crossover_pool[pair_idx, 1].structure):
                raise IndexError('Equations have diffferent number of terms')
            new_equation_1 = deepcopy(crossover_pool[pair_idx, 0])
            new_equation_2 = deepcopy(crossover_pool[pair_idx, 1])

            new_equation_1, new_equation_2 = self.suboperators['system_crossover'].apply(new_equation_1, new_equation_2)
            offsprings.extend([new_equation_1, new_equation_2])

        for offspring in offsprings:
            objective = self.suboperators(offspring, objective, self.params['PBI_penalty'])
        return objective

    def use_default_tags(self):
        self._tags = {'crossover', 'population level', 'contains suboperators'}      


class SystemCrossover(CompoundOperator):
    def apply(self, objective_1, objective_2):
        assert objective_1.vals.same_encoding(objective_2.vals)
        
        eqs_keys = objective_1.vals.equation_keys; params_keys = objective_2.vals.params_keys
        for eq_key in eqs_keys:
            objective_1, objective_2= self.suboperators['equation_crossover'].apply(objective_1.vals[eq_key],
                                                                                    objective_2.vals[eq_key])
            objective_1.vals.replace_gene(gene_key = eq_key, value = objective_1)
            objective_2.vals.replace_gene(gene_key = eq_key, value = objective_2)
            
        for param_key in params_keys:
            objective_1, objective_2 = self.suboperators['param_crossover'].apply(objective_1.vals[param_key],
                                                                                  objective_2.vals[param_key])
            objective_1.vals.replace_gene(gene_key = param_key, value = objective_1)
            objective_2.vals.replace_gene(gene_key = param_key, value = objective_2)
        return objective_1, objective_2

    def use_default_tags(self):
        self._tags = {'crossover', 'system level', 'contains suboperators'}


class ParamsCrossover(CompoundOperator):
    def apply(self, objective_1, objective_2):
        offspring_1 = objective_1 + self.params['proportion'] * (objective_2 - objective_1)
        offspring_2 = objective_1 + (1 - self.params['proportion']) * (objective_2 - objective_1)
        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'custom level', 'no suboperators'}       


class EquationCrossover(CompoundOperator):
    @Reset_equation_status(reset_input = True)
    @History_Extender(f'\n -> performing equation crossover', 'ba')
    def apply(self, equation1, equation2, separate_vars):
        equation1_terms, equation2_terms = detect_similar_terms(equation1, equation2)
        assert len(equation1_terms[0]) == len(equation2_terms[0]) and len(equation1_terms[1]) == len(equation2_terms[1])
        same_num = len(equation1_terms[0]); similar_num = len(equation1_terms[1])
        equation1.structure = flatten(equation1_terms); equation2.structure = flatten(equation2_terms)
    
        for i in range(same_num, same_num + similar_num):
            temp_term_1, temp_term_2 = self.suboperators['param_crossover'].apply(equation1.structure[i], equation2.structure[i]) 
            if (check_uniqueness(temp_term_1, equation1.structure[:i] + equation1.structure[i+1:]) and 
                check_uniqueness(temp_term_2, equation2.structure[:i] + equation2.structure[i+1:])):                     
                equation1.structure[i] = temp_term_1; equation2.structure[i] = temp_term_2

        for i in range(same_num + similar_num, len(equation1.structure)):
            if check_uniqueness(equation1.structure[i], equation2.structure) and check_uniqueness(equation2.structure[i], equation1.structure):
                internal_term = equation1.structure[i]
                equation1.structure[i] = equation2.structure[i]
                equation2.structure[i] = internal_term
                temp_term_1, temp_term_2 = self.suboperators['term_crossover'].apply(equation1.structure[i], equation2.structure[i])

    def use_default_tags(self):
        self._tags = {'crossover', 'equation level', 'contains suboperators'}


class EqParamCrossover(CompoundOperator):
    """
    The crossover exchange between parent terms with the same factor functions, that differ only in the factor parameters. 

    Noteable attributes:
    -----------
    params : dict
        Inhereted from the Specific_Operator class. 
        Main key - 'proportion', value - proportion, in which the offsprings' parameter values are chosen.
        
    Methods:
    -----------
    apply(population)
        return the offspring terms, constructed as the parents' factors with parameter values, selected between the parents' ones.        
    """
    def apply(self, term_1, term_2):
        """
        Get the offspring terms, constructed as the parents' factors with parameter values, selected between the parents' ones.
        
        Attributes:
        ------------
        term_1, term_2 : Term objects
            The parent terms.
            
        Returns:
        ------------
        offspring_1, offspring_2 : Term objects
            The offspring terms.
        
        """
        offspring_1 = deepcopy(term_1)
        offspring_2 = deepcopy(term_2)
        offspring_1.reset_saved_state(); offspring_2.reset_saved_state()
        
        if len(offspring_1.structure) != len(offspring_2.structure):
            print([(token.label, token.params) for token in offspring_1.structure], [(token.label, token.params) for token in offspring_2.structure])
            raise Exception('Wrong terms passed:')
        for term1_token_idx in np.arange(len(term_1.structure)):
            term2_token_idx = [i for i in np.arange(len(term_2.structure)) if term_2.structure[i].label == term_1.structure[term1_token_idx].label][0]
            for param_idx, param_descr in offspring_1.structure[term1_token_idx].params_description.items():
                if param_descr['name'] == 'power': power_param_idx = param_idx
                if param_descr['name'] == 'dim': dim_param_idx = param_idx                

            for param_idx in np.arange(offspring_1.structure[term1_token_idx].params.size):
                if param_idx != power_param_idx and param_idx != dim_param_idx:
                    try:
                        offspring_1.structure[term1_token_idx].params[param_idx] = (term_1.structure[term1_token_idx].params[param_idx] + 
                                                                   self.params['proportion']*(term_2.structure[term2_token_idx].params[param_idx] 
                                                                   - term_1.structure[term1_token_idx].params[param_idx]))
                    except KeyError:
                        print([(token.label, token.params) for token in offspring_1.structure], [(token.label, token.params) for token in offspring_2.structure])
                        raise Exception('Wrong set of parameters:', offspring_1.structure[term1_token_idx].params_description, offspring_2.structure[term1_token_idx].params_description)
                    offspring_2.structure[term2_token_idx].params[param_idx] = (term_1.structure[term1_token_idx].params[param_idx] + 
                                                               (1 - self.params['proportion'])*(term_2.structure[term2_token_idx].params[param_idx] 
                                                               - term_1.structure[term1_token_idx].params[param_idx]))
        offspring_1.reset_occupied_tokens(); offspring_2.reset_occupied_tokens()
        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'term level', 'exploitation', 'no suboperators'}

class TermCrossover(CompoundOperator):
    """
    The crossover exchange between parent terms, done by complete exchange of terms. 

    Noteable attributes:
    -----------
    params : dict
        Inhereted from the Specific_Operator class. 
        Main key - 'crossover_probability', value - probabilty of the term exchange.
        
    Methods:
    -----------
    apply(population)
        return the offspring terms, which are the same parents' ones, but in different order, if the crossover occured.
        .        
    """    
    def apply(self, term_1, term_2):
        """
        Get the offspring terms, which are the same parents' ones, but in different order, if the crossover occured.
        
        Attributes:
        ------------
        term_1, term_2 : Term objects
            The parent terms.
            
        Returns:
        ------------
        offspring_1, offspring_2 : Term objects
            The offspring terms.
        
        """        
        if np.random.uniform(0, 1) <= self.params['crossover_probability']:
            return term_2, term_1
        else:
            return term_1, term_2
        
    def use_default_tags(self):
        self._tags = {'crossover', 'term level', 'exploration', 'no suboperators'}