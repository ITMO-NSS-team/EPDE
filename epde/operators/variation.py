#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:43:19 2021

@author: mike_ubuntu
"""

from ast import operator
from operator import eq
import numpy as np
from copy import deepcopy

from functools import partial

from epde.structure.structure_template import check_uniqueness
from epde.moeadd.moeadd import ParetoLevels

from epde.supplementary import detect_similar_terms, flatten
from epde.decorators import History_Extender, Reset_equation_status

from epde.operators.template import CompoundOperator, add_param_to_operator
from epde.operators.moeadd_specific import get_basic_populator_updater
from epde.operators.mutations import get_basic_mutation


class ParetoLevelsCrossover(CompoundOperator):
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
    def apply(self, objective : ParetoLevels):
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
            if len(crossover_pool[pair_idx, 0].vals) != len(crossover_pool[pair_idx, 1].vals):
                raise IndexError('Equations have diffferent number of terms')
            new_system_1 = deepcopy(crossover_pool[pair_idx, 0])
            new_system_2 = deepcopy(crossover_pool[pair_idx, 1])

            new_system_1, new_system_2 = self.suboperators['chromosome_crossover'].apply(new_system_1, new_system_2)
            offsprings.extend([new_system_1, new_system_2])

        objective.unplaced_candidates = offsprings
        return objective

    def use_default_tags(self):
        self._tags = {'crossover', 'population level', 'contains suboperators', 'standard'}


class ChromosomeCrossover(CompoundOperator):
    def apply(self, objective : tuple):
        assert objective[0].vals.same_encoding(objective[1].vals)
        offspring_1 = deepcopy(objective[0]); offspring_2 = deepcopy(objective[1])        
        
        eqs_keys = offspring_1.vals.equation_keys; params_keys = offspring_2.vals.params_keys
        for eq_key in eqs_keys:
            offspring_1, offspring_2 = self.suboperators['equation_crossover'].apply((offspring_1.vals[eq_key],
                                                                                        offspring_2.vals[eq_key]))
            offspring_1.vals.replace_gene(gene_key = eq_key, value = offspring_1)
            offspring_2.vals.replace_gene(gene_key = eq_key, value = offspring_2)
            
        for param_key in params_keys:
            offspring_1, offspring_2 = self.suboperators['param_crossover'].apply((offspring_1.vals[param_key],
                                                                                   offspring_2.vals[param_key]))
            offspring_1.vals.replace_gene(gene_key = param_key, value = offspring_1)
            offspring_2.vals.replace_gene(gene_key = param_key, value = offspring_2)
            
            offspring_1.vals.pass_parametric_gene(key = param_key, value = offspring_1)
            offspring_2.vals.pass_parametric_gene(key = param_key, value = offspring_2)
            
        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'chromosome level', 'contains suboperators', 'standard'}


class MetaparamerCrossover(CompoundOperator):
    def apply(self, objective : tuple):
        offspring_1 = objective[0] + self.params['metaparam_proportion'] * (objective[1] - objective[0])
        offspring_2 = objective[0] + (1 - self.params['metaparam_proportion']) * (objective[1] - objective[0])
        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'gene level', 'no suboperators'}


class EquationCrossover(CompoundOperator):
    @Reset_equation_status(reset_input = True)
    @History_Extender(f'\n -> performing equation crossover', 'ba')
    def apply(self, objective : tuple):
        offspring_1 = deepcopy(objective[0]); offspring_2 = deepcopy(objective[1])
        
        equation1_terms, equation2_terms = detect_similar_terms(objective[0], objective[1])
        assert len(equation1_terms[0]) == len(equation2_terms[0]) and len(equation1_terms[1]) == len(equation2_terms[1])
        same_num = len(equation1_terms[0]); similar_num = len(equation1_terms[1])
        offspring_1.structure = flatten(equation1_terms); offspring_2.structure = flatten(equation2_terms)
    
        for i in range(same_num, same_num + similar_num):
            temp_term_1, temp_term_2 = self.suboperators['term_crossover'].apply((offspring_1.structure[i], offspring_2.structure[i])) 
            if (check_uniqueness(temp_term_1, offspring_1.structure[:i] + offspring_1.structure[i+1:]) and 
                check_uniqueness(temp_term_2, offspring_2.structure[:i] + offspring_2.structure[i+1:])):                     
                offspring_1.structure[i] = temp_term_1; offspring_2.structure[i] = temp_term_2

        for i in range(same_num + similar_num, len(offspring_1.structure)):
            if check_uniqueness(offspring_1.structure[i], offspring_2.structure) and check_uniqueness(offspring_2.structure[i], offspring_1.structure):
                offspring_1.structure[i], offspring_2.structure[i] = self.suboperators['term_crossover'].apply((offspring_1.structure[i], 
                                                                                                                offspring_2.structure[i]))
                
        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'gene level', 'contains suboperators', 'standard'}

class EquationExchangeCrossover(CompoundOperator):
    @Reset_equation_status(reset_input = True)
    @History_Extender(f'\n -> performing equation exchange crossover', 'ba')
    def apply(self, objective : tuple):
        offspring_1 = deepcopy(objective[0]); offspring_2 = deepcopy(objective[1])
        offspring_1.structure, offspring_2.structure = objective[1].structure, objective[0].structure
        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'gene level', 'contains suboperators', 'standard'}


class TermParamCrossover(CompoundOperator):
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
    def apply(self, objective : tuple):
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
        offspring_1 = deepcopy(objective[0]); offspring_2 = deepcopy(objective[1])
        offspring_1.reset_saved_state(); offspring_2.reset_saved_state()
        
        if len(offspring_1.structure) != len(offspring_2.structure):
            print([(token.label, token.params) for token in offspring_1.structure], [(token.label, token.params) for token in offspring_2.structure])
            raise Exception('Wrong terms passed:')
        for term1_token_idx in np.arange(len(objective[0].structure)):
            term2_token_idx = [i for i in np.arange(len(objective[1].structure)) 
                               if objective[1].structure[i].label == objective[0].structure[term1_token_idx].label][0]
            for param_idx, param_descr in offspring_1.structure[term1_token_idx].params_description.items():
                if param_descr['name'] == 'power': power_param_idx = param_idx
                if param_descr['name'] == 'dim': dim_param_idx = param_idx                

            for param_idx in np.arange(offspring_1.structure[term1_token_idx].params.size):
                if param_idx != power_param_idx and param_idx != dim_param_idx:
                    try:
                        offspring_1.structure[term1_token_idx].params[param_idx] = (offspring_1.structure[term1_token_idx].params[param_idx] + 
                                                                                    self.params['term_param_proportion'] 
                                                                                    * (offspring_2.structure[term2_token_idx].params[param_idx] 
                                                                                    - offspring_1.structure[term1_token_idx].params[param_idx]))
                    except KeyError:
                        print([(token.label, token.params) for token in offspring_1.structure], [(token.label, token.params) for token in offspring_2.structure])
                        raise Exception('Wrong set of parameters:', offspring_1.structure[term1_token_idx].params_description, offspring_2.structure[term1_token_idx].params_description)
                    offspring_2.structure[term2_token_idx].params[param_idx] = (offspring_1.structure[term1_token_idx].params[param_idx] + 
                                                                                (1 - self.params['term_param_proportion']) 
                                                                                * (offspring_2.structure[term2_token_idx].params[param_idx] 
                                                                                - offspring_1.structure[term1_token_idx].params[param_idx]))
        offspring_1.reset_occupied_tokens(); offspring_2.reset_occupied_tokens()
        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'term level', 'exploitation', 'no suboperators', 'standard'}

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
    def apply(self, objective : tuple):
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
        if (np.random.uniform(0, 1) <= self.params['crossover_probability'] and 
            objective[1].descr_variable_marker == objective[0].descr_variable_marker):
                return objective[1], objective[0]
        else:
                return objective[0], objective[1]
        
    def use_default_tags(self):
        self._tags = {'crossover', 'term level', 'exploration', 'no suboperators', 'standard'}


def get_basic_variation(variation_params : dict = {}):
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(add_param_to_operator, target_dict = variation_params)    

    term_param_crossover = TermParamCrossover(['term_param_proportion'])
    add_kwarg_to_operator(operator = term_param_crossover, labeled_base_val = {'term_param_proportion' : 0.4})
    term_crossover = TermCrossover(['crossover_probability'])
    add_kwarg_to_operator(operator = term_crossover, labeled_base_val = {'crossover_probability' : 0.3})

    equation_crossover = EquationCrossover()
    metaparameter_crossover = MetaparamerCrossover(['metaparam_proportion'])
    add_kwarg_to_operator(operator = metaparameter_crossover, labeled_base_val = {'term_param_proportion' : 0.4})
    equation_exchange_crossover = EquationExchangeCrossover()

    chromosome_crossover = ChromosomeCrossover()

    # chromosome_mutation = get_basic_mutation()
    # pl_updater = get_basic_populator_updater

    pl_cross = ParetoLevelsCrossover(['PBI_penalty'])
    add_kwarg_to_operator(operator = pl_cross, labeled_base_val = {'PBI_penalty' : 1.})

    equation_crossover.set_suboperators(operators = {'term_param_crossover' : term_param_crossover, 
                                                     'term_crossover' : term_crossover})
    chromosome_crossover.set_suboperators(operators = {'equation_crossover' : [equation_crossover, equation_exchange_crossover],
                                                       'param_crossover' : metaparameter_crossover},
                                          probas = {'equation_crossover' : [0.9, 0.1]})
    pl_cross.set_suboperators(operators = {'chromosome_crossover' : chromosome_crossover})
    return pl_cross