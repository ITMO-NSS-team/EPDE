#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:43:19 2021

@author: mike_ubuntu
"""

from operator import eq
import numpy as np
from copy import deepcopy

from typing import Union

from functools import partial

from epde.structure.structure_template import check_uniqueness
from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.optimizers.single_criterion.optimizer import Population

from epde.supplementary import detect_similar_terms, flatten
from epde.decorators import HistoryExtender, ResetEquationStatus

from epde.operators.utils.template import CompoundOperator, add_base_param_to_operator
from epde.operators.multiobjective.moeadd_specific import get_basic_populator_updater
from epde.operators.multiobjective.mutations import get_basic_mutation


class PopulationLevelCrossover(CompoundOperator):
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
    copy_properties_to
    """
    key = 'PopulationLevelCrossover'

    def apply(self, objective : Union[ParetoLevels, Population], arguments : dict):
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
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)        
        
        crossover_pool = []
        for solution in objective.population:
            crossover_pool.extend([solution,] * solution.crossover_times())
            solution.reset_counter()

        if len(crossover_pool) == 0:
            raise ValueError('crossover pool not created, probably solution.crossover_selected_times error')
        np.random.shuffle(crossover_pool)
        if len(crossover_pool) % 2:
            crossover_pool = crossover_pool[:-1]
        crossover_pool = np.array(crossover_pool, dtype = object).reshape((-1,2))

        offsprings = []
        for pair_idx in np.arange(crossover_pool.shape[0]):
            if len(crossover_pool[pair_idx, 0].vals) != len(crossover_pool[pair_idx, 1].vals):
                raise IndexError('Equations have diffferent number of terms')
            new_system_1 = deepcopy(crossover_pool[pair_idx, 0])
            new_system_2 = deepcopy(crossover_pool[pair_idx, 1])
            new_system_1.reset_state(); new_system_2.reset_state()
            
            new_system_1, new_system_2 = self.suboperators['chromosome_crossover'].apply(objective = (new_system_1, new_system_2), 
                                                                                         arguments = subop_args['chromosome_crossover'])
            offsprings.extend([new_system_1, new_system_2])

        # TODO: maybe a better operator will fit here?
        if isinstance(objective, ParetoLevels):
            objective.unplaced_candidates = offsprings
        elif isinstance(objective, Population):
            objective.population.extend(offsprings)
        return objective

    def use_default_tags(self):
        self._tags = {'crossover', 'population level', 'contains suboperators', 'standard'}


class ChromosomeCrossover(CompoundOperator):
    key = 'ChromosomeCrossover'

    def apply(self, objective : tuple, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
   
        assert objective[0].vals.same_encoding(objective[1].vals)
        # offspring_1 = deepcopy(objective[0]); offspring_2 = deepcopy(objective[1])      
        offspring_1 = objective[0]; offspring_2 = objective[1]
                
        eqs_keys = objective[0].vals.equation_keys; params_keys = objective[1].vals.params_keys
        for eq_key in eqs_keys:
            temp_eq_1, temp_eq_2 = self.suboperators['equation_crossover'].apply(objective = (objective[0].vals[eq_key],
                                                                                              objective[1].vals[eq_key]),
                                                                                 arguments = subop_args['equation_crossover'])
            # print(f'PARENT 1: objective[0].vals[eq_key] is {objective[0].vals[eq_key].text_form}')
            # print(f'PARENT 2: objective[1].vals[eq_key] is {objective[1].vals[eq_key].text_form}')            
            # print(f'OFFSPRING: temp_eq_1.vals[eq_key] is {temp_eq_1.text_form}')
            objective[0].vals.replace_gene(gene_key = eq_key, value = temp_eq_1)
            offspring_2.vals.replace_gene(gene_key = eq_key, value = temp_eq_2)
            
        for param_key in params_keys:
            temp_param_1, temp_param_2 = self.suboperators['param_crossover'].apply(objective = (objective[0].vals[param_key],
                                                                                                 objective[1].vals[param_key]),
                                                                                    arguments = subop_args['param_crossover'])
            objective[0].vals.replace_gene(gene_key = param_key, value = temp_param_1)
            objective[1].vals.replace_gene(gene_key = param_key, value = temp_param_2)

            objective[0].vals.pass_parametric_gene(key = param_key, value = temp_param_1)
            objective[1].vals.pass_parametric_gene(key = param_key, value = temp_param_2)

        # print(f'OFFSPRING CROSSOVER: {[ind.text_form for ind in objective[0].vals]}')
        # print(f'OFFSPRING CROSSOVER: {[ind.text_form for ind in objective[1].vals]}')
        return objective[0], objective[1]

    def use_default_tags(self):
        self._tags = {'crossover', 'chromosome level', 'contains suboperators', 'standard'}


class MetaparamerCrossover(CompoundOperator):
    key = 'MetaparamerCrossover'

    def apply(self, objective : tuple, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        offspring_1 = objective[0] + self.params['metaparam_proportion'] * (objective[1] - objective[0])
        offspring_2 = objective[0] + (1 - self.params['metaparam_proportion']) * (objective[1] - objective[0])
        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'gene level', 'no suboperators'}


class EquationCrossover(CompoundOperator):
    key = 'EquationCrossover'
    
    @HistoryExtender(f'\n -> performing equation crossover', 'ba')
    def apply(self, objective : tuple, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        equation1_terms, equation2_terms = detect_similar_terms(objective[0], objective[1])
        assert len(equation1_terms[0]) == len(equation2_terms[0]) and len(equation1_terms[1]) == len(equation2_terms[1])
        same_num = len(equation1_terms[0]); similar_num = len(equation1_terms[1])
        objective[0].structure = flatten(equation1_terms); objective[1].structure = flatten(equation2_terms)
    
        for i in range(same_num, same_num + similar_num):
            temp_term_1, temp_term_2 = self.suboperators['term_param_crossover'].apply(objective = (objective[0].structure[i], 
                                                                                                    objective[1].structure[i]),
                                                                                       arguments = subop_args['term_param_crossover']) 
            if (check_uniqueness(temp_term_1, objective[0].structure[:i] + objective[0].structure[i+1:]) and 
                check_uniqueness(temp_term_2, objective[1].structure[:i] + objective[1].structure[i+1:])):                     
                objective[0].structure[i] = temp_term_1; objective[1].structure[i] = temp_term_2

        for i in range(same_num + similar_num, len(objective[0].structure)):
            if check_uniqueness(objective[0].structure[i], objective[1].structure) and check_uniqueness(objective[1].structure[i], objective[0].structure):
                objective[0].structure[i], objective[1].structure[i] = self.suboperators['term_crossover'].apply(objective = (objective[0].structure[i], 
                                                                                                                              objective[1].structure[i]),
                                                                                                               arguments = subop_args['term_crossover'])
                
        return objective[0], objective[1]

    def use_default_tags(self):
        self._tags = {'crossover', 'gene level', 'contains suboperators', 'standard'}

class EquationExchangeCrossover(CompoundOperator):
    key = 'EquationExchangeCrossover'
    
    @HistoryExtender(f'\n -> performing equation exchange crossover', 'ba')
    def apply(self, objective : tuple, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        objective[0].structure, objective[1].structure = objective[1].structure, objective[0].structure
        return objective[0], objective[1]

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
    key = 'TermParamCrossover'
    
    def apply(self, objective : tuple, arguments : dict):
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
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        # offspring_1 = deepcopy(objective[0]); offspring_2 = deepcopy(objective[1])
        objective[0].reset_saved_state(); objective[1].reset_saved_state()
        
        if len(objective[0].structure) != len(objective[1].structure):
            print([(token.label, token.params) for token in objective[0].structure], [(token.label, token.params) for token in objective[1].structure])
            raise Exception('Wrong terms passed:')
        for term1_token_idx in np.arange(len(objective[0].structure)):
            term2_token_idx = [i for i in np.arange(len(objective[1].structure)) 
                               if objective[1].structure[i].label == objective[0].structure[term1_token_idx].label][0]
            for param_idx, param_descr in objective[0].structure[term1_token_idx].params_description.items():
                if param_descr['name'] == 'power': power_param_idx = param_idx
                if param_descr['name'] == 'dim': dim_param_idx = param_idx                

            for param_idx in np.arange(objective[0].structure[term1_token_idx].params.size):
                if param_idx != power_param_idx and param_idx != dim_param_idx:
                    try:
                        objective[0].structure[term1_token_idx].params[param_idx] = (objective[0].structure[term1_token_idx].params[param_idx] + 
                                                                                     self.params['term_param_proportion'] 
                                                                                     * (objective[1].structure[term2_token_idx].params[param_idx] 
                                                                                        - objective[0].structure[term1_token_idx].params[param_idx]))
                    except KeyError:
                        print([(token.label, token.params) for token in objective[0].structure], [(token.label, token.params) for token in objective[1].structure])
                        raise Exception('Wrong set of parameters:', objective[0].structure[term1_token_idx].params_description, objective[1].structure[term1_token_idx].params_description)
                    objective[1].structure[term2_token_idx].params[param_idx] = (objective[0].structure[term1_token_idx].params[param_idx] + 
                                                                                (1 - self.params['term_param_proportion']) 
                                                                                * (objective[1].structure[term2_token_idx].params[param_idx] 
                                                                                - objective[0].structure[term1_token_idx].params[param_idx]))
        objective[0].reset_occupied_tokens(); objective[1].reset_occupied_tokens()
        return objective[0], objective[1]

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
    key = 'TermCrossover'
    
    def apply(self, objective : tuple, arguments : dict):
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
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        if (np.random.uniform(0, 1) <= self.params['crossover_probability'] and 
            objective[1].descr_variable_marker == objective[0].descr_variable_marker):
                return objective[1], objective[0]
        else:
                return objective[0], objective[1]
        
    def use_default_tags(self):
        self._tags = {'crossover', 'term level', 'exploration', 'no suboperators', 'standard'}


def get_singleobjective_variation(variation_params : dict = {}):
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = variation_params)    

    term_param_crossover = TermParamCrossover(['term_param_proportion'])
    add_kwarg_to_operator(operator = term_param_crossover)
    term_crossover = TermCrossover(['crossover_probability'])
    add_kwarg_to_operator(operator = term_crossover)

    equation_crossover = EquationCrossover()

    chromosome_crossover = ChromosomeCrossover()

    pl_cross = PopulationLevelCrossover(['PBI_penalty'])
    add_kwarg_to_operator(operator = pl_cross)

    equation_crossover.set_suboperators(operators = {'term_param_crossover' : term_param_crossover, 
                                                     'term_crossover' : term_crossover})
    chromosome_crossover.set_suboperators(operators = {'equation_crossover' : equation_crossover},
                                          probas = {'equation_crossover' : [0.9, 0.1]})
    pl_cross.set_suboperators(operators = {'chromosome_crossover' : chromosome_crossover})
    return pl_cross


def get_multiobjective_variation(variation_params : dict = {}): # Rename function calls where necessary
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = variation_params)    

    term_param_crossover = TermParamCrossover(['term_param_proportion'])
    add_kwarg_to_operator(operator = term_param_crossover)
    term_crossover = TermCrossover(['crossover_probability'])
    add_kwarg_to_operator(operator = term_crossover)

    equation_crossover = EquationCrossover()
    metaparameter_crossover = MetaparamerCrossover(['metaparam_proportion'])
    add_kwarg_to_operator(operator = metaparameter_crossover)
    equation_exchange_crossover = EquationExchangeCrossover()

    chromosome_crossover = ChromosomeCrossover()

    pl_cross = PopulationLevelCrossover(['PBI_penalty'])
    add_kwarg_to_operator(operator = pl_cross)

    equation_crossover.set_suboperators(operators = {'term_param_crossover' : term_param_crossover, 
                                                     'term_crossover' : term_crossover})
    chromosome_crossover.set_suboperators(operators = {'equation_crossover' : [equation_crossover, equation_exchange_crossover],
                                                       'param_crossover' : metaparameter_crossover},
                                          probas = {'equation_crossover' : [0.9, 0.1]})
    pl_cross.set_suboperators(operators = {'chromosome_crossover' : chromosome_crossover})
    return pl_cross