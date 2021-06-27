#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:46:31 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy

from epde.src.structure import Term, Check_Unqueness
from epde.src.supplementary import *
from epde.src.supplementary import Filter_powers
from epde.src.operators.template import Compound_Operator, try_iterable

from epde.src.decorators import History_Extender, Reset_equation_status

class PopLevel_mutation(Compound_Operator):
    """
    The general operator of mutation, which applies all off the mutation suboperators, which are selected in its self.suboperators['Mutation'] 
    to the population    
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the Specific_Operator class. 
        Suboperators, performing tasks of calculation of weights for each terms & fitness function calculation. 
        Additionally, it contains suboperators of different mutation types. Dictionary: keys - strings from 'Mutation', 'Coeff_calc', 'Fitness_eval'.
        values - corresponding operators (objects of Specific_Operator class).

    params : dict
        Inhereted from the Specific_Operator class. 
        Parameters of the operator; main parameters: 
            
            elitism - number of inidividuals with highest fitness values, spared from the mutation to preserve their quality;
                
            indiv_mutation_prob - probability of an individual in a population to be affected by a mutation;
            
            r_mutation - probability of a term in an equation, selected for mutation, to be affected by any mutation operator;
            
            type_probabilities - propabilities for selecting each mutation suboperator to affect the equation (In this operator, set by euristic, to be updated).
            
    Methods:
    -----------
    apply(population)
        return the new population, created with the specified operators and containing mutated population.    
    
    """

#    def indiv_mut(self, population):
#        
    def apply(self, population):
        """
        Return the new population, created with the specified operators and containing mutated population.
        
        Parameters:
        -----------
        population : list of Equation objects
            The population, to which the mutation operators would be applied.
            
        Returns:
        ----------
        population : list of Equation objects
            The input population, altered by mutation operators.
            
        """
        
        population = Population_Sort(population)
        for indiv_idx in range(self.params['elitism'], len(population)):
            if np.random.uniform(0, 1) <= self.params['indiv_mutation_prob']:
                self.suboperators['Equatiion_mutation'].apply(population[indiv_idx])

#                self.suboperators['Coeff_calc'].apply(population[indiv_idx])
#                self.suboperators['Fitness_eval'].apply(population[indiv_idx])
        return population        


class Equation_mutation(Compound_Operator):
    @Reset_equation_status(reset_input = True)
    @History_Extender(f'\n -> mutating equation', 'ba')
    def apply(self, equation):
        for term_idx in range(equation.n_immutable, len(equation.structure)):
            if np.random.uniform(0, 1) <= self.params['r_mutation'] and term_idx != equation.target_idx:
                #if type(self.params['type_probabilities']) == type(None):
                self.params['type_probabilities'] = [1 - 1/pow(equation.structure[term_idx].total_params, 2), 1/pow(equation.structure[term_idx].total_params, 2)]
                if try_iterable(self.suboperators['Mutation']):
                    mut_operator = np.random.choice(self.suboperators['Mutation'], p=self.params['type_probabilities'])
                else:
                    mut_operator = self.suboperators['Mutation']
                if 'forbidden_tokens' in mut_operator.params.keys():
                    mut_operator.params['forbidden_tokens'] = [factor for factor in equation.structure[equation.target_idx].structure if factor.status['unique_for_right_part']]   # [factor.label for factor in equation.structure[].structure]
                equation.structure[term_idx] = mut_operator.apply(term_idx, equation)
                equation.check_split_correctness()
        
        

class Term_mutation(Compound_Operator): # Добавить "запрещённые" токены 
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
            The new, randomly created, term.
            
        """       
#        print('reference forbidden tokens:', self.params['forbidden_tokens'])
        new_term = Term(equation.pool, max_factors_in_term = equation.max_factors_in_term, forbidden_tokens = self.params['forbidden_tokens'])        #) #
        while not Check_Unqueness(new_term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
            new_term = Term(equation.pool, max_factors_in_term = equation.max_factors_in_term, forbidden_tokens = self.params['forbidden_tokens'])
#        print(equation.structure[term_idx].text_form, 'mutated into', new_term.text_form)
        new_term.use_cache()
        return new_term
        

#class Best_individual_mutation(Specific_Operator):
#    def apply(self, term_idx, equation):
#        new_term = Term(equation.tokens, max_factors_in_term = equation.max_factors_in_term, forbidden_tokens = self.params['forbidden_tokens'])        #) #
#        while not Check_Unqueness(new_term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
#            new_term = Term(equation.tokens, max_factors_in_term = equation.max_factors_in_term, forbidden_tokens = self.params['forbidden_tokens'])
##        print(equation.structure[term_idx].text_form, 'mutated into', new_term.text_form)
#        return new_term        


class Parameter_mutation_old(Compound_Operator):
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
        unmutable_params = {'dim'}
        while True:
            term = equation.structure[term_idx] 
            for factor in term.structure:   # Возможное место ошибок
                parameter_selection = deepcopy(factor.params)
                token_family = [token_family for token_family in term.tokens if factor.label in token_family.tokens][0]
                for param, interval in token_family.token_params.items():
                    if param == 'power':
                        continue
                    if np.random.random() < self.params['r_param_mutation'] and param not in unmutable_params:
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
                            parameter_selection[param] = np.min((np.max((parameter_selection[param] + shift, interval[0])), interval[1]))
                        else:
                            parameter_selection[param] = parameter_selection[param] + shift
                factor.Set_parameters(**parameter_selection)
            term.structure = Filter_powers(term.structure)        
            if Check_Unqueness(term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
                break
        term.reset_saved_state()
        return term


class Parameter_mutation(Compound_Operator):
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
            for factor in term.structure:   # Возможное место ошибок
                parameter_selection = deepcopy(factor.params)
                for param_idx, param_properties in factor.params_description.items():
#                    if param_properties['name'] == 'power':
#                        continue
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
            term.structure = Filter_powers(term.structure)        
            if Check_Unqueness(term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
                break
        term.reset_saved_state()
        return term