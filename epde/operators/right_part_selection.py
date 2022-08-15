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
from epde.operators.template import CompoundOperator
from epde.decorators import History_Extender

class PoplevelRightPartSelector(CompoundOperator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. This operator is executed on the population level, taking a population (in form 
    of python ``list``) and in-place modifying it.
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the CompoundOperator class
        key - str, value - instance of a class, inhereted from the CompoundOperator. 
        Suboperators, performing tasks of population/equation processing. In this case, only one suboperator is present: 
        fitness_calculation, dedicated to calculation of fitness function value.

    Methods:
    -----------
    apply(population)
        return None
        Inplace detection of index of the best separation into right part, saved into ``equation.target_idx`` for 
        every equation in population, if the right part for it has not already been calculated. 

    
    ''' 
    def apply(self, population, separate_vars = []):
        '''
        
        Select right part for the equation, selecting its terms one by one to be the right part of the equation, 
        approximating it with the other terms and calculating the fitness function for such structure. 
        
        Parameters:
        -------------
        
        population : list of Equation objects;
            The population, which contains the equations, that would be processed by the operator. 
        
        separate_vars : set of frozen sets with str elements
            The elements of the set - the frozen sets contain the variables, described by the other equation for other 
            equations of the system. Has meaning only if we discover the system of equations. 
            
        Returns:
        -------------
            
        None
        
        '''
        # to_rps = 0
        for equation in population:
            if not equation.right_part_selected:
                self.suboperators['eq_level_rps'].apply(equation, separate_vars)
        return population

    def use_default_tags(self):
        self._tags = {'equation right part selection', 'population level', 'contains suboperators'}
    
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
    @History_Extender('\n -> The equation structure was detected: ', 'a')        
    def apply(self, equation, separate_vars):
        max_fitness = 0
        max_idx = 0
        if not equation.contains_deriv:
            equation.reconstruct_to_contain_deriv()
        for target_idx, _ in enumerate(equation.structure): # target_term
            if not equation.structure[target_idx].contains_deriv:
                continue
            equation.target_idx = target_idx
            self.suboperators['fitness_calculation'].apply(equation)
            if equation.fitness_value > max_fitness:
                max_fitness = equation.fitness_value
                max_idx = target_idx                 
            else:
                pass

        equation.target_idx = max_idx
        self.suboperators['fitness_calculation'].apply(equation)
        if not np.isclose(equation.fitness_value, max_fitness) and global_var.verbose.show_warnings:
            warnings.warn('Reevaluation of fitness function for equation has obtained different result. Not an error, if ANN DE solver is used.')
        equation.right_part_selected = True    

    def use_default_tags(self):
        self._tags = {'equation right part selection', 'equation level', 'contains suboperators'}