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
from epde.operators.template import Compound_Operator
from epde.decorators import History_Extender

class Poplevel_Right_Part_Selector(Compound_Operator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. This operator is executed on the population level, taking a population (in form 
    of python ``list``) and in-place modifying it.
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the Compound_Operator class
        key - str, value - instance of a class, inhereted from the Compound_Operator. 
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
                # to_rps += 1
        # print(f'Running right part selection. Equation w/o rp: {to_rps}')                
        return population

    @property
    def operator_tags(self):
        return {'equation right part selection', 'population level', 'contains suboperators'}
    
class Eq_Right_Part_Selector(Compound_Operator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the Compound_Operator class
        key - str, value - instance of a class, inhereted from the Compound_Operator. 
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
            # print(equation.target_idx, '-th term ')
            self.suboperators['fitness_calculation'].apply(equation)
            if equation.fitness_value > max_fitness: 
                max_fitness = equation.fitness_value
                max_idx = target_idx                 
            else:
                pass

        equation.target_idx = max_idx
        self.suboperators['fitness_calculation'].apply(equation)
        if not np.isclose(equation.fitness_value, max_fitness) and global_var.verbose.show_warnings:
            # print(equation.fitness_value, max_fitness)
            # print(equation.text_form)
            warnings.warn('Reevaluation of fitness function for equation has obtained different result. Not an error, if ANN DE solver is used.')
        equation.right_part_selected = True    

    @property
    def operator_tags(self):
        return {'equation right part selection', 'equation level', 'contains suboperators'}    

class Status_respecting_ERPS(Compound_Operator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the Compound_Operator class
        key - str, value - instance of a class, inhereted from the Compound_Operator. 
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
        candidate_equation = None
        for target_idx, _ in enumerate(equation.structure):
            if not equation.structure[target_idx].contains_deriv:
                continue
            reconstructed_equation = equation.reconstruct_by_right_part(target_idx)
            reconstructed_equation.target_idx = target_idx
            self.suboperators['fitness_calculation'].apply(reconstructed_equation)
            if reconstructed_equation.fitness_value > max_fitness:
                max_fitness = reconstructed_equation.fitness_value
                max_idx = target_idx
                candidate_equation = deepcopy(reconstructed_equation)
                reconstructed_equation.copy_properties_to(candidate_equation) 
        
        if candidate_equation is None:
            print(equation.text_form, max_fitness, max_idx)
            print([term.contains_deriv for term in equation.structure])
        equation.structure = deepcopy(candidate_equation.structure)

        try:    
            candidate_equation.copy_properties_to(equation)
        except AttributeError:
            print('Highest achieved fitness:', max_fitness)
            print([term.name for term in equation.structure])
            raise TypeError('Candidate equation right part term was not selected during search. Bug!')
            
        equation.target_idx = max_idx
        
        self.suboperators['fitness_calculation'].apply(equation)
        if not np.isclose(equation.fitness_value, max_fitness) and global_var.verbose.show_warnings:
            warnings.warn('Reevaluation of fitness function for equation has obtained different result. Not an error, if ANN DE solver is used.')
        equation.right_part_selected = True    
          

    @property
    def operator_tags(self):
        return {'equation right part selection', 'equation level', 'contains suboperators'}            