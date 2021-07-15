#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:20:59 2021

@author: mike_ubuntu
"""

import numpy as np

from epde.operators.template import Compound_Operator

#class PopLevel_fitness(Compound_Operator):
#    def apply(self, population):
#        for equation in population:
#            if not equation.fitness_calculated:
#                self.suboperators['Equation_fitness'].apply(equation)

class L2_fitness(Compound_Operator):
    """
    The operator, which calculates fitness function to the individual (equation) as the L2 norm 
    of the vector of disrepancy between left part of the equation and the right part, evaluated
    on the grid nodes.
    
    Notable attributes:
    -------------------
        
    params : dict
        Inhereted from the ``Compound_Operator`` class. 
        Parameters of the operator; main parameters: 
            
            penalty_coeff - penalty coefficient, to that the fitness function value of equation with no non-zero coefficients, is multiplied;
            
    suboperators : dict
        
        
    Methods:
    -----------
    apply(equation)
        calculate the fitness function of the equation, that will be stored in the equation.fitness_value.    
        
    """
    def apply(self, equation):
        """
        Calculate the fitness function values. The result is not returned, but stored in the equation.fitness_value attribute.
        
        Parameters:
        ------------
        equation : Equation object
            the equation object, to that the fitness function is obtained.
            
        Returns:
        ------------
        
        None
        """        

        self.suboperators['sparsity'].apply(equation)
        self.suboperators['coeff_calc'].apply(equation)
        
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        try:
            rl_error = np.linalg.norm(np.dot(features, equation.weights_final[:-1]) + 
                                  np.full(target.shape, equation.weights_final[-1]) - target, ord = 2)

        except ValueError:
            raise ValueError('An error in getting weights ')
        if rl_error == 0:
            fitness_value = np.inf
            print('infinite fitness!', equation.text_form)
        else:
            fitness_value = 1 / (rl_error)
        if np.sum(equation.weights_final) == 0:
            fitness_value = fitness_value * self.params['penalty_coeff']

        equation.fitness_calculated = True
        equation.fitness_value = fitness_value

    @property
    def operator_tags(self):
        return {'fitness evaluation', 'equation level', 'contains suboperators'}        