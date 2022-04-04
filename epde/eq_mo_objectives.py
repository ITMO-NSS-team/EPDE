#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 18:48:23 2021

@author: mike_ubuntu
"""

import numpy as np

def system_discrepancy(system):
    '''
    Evaluate the discrepancy of the system of PDEs, using sum of the L2 norm of the discrepancy 
    of each equation in the system from zero.
    
    Parameters:
    -----------
        system - ``epde.structure.SoEq`` object
        The system, that is to be evaluated.
        
    Returns:
    ----------
        discrepancy : float.
        The value of the error metric.
    '''
    res = system.evaluate(normalize = False)
    # print(f'achieved system discrepancy is {res}')
    return res

def system_complexity_by_terms(system):
    '''
    Evaluate the complexity of the system of PDEs, evaluating a number of terms for each equation. 
    In the evaluation, we consider only terms with non-zero weights, and the target term with the free 
    coefficient are not included in the final metric due to their ubiquty in the equations.
    
    Parameters:
    -----------
        system - ``epde.structure.SoEq`` object
        The system, that is to be evaluated.
        
    Returns:
    ----------
        discrepancy : list of integers.
        The values of the error metric: list entry for each of the equations.
    '''    
    return [np.count_nonzero(eq.weights_internal) for eq in system.structure] 

def system_complexity_by_factors(system):
    '''
    Evaluate the complexity of the system of PDEs, evaluating a number of factors in terms for each 
    equation. In the evaluation, we consider only terms with non-zero weights and target, while
    the free coefficient is not included in the final metric. Also, the real-valued factors are
    not considered in the result.
    
    Parameters:
    -----------
        system - ``epde.structure.SoEq`` object
        The system, that is to be evaluated.
        
    Returns:
    ----------
        discrepancy : list of integers.
        The values of the error metric: list entry for each of the equations.
    '''    
    complexities = []
    for equation in system.structure:
        eq_compl = 0
        # print(equation)
        # print(equation.text_form)
        
        for idx, term in enumerate(equation.structure):
            if idx < equation.target_idx:
                if not equation.weights_final[idx] == 0: eq_compl += len(term.structure)
            elif idx > equation.target_idx:
                if not equation.weights_final[idx-1] == 0: eq_compl += len(term.structure)
            else:
                eq_compl += len(term.structure)
        complexities.append(eq_compl)
    return complexities 