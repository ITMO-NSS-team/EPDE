#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 18:48:23 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import partial

def generate_partial(obj_function, equation_idx):
    return partial(obj_function, equation_idx = equation_idx)

def equation_discrepancy(system, equation_idx):
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
    res = np.sum(np.abs(system.structure[equation_idx].evaluate(normalize = False, return_val = True)[0]))
    return res

def equation_complexity_by_terms(system, equation_idx):
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
    return np.count_nonzero(system.structure[equation_idx].weights_internal)

def equation_complexity_by_factors(system, equation_idx):
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
    eq_compl = 0
    # print(equation)
    # print(equation.text_form)
    
    for idx, term in enumerate(system.structure[equation_idx].structure):
        if idx < system.structure[equation_idx].target_idx:
            if not system.structure[equation_idx].weights_final[idx] == 0: 
                eq_compl += len(term.structure)
        elif idx > system.structure[equation_idx].target_idx:
            if not system.structure[equation_idx].weights_final[idx-1] == 0: eq_compl += len(term.structure)
        else:
            eq_compl += len(term.structure)
    return eq_compl 
