#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 18:48:23 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import partial


def generate_partial(obj_function, equation_key):
    return partial(obj_function, equation_key=equation_key)


def equation_fitness(system, equation_key):
    '''
    Evaluate the quality of the system of PDEs, using the individual values of fitness function for equations.

    Parameters:
    -----------
        system - ``epde.structure.main_structures.SoEq`` object
        The system, that is to be evaluated.

    Returns:
    ----------
        error : float.
        The value of the error metric.
    '''
    # print(f'System, for which we evaluate fitness: {system.text_form}')
    # print(f'For equation key {equation_key}, {system.vals[equation_key].fitness_calculated}')
    assert system.vals[equation_key].fitness_calculated, 'Trying to call fitness before its evaluation.'
    res = system.vals[equation_key].fitness_value
    return res


def equation_complexity_by_terms(system, equation_key):
    '''
    Evaluate the complexity of the system of PDEs, evaluating a number of terms for each equation.
    In the evaluation, we consider only terms with non-zero weights, and the target term with the free
    coefficient are not included in the final metric due to their ubiquty in the equations.

    Parameters:
    -----------
        system - ``epde.structure.main_structures.SoEq`` object
        The system, that is to be evaluated.

    Returns:
    ----------
        discrepancy : list of integers.
        The values of the error metric: list entry for each of the equations.
    '''
    return np.count_nonzero(system.vals[equation_key].weights_internal)


def equation_complexity_by_factors(system, equation_key):
    '''
    Evaluate the complexity of the system of PDEs, evaluating a number of factors in terms for each
    equation. In the evaluation, we consider only terms with non-zero weights and target, while
    the free coefficient is not included in the final metric. Also, the real-valued factors are
    not considered in the result.

    Parameters:
    -----------
        system - ``epde.structure.main_structures.SoEq`` object
        The system, that is to be evaluated.

    Returns:
    ----------
        discrepancy : list of integers.
        The values of the error metric: list entry for each of the equations.
    '''
    # eq_compl = 0

    # for idx, term in enumerate(system.vals[equation_key].structure):
    #     if idx < system.vals[equation_key].target_idx:
    #         if not system.vals[equation_key].weights_final[idx] == 0:
    #             eq_compl += len(term.structure)
    #     elif idx > system.vals[equation_key].target_idx:
    #         if not system.vals[equation_key].weights_final[idx-1] == 0:
    #             eq_compl += len(term.structure)
    #     else:
    #         eq_compl += len(term.structure)
    # return eq_compl
    eq_compl = 0

    for idx, term in enumerate(system.vals[equation_key].structure):
        if idx < system.vals[equation_key].target_idx:
            if not system.vals[equation_key].weights_final[idx] == 0:
                eq_compl += complexity_deriv(term.structure)
        elif idx > system.vals[equation_key].target_idx:
            if not system.vals[equation_key].weights_final[idx-1] == 0:
                eq_compl += complexity_deriv(term.structure)
        else:
            eq_compl += complexity_deriv(term.structure)
    return eq_compl

def complexity_deriv(term_list: list):
    total = 0
    for factor in term_list:
        if factor.deriv_code == [None]:
            total += 0.5
        elif factor.deriv_code is None:
            total += 0.5
        else:
            total += len(factor.deriv_code)
    return total*factor.param('power')