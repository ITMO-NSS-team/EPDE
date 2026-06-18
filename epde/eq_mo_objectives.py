#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 18:48:23 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import partial
from sklearn.linear_model import LinearRegression
import epde.globals as global_var


def generate_partial(obj_function, equation_key):
    return partial(obj_function, equation_key=equation_key)


def equation_fitness(system, equation_key = None):
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
    if equation_key:
        assert all(equation.fitness_calculated for equation in system.vals), 'Trying to call fitness before its evaluation.'
        res = system.vals[equation_key].fitness_value
    else:
        for equation in system.vals:
            assert equation.fitness_calculated
        # res = np.sum([equation.fitness_value for equation in system.vals])
        res = tuple([equation.fitness_value for equation in system.vals])
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


def _complexity_single_eq(system, equation_key):
    # Index by ``weights_internal`` (always length ``len(structure)-1``,
    # one entry per non-target term in structure order) rather than
    # ``weights_final`` (zero-filtered to ``nnz+1`` by ``LASSOSparsity``
    # and ``VWSRSparsity``): structure-position indexing breaks against
    # ``weights_final`` whenever the sparsity step zeros more than one
    # weight.
    equation = system.vals[equation_key]
    eq_compl = 0
    for idx, term in enumerate(equation.structure):
        if idx < equation.target_idx:
            if not equation.weights_internal[idx] == 0:
                eq_compl += complexity_deriv(term.structure)
        elif idx > equation.target_idx:
            if not equation.weights_internal[idx-1] == 0:
                eq_compl += complexity_deriv(term.structure)
        else:
            eq_compl += complexity_deriv(term.structure)
    return eq_compl


def equation_complexity_by_factors(system, equation_key=None):
    '''
    Evaluate the complexity of the system of PDEs as a number of factors in
    non-zero terms for each equation, excluding the free coefficient and
    real-valued factors. When ``equation_key`` is None, returns a per-equation
    tuple matching the ``system.vars_to_describe`` order; otherwise the scalar
    complexity for the named equation.
    '''
    if equation_key is None:
        return tuple(_complexity_single_eq(system, k) for k in system.vars_to_describe)
    return _complexity_single_eq(system, equation_key)


def equation_terms_stability(system, equation_key = None):
    if equation_key:
        assert system.vals[equation_key].stability_calculated
        res = system.vals[equation_key].coefficients_stability
    else:
        for equation in system.vals:
            assert equation.stability_calculated
        # res = np.sum([equation.coefficients_stability for equation in system.vals])
        res = tuple([equation.coefficients_stability for equation in system.vals])
    return res

def equation_aic(system, equation_key):
    assert system.vals[equation_key].aic_calculated
    res = system.vals[equation_key].aic
    return res

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
