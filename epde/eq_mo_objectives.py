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


def equation_terms_stability(system, equation_key):
    # g_fun_vals = global_var.grid_cache.g_func
    # eq = system.vals[equation_key]
    # # Calculate r-loss
    # target = eq.structure[eq.target_idx]
    # target_vals = target.evaluate(False)
    # features_vals = []
    # nonzero_features_indexes = []
    #
    # for i in range(len(eq.structure)):
    #     if i == eq.target_idx:
    #         continue
    #     idx = i if i < eq.target_idx else i - 1
    #     if eq.weights_internal[idx] != 0:
    #         features_vals.append(eq.structure[i].evaluate(False))
    #         nonzero_features_indexes.append(idx)
    #
    # if len(features_vals) == 0:
    #     eq.weights_final = np.zeros(len(eq.structure))
    #     lr = 0
    # else:
    #     features = features_vals[0]
    #     if len(features_vals) > 1:
    #         for i in range(1, len(features_vals)):
    #             features = np.vstack([features, features_vals[i]])
    #     features = np.vstack([features, np.ones(features_vals[0].shape)])  # Add constant feature
    #     features = np.transpose(features)
    #
    #     window_size = len(target_vals) // 2
    #     num_horizons = len(target_vals) - window_size + 1
    #     eq_window_weights = []
    #
    #     # Compute coefficients and collect statistics over horizons
    #     for start_idx in range(num_horizons):
    #         end_idx = start_idx + window_size
    #
    #         target_window = target_vals[start_idx:end_idx]
    #         feature_window = features[start_idx:end_idx, :]
    #
    #         estimator = LinearRegression(fit_intercept=False)
    #         if feature_window.ndim == 1:
    #             feature_window = feature_window.reshape(-1, 1)
    #         try:
    #             g_fun_vals_window = g_fun_vals.reshape(-1)[start_idx:end_idx]
    #         except AttributeError:
    #             g_fun_vals_window = None
    #         estimator.fit(feature_window, target_window, sample_weight=g_fun_vals_window)
    #         valuable_weights = estimator.coef_
    #
    #         window_weights = np.zeros(len(eq.structure))
    #         for weight_idx in range(len(window_weights)):
    #             if weight_idx in nonzero_features_indexes:
    #                 window_weights[weight_idx] = valuable_weights[nonzero_features_indexes.index(weight_idx)]
    #         window_weights[-1] = valuable_weights[-1]
    #         eq_window_weights.append(window_weights)
    #
    #     eq_cv = np.array([np.abs(np.std(_) / (np.mean(_))) for _ in zip(*eq_window_weights)]) # As in paper's repo
    #     # eq_cv_valuable = [x for x in eq_cv if not np.isnan(x)]
    #     # lr = np.mean(eq_cv_valuable)
    #     lr = eq_cv.mean()
    # return lr
    assert system.vals[equation_key].stability_calculated
    res = system.vals[equation_key].coefficients_stability
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