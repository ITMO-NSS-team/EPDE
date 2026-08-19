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


def _terms_of_equation(equation):
    """Per-equation ``'terms'`` complexity: the number of active non-target
    terms plus 1 when the fitted intercept is non-zero.

    UNIFORM across the sparsity pairings (the whole point -- the raw
    ``nnz(weights_internal)`` this replaces counted the intercept slot under
    VWSR but could not see the intercept under LASSO):

    * VWSR: ``weights_internal`` is length ``m + 1`` (``m`` non-target
      terms + a trailing intercept slot) -- the intercept state is
      ``weights_internal[-1] != 0``.
    * LASSO: ``weights_internal`` is length ``m`` (no intercept slot); the
      intercept always trails ``weights_final``. On a degenerate-stamped
      equation (``weights_final_evald`` False) the intercept contributes 0
      instead of raising -- the discrepancy axis is already LOSS_NAN_VAL
      there, so the value cannot influence selection.

    The target term is excluded as ubiquitous. Requires ``weights_internal``
    (raises pre-fit, exactly like the ``'factors'`` core).
    """
    wi = equation.weights_internal
    m = len(equation.structure) - 1
    count = int(np.count_nonzero(wi[:m]))
    if len(wi) == m + 1:                       # VWSR: trailing intercept slot
        intercept_nonzero = wi[-1] != 0
    else:                                      # LASSO: intercept in weights_final
        intercept_nonzero = (bool(getattr(equation, 'weights_final_evald', False))
                             and equation.weights_final[-1] != 0)
    return count + (1 if intercept_nonzero else 0)


def equation_complexity_by_terms(system, equation_key=None):
    '''
    Evaluate the complexity of the system of PDEs as the number of ACTIVE
    terms per equation: non-zero-weight non-target terms, plus 1 when the
    fitted intercept is non-zero (see ``_terms_of_equation`` for the
    LASSO/VWSR uniformity contract). The target term is excluded due to its
    ubiquity. When ``equation_key`` is None, returns a per-equation tuple
    matching the ``system.vars_to_describe`` order; otherwise the scalar
    count for the named equation.

    NOTE: the pre-family-refactor body (raw
    ``np.count_nonzero(weights_internal)``, required positional key) was
    dead code -- zero call sites -- so this semantics change is safe.
    '''
    if equation_key is None:
        return tuple(_terms_of_equation(system.vals[k]) for k in system.vars_to_describe)
    return _terms_of_equation(system.vals[equation_key])


def _complexity_of_equation(equation):
    """Per-equation ``'factors'`` complexity core (the legacy semantics).

    Index by ``weights_internal`` (always length ``len(structure)-1``,
    one entry per non-target term in structure order) rather than
    ``weights_final`` (zero-filtered to ``nnz+1`` by ``LASSOSparsity``
    and ``VWSRSparsity``): structure-position indexing breaks against
    ``weights_final`` whenever the sparsity step zeros more than one
    weight.
    """
    tgt = equation.target_idx
    eq_compl = 0
    for idx, term in enumerate(equation.structure):
        if idx < tgt:
            if not equation.weights_internal[idx] == 0:
                eq_compl += complexity_deriv(term.structure)
        elif idx > tgt:
            if not equation.weights_internal[idx-1] == 0:
                eq_compl += complexity_deriv(term.structure)
        else:
            eq_compl += complexity_deriv(term.structure)
    return eq_compl


def _complexity_single_eq(system, equation_key):
    # Thin system-level wrapper kept for the existing callers/tests; the
    # per-equation core moved to ``_complexity_of_equation`` so the
    # ``Complexity`` filler (which only sees an Equation) can share it.
    return _complexity_of_equation(system.vals[equation_key])


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


def equation_complexity(system, equation_key=None):
    """The COMPLEXITY-family Pareto reader (the second-axis counterpart of
    ``equation_terms_stability``).

    Reads the ``complexity_value`` attribute the ``Complexity`` filler wrote
    when ``complexity_calculated`` is set (the live-search path). Otherwise
    FALLS BACK to lazy computation via the same per-equation cores, routed
    on ``epde.globals.resolve_complexity_metric()`` -- translated truth
    systems and the offline tools build SoEqs with no fitness host, and
    complexity is deterministic from structure + weights, so the fallback is
    exact (this is deliberately laxer than the stability reader's assert).
    Staleness is not a risk: the same reset path that clears
    ``fitness_calculated`` on mutation clears ``complexity_calculated``.
    """
    def _one(equation):
        if getattr(equation, 'complexity_calculated', False):
            return equation.complexity_value
        if global_var.resolve_complexity_metric() == 'terms':
            return _terms_of_equation(equation)
        return _complexity_of_equation(equation)
    if equation_key is None:
        return tuple(_one(system.vals[k]) for k in system.vars_to_describe)
    return _one(system.vals[equation_key])


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
    # KNOWN QUIRK, kept deliberately: ``factor`` leaks from the loop, so the
    # whole term's summed total is multiplied by the LAST factor's power only
    # (not per-factor). Every legacy-pipeline artifact was scored with this
    # formula -- fixing it silently would break bit-compatibility of the
    # 'factors' complexity axis. Pinned by a unit test; any fix must be an
    # explicit, separately-gated change.
    return total*factor.param('power')
