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
from epde.interface.search_config import active_config


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

    Both live entirely in ``weights_internal`` under the unified layout
    (``Equation._validate_weight_layout``): the term coefficients occupy
    ``[:-1]`` and the intercept the trailing slot. The VWSR/LASSO branch this
    replaces existed because the two sparsity operators used to emit different
    lengths -- VWSR a bare ``estimator.coef_``, LASSO the same plus a
    separately-tracked ``weights_final`` intercept -- so the count had to sniff
    ``len(wi) == m + 1`` and reach across to ``weights_final`` in one arm.

    The target term is excluded as ubiquitous. Requires ``weights_internal``
    (raises pre-fit, exactly like the ``'factors'`` core).
    """
    wi = equation.weights_internal
    return int(np.count_nonzero(wi[:-1])) + (1 if wi[-1] != 0 else 0)


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

    Reads ``weights_internal``, the vector that DECIDES support, via
    ``Equation.weight_index``. (Both vectors are structure-aligned now, so
    either would index correctly; ``weights_final`` merely carries the refit
    magnitudes at the same positions.)
    """
    tgt = equation.target_idx
    eq_compl = 0
    for idx, term in enumerate(equation.structure):
        if idx == tgt:
            eq_compl += complexity_deriv(term.structure)
        elif equation.weights_internal[equation.weight_index(idx, tgt)] != 0:
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
    on ``active_config().objectives.complexity_metric`` -- translated truth
    systems and the offline tools build SoEqs with no fitness host, and
    complexity is deterministic from structure + weights, so the fallback is
    exact (this is deliberately laxer than the stability reader's assert).
    Staleness is not a risk: the same reset path that clears
    ``fitness_calculated`` on mutation clears ``complexity_calculated``.
    """
    def _one(equation):
        if getattr(equation, 'complexity_calculated', False):
            return equation.complexity_value
        if active_config().objectives.complexity_metric == 'terms':
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


# ---------------------------------------------------------------------------
# Ideal values, mirrored from the objective fillers
# ---------------------------------------------------------------------------
# These readers are what ``SoEq.set_objective_functions`` receives, so they
# are the SoEq-side half of the ideal-point lockstep. The numbers are read off
# the filler classes rather than restated, so there is exactly one place where
# an objective's ideal is declared (``EquationObjective.ideal_value``).
def _mirror_ideal_values():
    from epde.operators.common.objectives import Complexity, Discrepancy, Instability

    equation_fitness.ideal_value = Discrepancy.ideal_value
    equation_terms_stability.ideal_value = Instability.ideal_value
    for reader in (equation_complexity, equation_complexity_by_factors,
                   equation_complexity_by_terms):
        reader.ideal_value = Complexity.ideal_value


_mirror_ideal_values()


def objective_ideal_values(obj_funs):
    """Ideal values for a system's registered objective readers, or ``None``.

    Returns ``None`` when any reader carries no ``ideal_value`` -- notably the
    ``functools.partial`` wrappers built by ``generate_partial`` for the
    per-variable single-objective and legacy PIC paths, which do not forward
    attribute access. Callers use this as a cross-check on an ideal point
    derived elsewhere, so "cannot tell" must stay distinguishable from a
    genuine mismatch.
    """
    values = []
    for reader in obj_funs:
        ideal = getattr(reader, 'ideal_value', None)
        if ideal is None:
            return None
        values.append(float(ideal))
    return values
