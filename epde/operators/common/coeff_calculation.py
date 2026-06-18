#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:58:18 2021

@author: mike_ubuntu
"""

import numpy as np
from sklearn.linear_model import LinearRegression

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.structure.main_structures import Equation


# Marker attribute set by ``LASSOSparsity.apply`` on the equation
# instance to indicate that the legacy LASSO post-processing refit
# (LinearRegression on the LASSO survivors, un-normalised features)
# should run on this equation. ``VWSRSparsity`` does NOT set this
# marker; its PhysicsInformedLasso output is already on the physical
# scale and would be corrupted by the refit. Gating on the equation
# (rather than on the operator) keeps the strategy wiring untouched.
LEGACY_REFIT_MARKER = '_legacy_refit_pending'


class LinRegBasedCoeffsEquation(CompoundOperator):
    '''
    Refit the LASSO survivors with ``LinearRegression`` on
    *un-normalised* features, replacing ``weights_final`` with
    physically-scaled coefficients.

    Restores the legacy two-step pipeline:
        1. LASSOSparsity fits Lasso on min-max-normalised features and
           identifies the surviving (non-zero) terms.
        2. This operator re-fits those survivors with ordinary least
           squares on the un-normalised features to recover physical
           coefficient magnitudes (LASSO coefficients are biased by
           both L1 shrinkage and the upstream normalisation).

    Gated by the per-equation marker ``LEGACY_REFIT_MARKER`` that
    ``LASSOSparsity`` sets and ``VWSRSparsity`` leaves unset, so this
    operator can be wired into both pipelines without a strategy flag.

    Output shape matches the upstream sparsity convention:
    ``np.append(coef_, intercept)`` -- one entry per surviving non-zero
    feature plus a trailing intercept slot. Downstream consumers
    (``L2Fitness.apply``, ``L2LRFitness.apply``) need no change.
    '''
    key = 'LinRegCoeffCalc'

    @staticmethod
    def _legacy_evaluate_nonzero(objective: Equation):
        """Build target + un-normalised feature matrix from the LASSO
        survivors, independent of ``Equation.evaluate``.

        Iterate the structure, skip the target, and emit columns only for
        terms whose ``weights_internal`` slot is non-zero. Returns
        ``(target, features)`` with ``features=None`` when every
        non-target slot was filtered to zero.
        """
        target = objective.structure[objective.target_idx].evaluate(False)
        feats = []
        for term_idx, term in enumerate(objective.structure):
            if term_idx == objective.target_idx:
                continue
            wi_pos = (term_idx if term_idx < objective.target_idx
                      else term_idx - 1)
            if objective.weights_internal[wi_pos] != 0:
                feats.append(term.evaluate(False))
        if not feats:
            return target, None
        features = np.vstack(feats)
        if features.ndim == 1:
            features = np.expand_dims(features, 1).T
        features = np.transpose(features)
        return target, features

    def apply(self, objective : Equation, arguments : dict = None):
        """Refit LASSO survivors with un-normalised LinearRegression.

        Skipped unless ``LASSOSparsity`` set ``LEGACY_REFIT_MARKER`` on
        the equation. The marker is cleared after the refit so a stale
        marker from a previous run doesn't double-trigger work.
        """
        assert objective.weights_internal_evald, (
            'Trying to calculate final weights before evaluating '
            'intermediate ones (no sparsity).'
        )
        if not getattr(objective, LEGACY_REFIT_MARKER, False):
            return

        target, features = self._legacy_evaluate_nonzero(objective)
        if features is None:
            # No non-zero terms to refit -- leave ``weights_final`` as
            # set by the upstream sparsity step (just the intercept).
            objective.weights_final_evald = True
            setattr(objective, LEGACY_REFIT_MARKER, False)
            return

        self.g_fun_vals = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask]
        estimator = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1)
        estimator.fit(features, target, sample_weight=self.g_fun_vals)
        objective.weights_final = np.append(estimator.coef_, estimator.intercept_)
        objective.weights_final_evald = True
        setattr(objective, LEGACY_REFIT_MARKER, False)

    def use_default_tags(self):
        self._tags = {'coefficient calculation', 'gene level', 'no suboperators', 'inplace'}
