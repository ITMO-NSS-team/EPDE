#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:58:18 2021

@author: mike_ubuntu
"""
from typing import Dict, Union

import numpy as np
from sklearn.linear_model import LinearRegression

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.structure.main_structures import Equation


class LinRegBasedCoeffsEquation(CompoundOperator):
    '''
    THE SOLE AUTHOR of ``weights_final`` and ``weights_final_evald``.

    The sparsity step decides the SUPPORT and writes ``weights_internal``;
    this operator turns that support into physical magnitudes. Which of its
    two branches runs is the sparsity operator's own declaration, read from
    ``fits_physical_scale`` (see ``LASSOSparsity`` / ``VWSRSparsity`` /
    ``KneeSparsity``) and pushed onto this instance by
    ``EqRightPartSelector.set_suboperators``:

        * ``False`` (legacy LASSO) -- the sparsity fit ran on min-max-rescaled
          features and target, so its coefficients are a support verdict, not
          magnitudes. Refit the survivors with un-normalised
          ``LinearRegression``.
        * ``True`` (VWSR, Knee) -- the sparsity fit was already on the physical
          scale, so the final magnitudes ARE the internal ones. Promote them.

    Every sparsity operator used to raise ``weights_final_evald`` itself, which
    meant four writers and, on the LASSO path, a window where a rescaled
    selection vector was flagged as the fitted answer. Now the flag goes up
    here, once, after the magnitudes exist.

    Output shape is the unified layout (``Equation._validate_weight_layout``):
    one slot per non-target term in structure order, zeros at the terms the
    sparsity step killed, then the intercept. The intercept follows THE
    INTERCEPT RULE -- it is refitted only when the sparsity step kept it.
    '''
    key = 'LinRegCoeffCalc'

    #: Set per run by ``EqRightPartSelector.set_suboperators`` from the wired
    #: sparsity operator. Defaults to the legacy refit so a directly
    #: instantiated, unwired operator behaves as it always did.
    sparsity_fits_physical = False

    @staticmethod
    def _sample_keys():
        """Trajectory IDs in ONE fixed order, shared by every stacking site
        below so the row order of the features, the target and the sample
        weights cannot drift apart."""
        return list(global_var.samples_manager.inner_shapes.keys())

    @classmethod
    def _stacked_g_func(cls):
        """Boundary-masked ``g_func`` weights of every trajectory, concatenated
        in ``_sample_keys`` order to match the stacked design matrix.

        The legacy path used ``global_var.grid_cache.g_func[g_func_mask]``,
        which is single-sample and is ``None`` outright when the caches were
        built with ``set_grids=False``.
        """
        gfuncs = global_var.samples_manager.gFunc('dm')
        return np.concat([np.asarray(gfuncs[key]).reshape(-1)
                          for key in cls._sample_keys()], axis=0)

    @staticmethod
    def _legacy_evaluate_nonzero(objective: Equation):
        """Build target + un-normalised feature matrix from the LASSO
        survivors, independent of ``Equation.evaluate``.

        Iterate the structure, skip the target, and emit columns only for
        terms whose ``weights_internal`` slot is non-zero. Returns
        ``(target, features)`` with ``features=None`` when every
        non-target slot was filtered to zero.
        """
        tgt = objective.target_idx
        keys = LinRegBasedCoeffsEquation._sample_keys()
        target: Dict[int, np.ndarray] = objective.target.evaluate()
        target = np.concat([target[key] for key in keys], axis = 0)
        feats = []
        for term_idx, term in enumerate(objective.structure):
            if term_idx == tgt:
                continue
            if objective.weights_internal[objective.weight_index(term_idx, tgt)] != 0:
                fdict = term.evaluate()
                feats.append(np.concat([fdict[key] for key in keys], axis = 0))
        if not feats:
            return target, None
        features = np.vstack(feats)
        if features.ndim == 1:
            features = np.expand_dims(features, 1).T
        features = np.transpose(features)
        return target, features

    def apply(self, objective : Equation, arguments : dict = None):
        """Put physical magnitudes on the support the sparsity step chose.

        Refits un-normalised when that support came from a rescaled fit;
        promotes ``weights_internal`` when the sparsity operator already fitted
        on the physical scale. Either way this is where ``weights_final_evald``
        goes up.
        """
        assert objective.weights_internal_evald, (
            'Trying to calculate final weights before evaluating '
            'intermediate ones (no sparsity).'
        )
        if getattr(self, 'sparsity_fits_physical', False):
            objective.weights_final = np.asarray(objective.weights_internal,
                                                 dtype=float).copy()
            objective.weights_final_evald = True
            return

        target, features = self._legacy_evaluate_nonzero(objective)
        self.g_fun_vals = self._stacked_g_func()

        # THE INTERCEPT RULE. ``weights_internal[-1]`` is the sparsity step's
        # SUPPORT decision for the free coefficient; when LASSO regularized it
        # away it is not a column of any later model, so the physical refit runs
        # through the origin and the verdict survives into ``weights_final``.
        # Either way the intercept is never appended to ``features`` -- sklearn
        # estimates it internally when the flag is set.
        fit_intercept = bool(objective.weights_internal[-1] != 0)

        # Full-structure layout: one slot per non-target term, then the
        # intercept, zeros retained (Equation._validate_weight_layout). The
        # former ``np.append(estimator.coef_, estimator.intercept_)`` produced a
        # COMPACT nnz+1 vector that only lined up with the structure because
        # ``remove_zero_terms`` happened to run first.
        weights_final = np.zeros(len(objective.structure))
        if features is None:
            # Every non-target term was zeroed. What the sparsity step left in
            # ``weights_final[-1]`` is a MIN-MAX-RESCALED intercept (see
            # LASSOSparsity), not a physical free coefficient -- re-estimate it
            # here as the g_func-weighted target mean, or leave it at 0.0 when
            # the penalty killed it.
            if fit_intercept:
                w_sum = float(np.sum(self.g_fun_vals))
                weights_final[-1] = (float(np.average(target, weights=self.g_fun_vals))
                                     if w_sum > 0 else float(np.mean(target)))
        else:
            estimator = LinearRegression(copy_X=True, fit_intercept=fit_intercept, n_jobs=-1)
            estimator.fit(features, target, sample_weight=self.g_fun_vals)
            # ``features`` holds only the LASSO survivors, so ``coef_`` is
            # nnz-long -- scatter it back onto the surviving full-structure
            # positions and leave zeros at the killed ones.
            weights_final[np.flatnonzero(objective.active_mask)] = estimator.coef_
            # Exactly 0.0 when fit_intercept is False.
            weights_final[-1] = float(estimator.intercept_)
        objective.weights_final = weights_final
        objective.weights_final_evald = True

    def use_default_tags(self):
        self._tags = {'coefficient calculation', 'gene level', 'no suboperators', 'inplace'}
