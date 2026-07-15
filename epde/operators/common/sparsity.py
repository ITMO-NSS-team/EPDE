#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:35:18 2021

@author: mike_ubuntu
"""

import numpy as np
from sklearn.linear_model import Lasso

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.structure.main_structures import Equation
import time
from sklearn.base import BaseEstimator, RegressorMixin
# import seaborn as sns
import matplotlib.pyplot as plt
from epde.operators.common.stability import (calculate_weights, GramSetup,
                                              VaryingCoefSetup)
from epde import _loop_stats


def _minmax_normalize_1d(values: np.ndarray) -> np.ndarray:
    """Rescale a 1-D vector to [-1, 1]. Constant vectors map to zeros."""
    vmin = values.min()
    vmax = values.max()
    if vmax == vmin:
        return np.zeros_like(values, dtype=float)
    return 2.0 * (values - vmin) / (vmax - vmin) - 1.0


def _minmax_normalize_columns(features: np.ndarray) -> np.ndarray:
    """Rescale each column of a 2-D feature matrix to [-1, 1].
    Constant columns map to zeros (no informative variance for L1)."""
    out = np.empty(features.shape, dtype=float)
    for j in range(features.shape[1]):
        col = features[:, j]
        cmin = col.min()
        cmax = col.max()
        if cmax == cmin:
            out[:, j] = 0.0
        else:
            out[:, j] = 2.0 * (col - cmin) / (cmax - cmin) - 1.0
    return out


class PhysicsInformedLasso(BaseEstimator, RegressorMixin):
    """
    Physics-Informed Lasso using Coordinate Descent and Adaptive CV-Penalties.

    Features:
    - Adaptive: Replaces alpha with Coefficient of Variation (CV) from physical priors.
    - Scale-Invariant: Anchors penalties to the maximum correlation [X.T @ y].
    - Augmented: Treats the intercept as a penalized feature based on its own stability.
    - Aggressive: Instant elimination of features that hit zero during optimization.
    """

    def __init__(self, max_iter=1000, tol=1e-4, grid_shape=None,
                 main_var: str = None):
        self.max_iter = max_iter
        self.tol = tol
        self.grid_shape = grid_shape
        # Threaded through to ``VaryingCoefSetup`` so the basis-mode
        # resolver picks the equation's own primary variable when
        # multi-var systems use different scales per equation.
        self.main_var = main_var
        self.coef_ = None
        self.full_coef_ = None  # Includes the intercept

    def _soft_threshold(self, x, lambda_):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)

    def get_cv(self, weights):
        """Per-feature CV-stability metric for the axis backup path:
        ``(std / mean)^2 = var / mu^2`` across the sliding windows.

        The squared coefficient of variation of each feature's per-window
        weight. It blows up (large CV) for features whose fitted coefficient
        is unstable or near-zero-mean across horizons, so the
        ``active_thresholds = cv * max_corr`` step in
        :meth:`PhysicsInformedLasso.fit` prunes them first. The default
        ``gram_mode='vcoef'`` path does not call this -- it scores via
        ``VaryingCoefSetup.score`` instead.
        """
        weights_arr = np.asarray(weights)
        with np.errstate(divide='ignore', invalid='ignore'):
            std = weights_arr.std(axis=0, ddof=1)
            mu = weights_arr.mean(axis=0)
            cv = (std ** 2) / (mu ** 2)
            cv[mu == 0] = 0.0
        return np.nan_to_num(cv)

    @_loop_stats.timed('PhysicsInformedLasso.fit')
    def fit(self, X, y, sample_weights=None, gram_setup=None):
        n_samples, n_features = X.shape

        total_features = n_features + 1  # + intercept (constant term C)

        # Master state trackers
        active_mask = np.ones(total_features, dtype=bool)
        self.full_coef_ = np.zeros(total_features)

        # Precompute the augmented Gram blocks ONCE. The RFE outer loop and the
        # coordinate descent below run entirely in Gram space (slicing these by
        # ``active_mask``), so the expensive per-iteration ``X^T W X`` re-matmul
        # and the per-coordinate O(N) dot products collapse to O(P^2) / O(P) ops
        # -- the dominant cost on large-N systems. The intercept (a constant ones
        # column) is folded in block-wise so the full N x (P+1) ``X_aug`` is never
        # materialized. A single WEIGHTED Gram drives EVERYTHING (CD selection,
        # anchor, OLS init, relaxed refit); a sub-block of a Gram equals the Gram of
        # the sub-columns, so this is exact. The sample weights are the grid
        # ``g_func``, which is 1 in the interior and 0 in the boundary margin -- and
        # the boundary zeros are masked out upstream, so by DEFAULT W = I and this
        # is the plain Gram. When a non-uniform g_func IS supplied, weighting the
        # CD too means boundary downweighting applies to term SELECTION, not just
        # the final magnitudes (which the relaxed refit already weighted).
        Xf = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        sw = (np.asarray(sample_weights, dtype=float).reshape(-1)
              if sample_weights is not None else np.ones(n_samples))

        _wX = sw[:, None] * Xf
        G_w = np.empty((total_features, total_features))
        G_w[:n_features, :n_features] = Xf.T @ _wX
        _ws = _wX.sum(axis=0)  # = X^T w
        G_w[:n_features, -1] = _ws
        G_w[-1, :n_features] = _ws
        G_w[-1, -1] = float(sw.sum())
        Gy_w = np.empty(total_features)
        Gy_w[:n_features] = Xf.T @ (sw * y)
        Gy_w[-1] = float((sw * y).sum())

        norm_sq_features = np.diag(G_w).copy()  # weighted column norms^2
        X_T_y = Gy_w                            # raw-target anchor: max|X^T W y|

        # Pre-build the full sliding-window Gram matrix ONCE. The outer
        # RFE loop below will slice it by ``active_mask`` per iteration
        # instead of re-running the expensive ``X^T diag(w) X`` matmul on
        # the surviving columns. The math is exact: a sub-block of the
        # full Gram equals the Gram of the corresponding sub-columns.
        #
        # Tier 3 fast path: when the caller (EqRPS's term-sweep) has
        # already built a per-target ``GramSetup`` view from the
        # super-Gram, reuse it -- saves the windowed matmul that
        # otherwise repeats for every candidate target_idx in one sweep.
        if gram_setup is None:
            if global_var.gram_mode == 'vcoef':
                gram_setup = VaryingCoefSetup(
                    X, y, sample_weights, self.grid_shape,
                    main_var=self.main_var)
            else:  # 'axis' backup
                gram_setup = GramSetup(X, y, sample_weights, self.grid_shape)

        # Varying-coefficient mode returns a per-feature stability score
        # directly (no per-window weight stack), so the in-fit CV-threshold
        # path branches on it below.
        is_vcoef = getattr(gram_setup, 'is_vcoef', False)

        outer_iteration = 0
        max_outer_iters = total_features  # Max possible eliminations
        outer_iters_executed = 0

        # =================================================================
        # OUTER LOOP: Library Stabilization & RFE (Recursive Feature Elimination)
        # =================================================================
        while outer_iteration < max_outer_iters:
            outer_iters_executed += 1

            # ``vcoef`` yields the per-feature score directly; the axis path
            # returns a per-window weight stack reduced by ``get_cv``.
            weights = None if is_vcoef else gram_setup.solve(active_mask)

            # Slice the (weighted) precomputed Gram by the current active set.
            active_idx = np.where(active_mask)[0]
            Gua = G_w[np.ix_(active_idx, active_idx)]   # active x active (weighted)
            Gya = Gy_w[active_idx]
            norm_sq_active = norm_sq_features[active_idx]

            # Anchor the penalty to the max WEIGHTED correlation on the surviving
            # subspace. 'anchor_on_residual': X_active^T W (y - X_aug @ full_coef_)
            # = Gy_w[active] - G_w[active, :] @ full_coef_ (no N; full_coef_ = 0 on
            # the first pass so this reduces to max|X^T W y| then).
            if getattr(global_var, 'anchor_on_residual', False):
                max_corr = np.max(np.abs(Gya - G_w[active_idx, :] @ self.full_coef_))
            else:
                max_corr = np.max(np.abs(X_T_y[active_mask]))

            # CV performs as adaptive alpha. In vcoef mode this is each term's
            # instability score NC/gamma_0^2 (biased non-constant energy), which
            # prunes weak / zero / unstable / spuriously-varying terms.
            active_cv = (gram_setup.score(active_mask)
                         if is_vcoef else self.get_cv(weights))

            # Tackle the most physically unstable feature first.
            active_thresholds = active_cv * max_corr
            cv_order = np.argsort(active_cv)[::-1]

            # Initialize coefficients from a single global WEIGHTED-OLS, in Gram
            # space: solve(G_w[active, active], Gy_w[active]). The axis path falls
            # back to its per-window mean when the system is singular.
            try:
                active_coef = np.linalg.solve(
                    G_w[np.ix_(active_idx, active_idx)], Gy_w[active_idx])
            except np.linalg.LinAlgError:
                active_coef = (np.zeros(active_idx.size) if is_vcoef
                               else weights.mean(axis=0))

            # Running product q = (X_active^T X_active) @ active_coef, maintained
            # incrementally so each coordinate update is O(active) instead of the
            # legacy O(N) residual update: rho_j = X_j^T(y - X_active beta) +
            # beta_j*||X_j||^2 = Gya[j] - q[j] + beta_j*norm_sq.
            q = Gua @ active_coef

            # =================================================================
            # INNER LOOP: Pure Coordinate Descent on the Stabilized Library
            # =================================================================
            cd_iteration = 0
            cd_iters_executed = 0
            killed_feature = False
            # errstate hoisted OUT of the inner loop (was entered per-coordinate).
            with np.errstate(divide='ignore', invalid='ignore'):
                while cd_iteration < self.max_iter:
                    cd_iters_executed += 1
                    max_change = 0.0

                    for j in cv_order:
                        old_coef = active_coef[j]
                        norm_sq = norm_sq_active[j]

                        # rho = X_j^T(y - X_active beta) + beta_j*||X_j||^2,
                        # computed in Gram space (no O(N) dot).
                        rho = Gya[j] - q[j] + old_coef * norm_sq
                        new_coef = self._soft_threshold(
                            rho, active_thresholds[j]) / norm_sq

                        delta = new_coef - old_coef
                        active_coef[j] = new_coef
                        if delta != 0.0:
                            q += delta * Gua[:, j]   # O(active) Gram-space update

                        if new_coef == 0 and old_coef != 0:
                            # A feature died -- hand control to the outer loop so
                            # CVs/anchor/thresholds recompute on the smaller set.
                            killed_feature = True
                            break

                        change = abs(delta)
                        if old_coef != 0:
                            change /= abs(old_coef)
                        if change > max_change:
                            max_change = change

                    if killed_feature:
                        break

                    # Inner loop convergence check
                    if max_change <= self.tol:
                        break

                    cd_iteration += 1
            _loop_stats.record('PhysicsInformedLasso.CD_inner', cd_iters_executed, self.max_iter)

            # =================================================================
            # THE BRIDGE: Check for Eliminations
            # =================================================================
            # Map the inner loop results back to the master array
            self.full_coef_.fill(0.0)
            self.full_coef_[active_mask] = active_coef

            # Did the CD optimizer kill any features?
            new_active_mask = self.full_coef_ != 0

            # If the library didn't change, we have reached global stability!
            if np.array_equal(active_mask, new_active_mask):
                break

            # Otherwise, update the mask and restart the Outer Loop to recalculate CVs
            active_mask = new_active_mask
            outer_iteration += 1

            # Emergency break if everything died. `weights` still references
            # the prior (now-stale) mask, so drop it instead of caching.
            if not np.any(active_mask):
                weights = None
                break

        _loop_stats.record('PhysicsInformedLasso.RFE_outer', outer_iters_executed, max_outer_iters)
        self.cached_weights_ = weights
        # Per-active-term stability scores on the converged mask, summed as the
        # stability objective in fitness. ``None`` for the axis backup path.
        self.cached_vc_score_ = (gram_setup.score(active_mask)
                                 if is_vcoef else None)

        # Relaxed-LASSO refit: replace the surviving CD-output
        # coefficients with a single global weighted-OLS on
        # ``X[:, active_mask]``. Sparsity decisions (which features
        # survived) are preserved; only the magnitudes become unbiased
        # global estimates.
        if np.any(active_mask):
            active_idx = np.where(active_mask)[0]
            try:
                self.full_coef_[active_mask] = np.linalg.solve(
                    G_w[np.ix_(active_idx, active_idx)], Gy_w[active_idx])
            except np.linalg.LinAlgError:
                pass  # singular -> keep CD result

        # Map back to standard sklearn attributes
        self.coef_ = self.full_coef_[:-1]
        self.intercept_ = self.full_coef_[-1]

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class LASSOSparsity(CompoundOperator):
    """
    The operator, which applies LASSO regression to the equation object to detect the 
    valuable term coefficients.
    
    Notable attributes:
    -------------------
        
    params : dict
        Inhereted from the ``CompoundOperator`` class. 
        Parameters of the operator; main parameters: 
            
            sparsity - value of the sparsity constant in the LASSO operator;
            
    g_fun : np.ndarray or None:
        values of the function, used during the weak derivatives estimations. 
            
    Methods:
    -----------
    apply(equation)
        calculate the coefficients of the equation, that will be stored in the equation.weights np.ndarray.    
        
    """
    key = 'LASSOBasedSparsity'

    @_loop_stats.timed('LASSOSparsity.apply')
    def apply(self, objective : Equation, arguments : dict):
        """
        Apply the operator, to fit the LASSO regression to the equation object to detect the 
        valueable terms. In the Equation class, a term is selected to represent the right part of
        the equation, and its values are used here as the target, and the values of the other 
        terms are utilizd as the features. The method does not return the vector of coefficients, 
        but rather assigns the result to the equation attribute ``equation.weights_internal``
        
        Parameters:
        ------------
        equation : Equation object
            the equation object, to that the coefficients are obtained.
            
        Returns:
        ------------
        None
        """
        # print(f'Metaparameter: {objective.metaparameters}, objective.metaparameters[("sparsity", objective.main_var_to_explain)]')
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        estimator = Lasso(alpha=objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
                          copy_X=True, fit_intercept=True, max_iter=1000,
                          positive=False, precompute=False, random_state=None,
                          selection='random', tol=0.0001, warm_start=False)

        _, target, features = objective.evaluate(normalize = True, return_val = False)

        # Legacy LASSO step: min-max-rescale target + each feature
        # column to [-1, 1] before the L1 fit, so the alpha penalty is
        # comparable across features whose physical magnitudes span
        # many orders. The downstream ``LinRegBasedCoeffsEquation``
        # refit (see coeff_calculation.py) re-evaluates the surviving
        # terms with ``term.evaluate(False)`` to recover physically-
        # scaled coefficients on un-normalised features. MOEA/D
        # optimises the LASSO alpha (via the metaparameter mutation),
        # so we do not need to rescale it here -- the search will
        # discover effective values for the normalised feature space.
        try:
            if features is not None and np.all(np.isfinite(features)):
                features = _minmax_normalize_columns(features)
            if target is not None and np.all(np.isfinite(target)):
                target = _minmax_normalize_1d(target)
        except Exception:
            # Defensive: any normalisation hiccup falls through to the
            # degenerate-features path below rather than aborting the
            # whole sparsity step.
            pass

        self.g_fun_vals = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask]

        n_features = features.shape[1] if (features is not None and hasattr(features, 'ndim') and features.ndim > 1) else 0
        if features is None or not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
            # Degenerate features (e.g. constant column triggering divide-by-zero
            # in objective.evaluate's min-max normalisation). Fall back to a
            # zero-weight assignment so the candidate is treated as "empty"
            # rather than aborting the whole optimisation run.
            coef = np.zeros(n_features)
            intercept = 0.0
        else:
            estimator.fit(features, target, self.g_fun_vals)
            coef = estimator.coef_
            intercept = estimator.intercept_
        objective.weights_internal = coef
        objective.weights_internal_evald = True
        objective.weights_final = np.append([weight for weight in coef if weight != 0], intercept)
        objective.weights_final_evald = True
        # Flag the equation for the un-normalised LinearRegression refit
        # performed by ``LinRegBasedCoeffsEquation`` -- see
        # ``epde/operators/common/coeff_calculation.py``. VWSRSparsity
        # does NOT set this marker; only the LASSO path opts into the
        # legacy two-step (min-max LASSO + linreg-on-survivors) flow.
        objective._legacy_refit_pending = True
        # Note: _eval_cache is intentionally NOT wiped here. The cache stores
        # (value, target, features) tuples keyed on (normalize, return_val,
        # grids is None); none of those depend on the weights this operator
        # just updated. Structural mutations call ``Equation.reset_state``
        # which performs the wipe at the right moment.


    def use_default_tags(self):
        self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}


class VWSRSparsity(CompoundOperator):
    """
    Variance-Weighted Sparse Regression operator.

    Mirrors :class:`LASSOSparsity` but swaps the sklearn ``Lasso`` estimator
    for :class:`PhysicsInformedLasso`, which derives feature-specific L1
    penalties from the squared coefficient of variation of sliding-window
    fits. Used as the regression step of the "new" pipeline.
    """
    key = 'VWSRBasedSparsity'

    @_loop_stats.timed('VWSRSparsity.apply')
    def apply(self, objective : Equation, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        estimator = PhysicsInformedLasso(
            grid_shape=global_var.grid_cache.inner_shape,
            main_var=objective.main_var_to_explain)

        self.g_fun_vals = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask]

        # Tier 3 fast path: if the upstream EqRPS term-sweep has
        # precomputed a super-Gram (and the cached Z over all terms),
        # derive ``target`` / ``features`` plus the per-target
        # ``GramSetup`` by slicing -- skips both objective.evaluate's
        # vstack + transpose AND the windowed XTWX matmul.
        gram_super = getattr(objective, '_gram_super', None)
        if gram_super is not None:
            Z = gram_super['Z']
            t = objective.target_idx
            target = Z[:, t]
            feature_indexes = [i for i in range(Z.shape[1]) if i != t]
            features = Z[:, feature_indexes]
            if gram_super.get('mode') == 'vcoef':
                gram_setup = VaryingCoefSetup.from_full(gram_super, t)
            else:
                gram_setup = GramSetup.from_full(gram_super, t)
        else:
            _, target, features = objective.evaluate(normalize=True, return_val=False)
            gram_setup = None
        estimator.fit(features, target, self.g_fun_vals, gram_setup=gram_setup)
        objective.weights_internal = np.array([*estimator.coef_, estimator.intercept_])
        objective.weights_internal_evald = True
        objective.weights_final = np.array([weight for weight in objective.weights_internal if weight != 0])
        objective.weights_final_evald = True
        objective._cached_sw_weights = estimator.cached_weights_
        objective._cached_vc_score = estimator.cached_vc_score_
        # See LASSOSparsity.apply: _eval_cache survives a weights update;
        # only structural resets via ``Equation.reset_state`` should wipe it.

    def use_default_tags(self):
        self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}


