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


# class PhysicsInformedLasso(BaseEstimator, RegressorMixin):
#     """
#     Physics-Informed Lasso Regression via Coordinate Descent.
#
#     This estimator uses a custom Coefficient of Variation (CV) metric derived from
#     a physical sliding-window to assign feature-specific penalty thresholds.
#     It features an "Instant Elimination" mechanism that aggressively prunes features
#     the moment their coordinate descent update reaches zero.
#     """
#
#     def __init__(self, max_iter=1000, tol=1e-4, grid_shape=None):
#         self.max_iter = max_iter
#         self.tol = tol
#         self.grid_shape = grid_shape
#
#     def _soft_threshold(self, x, lambda_):
#         """
#         L1 proximal operator. Shrinks the partial correlation 'x' by the penalty 'lambda_'.
#         If the penalty exceeds the correlation, it forces the coefficient to exactly 0.0.
#         """
#         return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)
#
#     def get_cv(self, weights):
#         """
#         Calculates the Squared Coefficient of Variation (CV^2) as a measure of physical instability.
#         Features with high variance relative to their mean get higher CVs (and thus higher penalties).
#         """
#         weights_arr = np.array(weights)
#         std = weights_arr.std(axis=0, ddof=1)
#         mu = weights_arr.mean(axis=0)
#
#         # Suppress warnings for division by zero, safely handling perfectly stable/dead features
#         with np.errstate(divide='ignore', invalid='ignore'):
#             cv = (std ** 2) / (mu ** 2)
#             cv[mu == 0] = 0.0
#
#         return np.nan_to_num(cv)
#
#     def fit(self, X, y, sample_weights):
#         self.n_samples, self.n_features = X.shape
#         self.cached_weights_ = None
#
#         # ==========================================
#         # 1. PRECOMPUTATION & INITIALIZATION
#         # ==========================================
#         # Precompute static matrix operations to avoid O(P*N) overhead inside the inner loops
#         X_T_y = X.T @ y
#         X_sum = X.sum(axis=0)
#         norm_sq_features = np.sum(X ** 2, axis=0)
#
#         # Calculate initial physical weights and their corresponding instability penalties (CV)
#         weights = calculate_weights(X, y, sample_weights=sample_weights, grid_shape=self.grid_shape)
#         self.cached_weights_ = weights
#         cv = self.get_cv(weights[:, :-1])
#
#         # Initialize model parameters based on physical weight priors
#         self.coef_ = weights.mean(axis=0)[:-1]
#         self.intercept_ = weights.mean(axis=0)[-1]
#         residual = y - (X @ self.coef_ + self.intercept_)
#
#         # Sort features so Coordinate Descent tackles the most unstable features first
#         indices = np.argsort(cv)[::-1]
#
#         # Initialize the global threshold anchor (Maximum Correlation)
#         max_corr = np.max(np.abs(X_T_y - X_sum * self.intercept_))
#         thresholds = cv * max_corr
#
#         iteration = 0
#
#         # ==========================================
#         # 2. COORDINATE DESCENT LOOP
#         # ==========================================
#         while iteration < self.max_iter and not np.all(cv == 0):
#             max_change = 0.0
#
#             for j in indices:
#                 # Since the array is sorted descending, hitting 0 means all remaining features are 0.
#                 # We skip evaluating physically perfect features (CV=0).
#                 if cv[j] == 0:
#                     break
#
#                 old_coef = self.coef_[j]
#                 norm_sq = norm_sq_features[j]
#
#                 # Calculate partial correlation (rho) for the j-th feature
#                 rho = np.dot(X[:, j], residual) + old_coef * norm_sq
#
#                 # Apply the soft-thresholding penalty
#                 new_coef = self._soft_threshold(rho, thresholds[j]) / norm_sq
#                 self.coef_[j] = new_coef
#
#                 # ==========================================
#                 # 3. INSTANT ELIMINATION BLOCK
#                 # ==========================================
#                 if new_coef == 0:
#                     # Isolate surviving features
#                     active_mask = self.coef_ != 0
#
#                     # Recalculate physical weights strictly on the surviving subset
#                     weights = calculate_weights(
#                         X[:, active_mask], y, sample_weights=sample_weights, grid_shape=self.grid_shape
#                     )
#                     self.cached_weights_ = weights
#
#                     # Vectorized array reconstruction (re-maps local subset back to global arrays)
#                     cv.fill(0.0)
#                     cv[active_mask] = self.get_cv(weights[:, :-1])
#
#                     self.coef_.fill(0.0)
#                     self.coef_[active_mask] = weights.mean(axis=0)[:-1]
#                     self.intercept_ = weights.mean(axis=0)[-1]
#
#                     # Reset tracking variables as the objective function has fundamentally changed
#                     residual = y - (X @ self.coef_ + self.intercept_)
#                     indices = np.argsort(cv)[::-1]
#
#                     iteration = 0
#                     max_change = 1.0  # Force loop to continue since the system restarted
#                     break
#
#                 # ==========================================
#                 # 4. STANDARD RESIDUAL & TOLERANCE UPDATE
#                 # ==========================================
#                 residual -= (new_coef - old_coef) * X[:, j]
#
#                 # Calculate relative change to determine model convergence
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     change = abs(new_coef - old_coef) / old_coef
#
#                 if change > max_change:
#                     max_change = change
#
#             # ==========================================
#             # 5. END OF EPOCH RE-CENTERING
#             # ==========================================
#             # Update the unpenalized intercept based on the new coefficients
#             new_intercept = np.mean(y - X @ self.coef_)
#
#             # Shift residuals to remain mathematically accurate with the new intercept
#             residual -= (new_intercept - self.intercept_)
#             self.intercept_ = new_intercept
#
#             # Recalculate max_corr and thresholds because the intercept shifted.
#             max_corr = np.max(np.abs(X_T_y - X_sum * self.intercept_))
#             thresholds = cv * max_corr
#
#             # ==========================================
#             # 6. CONVERGENCE CHECK (DUAL GAP)
#             # ==========================================
#             if max_change <= self.tol:
#                 valid_mask = thresholds > 0
#
#                 # Calculate correlation of all features with the final residuals
#                 xt_residual = X.T[valid_mask] @ residual
#                 y_sq_sum = np.sum((y - self.intercept_) ** 2)
#
#                 # Vectorized search for the maximum dual norm scaling factor
#                 dual_norm = 0.0
#                 if np.any(valid_mask):
#                     dual_norm = np.max(np.abs(xt_residual) / thresholds[valid_mask])
#
#                 # Scale residuals to force them into the dual feasible region
#                 const_residual = residual / dual_norm if dual_norm > 1.0 else residual
#
#                 # Calculate the Fenchel duality gap using fast vector dot products
#                 primal_obj = 0.5 * np.dot(residual, residual) + np.dot(thresholds, np.abs(self.coef_))
#                 dual_obj = 0.5 * y_sq_sum - 0.5 * np.sum((y - self.intercept_ - const_residual) ** 2)
#
#                 dual_gap = primal_obj - dual_obj
#
#                 # If the gap between the primal and dual objectives is near zero, we found the global minimum
#                 if dual_gap <= self.tol * (y_sq_sum / self.n_samples):
#                     break
#
#             iteration += 1
#
#         return self

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

        # 1. AUGMENTATION: Treat intercept as a constant physical term C
        X_aug = np.column_stack((X, np.ones(n_samples)))
        total_features = n_features + 1

        # Master state trackers
        active_mask = np.ones(total_features, dtype=bool)
        self.full_coef_ = np.zeros(total_features)

        # Precompute static operations for speed
        norm_sq_features = np.sum(X_aug ** 2, axis=0)
        X_T_y = X_aug.T @ y  # Cached once; slice by active_mask each outer iter.

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

            # 1. Isolate the currently "stabilized" library
            surviving_features_mask = active_mask[:-1]
            intercept_is_active = active_mask[-1]

            # 2. Calculate physical priors ONLY for the active library --
            # slice the precomputed full Gram by the current active mask.
            # ``vcoef`` yields the per-feature score directly; the axis
            # path returns a per-window weight stack reduced by ``get_cv``.
            weights = None if is_vcoef else gram_setup.solve(active_mask)

            # Slice data for the CD run
            X_active = X_aug[:, active_mask]
            norm_sq_active = norm_sq_features[active_mask]

            # Anchor the penalty to the max correlation on the SURVIVING subspace
            # so threshold scale tracks the current problem as features drop.
            max_corr = np.max(np.abs(X_T_y[active_mask]))

            # 3. CV performs as adaptive alpha. In vcoef mode this is each
            # term's stability score (Var(gamma_0) + NC_deb)/gamma_0^2 --
            # significance plus debiased region-variation -- which prunes weak /
            # zero / unstable / spuriously-varying terms.
            active_cv = (gram_setup.score(active_mask) if is_vcoef
                         else self.get_cv(weights))

            # Tackle the most physically unstable feature first so unstable
            # terms get shrunk to zero before they pollute the residual.
            active_thresholds = active_cv * max_corr
            # active_thresholds = active_cv
            # active_thresholds = active_cv * norm_sq_active

            cv_order = np.argsort(active_cv)[::-1]
            # cv_order = np.argsort(active_thresholds)[::-1]

            # Initialize coefficients from a single global weighted-OLS on
            # the full dataset rather than the mean over per-window OLS
            # coefficients. The per-window mean is biased toward zero on
            # heterogeneous data (e.g. KdV solitons, where most windows see
            # ~0 signal so the mean is shrunk by ``M_signal / M``); the
            # global OLS is unbiased.
            sw_active = (sample_weights if sample_weights is not None
                          else np.ones(n_samples))
            try:
                XTWX_full = X_active.T @ (sw_active[:, None] * X_active)
                XTWy_full = X_active.T @ (sw_active * y)
                active_coef = np.linalg.solve(XTWX_full, XTWy_full)
            except np.linalg.LinAlgError:
                # ``vcoef`` has no per-window stack to average; fall back to
                # a zero start (CD recovers it) instead of weights.mean.
                active_coef = (np.zeros(int(active_mask.sum())) if is_vcoef
                               else weights.mean(axis=0))

            residual = y - (X_active @ active_coef)

            # =================================================================
            # INNER LOOP: Pure Coordinate Descent on the Stabilized Library
            # =================================================================
            cd_iteration = 0
            cd_iters_executed = 0
            killed_feature = False
            while cd_iteration < self.max_iter:
                cd_iters_executed += 1
                max_change = 0.0

                for j in cv_order:
                    old_coef = active_coef[j]
                    norm_sq = norm_sq_active[j]

                    # Partial correlation rho
                    rho = np.dot(X_active[:, j], residual) + old_coef * norm_sq

                    # Apply CV-based soft thresholding (Penalty is FIXED for this inner loop)
                    new_coef = self._soft_threshold(rho, active_thresholds[j]) / norm_sq

                    # Standard residual update
                    residual -= (new_coef - old_coef) * X_active[:, j]
                    active_coef[j] = new_coef

                    if new_coef == 0 and old_coef != 0:
                        # A feature just died — hand control back to the outer
                        # loop so CVs/anchor/thresholds get recomputed on the
                        # smaller library before doing any more CD work.
                        killed_feature = True
                        break

                    with np.errstate(divide='ignore', invalid='ignore'):
                        change = abs(new_coef - old_coef)
                        if old_coef != 0:
                            change /= abs(old_coef)
                        if change > max_change:
                            max_change = change

                if killed_feature:
                    break

                # Inner loop convergence check
                if max_change <= self.tol:
                    # You can add your Dual Gap check here if desired,
                    # but max_change is usually sufficient for the inner loop
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
            sw_active = (sample_weights if sample_weights is not None
                          else np.ones(n_samples))
            X_final = X_aug[:, active_mask]
            try:
                XTWX_final = X_final.T @ (sw_active[:, None] * X_final)
                XTWy_final = X_final.T @ (sw_active * y)
                refit = np.linalg.solve(XTWX_final, XTWy_final)
                self.full_coef_[active_mask] = refit
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
        # objective._cached_sw_weights = estimator.cached_weights_
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


