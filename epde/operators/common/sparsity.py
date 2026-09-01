#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:35:18 2021

@author: mike_ubuntu
"""

import numpy as np
from sklearn.linear_model import Lasso
import time
from sklearn.base import BaseEstimator, RegressorMixin

from functools import partial
# import matplotlib.pyplot as plt

import epde.globals as global_var
from epde.interface.search_config import active_config
from epde.operators.utils.template import CompoundOperator
from epde.structure.main_structures import Equation
from epde.operators.common.stability import GramSetup, VaryingCoefSetup  #(calculate_weights, GramSetup,
                                                                         # VaryingCoefSetup)
from epde.operators.common.survival import (chi2_scores, heterogeneity_scores,
                                            survival_scores, tile_scores)
from epde import _loop_stats

# The keep-rule's per-term score, by instability metric. ONE estimator drives
# both the L1 threshold here and the Instability Pareto axis: a term the
# objective calls unstable is the term the threshold prunes hardest, and no
# variant may be wired into one side only. 'vcoef' and 'cv' are not listed
# because they come from the Gram setup itself (``VaryingCoefSetup.score`` /
# ``get_cv``) rather than from the basis-free estimators in ``survival.py``.
_KEEP_RULE_ESTIMATORS = {'chi2': chi2_scores, 'het': heterogeneity_scores,
                         'survival': survival_scores, 'tile': tile_scores}


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


def cv_scores(weights):
    """Per-feature CV-stability metric for the axis backup path:
    ``(std / mean)^2 = var / mu^2`` across the sliding windows.

    The squared coefficient of variation of each feature's per-window weight.
    It blows up (large CV) for features whose fitted coefficient is unstable
    or near-zero-mean across horizons, so the
    ``active_thresholds = cv * max_corr`` step in
    :meth:`PhysicsInformedLasso.fit` prunes them first. Only the ``'cv'``
    instability metric reaches it -- ``'vcoef'`` scores via
    ``VaryingCoefSetup.score`` and the basis-free estimators need no window
    stack at all.

    Module level so every consumer of the ``'cv'`` metric shares ONE
    reduction of the window stack (see :func:`instability_scores`).
    """
    weights_arr = np.asarray(weights)
    with np.errstate(divide='ignore', invalid='ignore'):
        std = weights_arr.std(axis=0, ddof=1)
        mu = weights_arr.mean(axis=0)
        cv = (std ** 2) / (mu ** 2)
        cv[mu == 0] = 0.0
    return np.nan_to_num(cv)


def instability_scores(metric, X, y, sw, grid_shape, active_mask, n_features,
                       *, gram_setup=None, cv_reducer=None, weights=None):
    """Per-active-COLUMN instability score for a support mask.

    THE SAME estimator the ``Instability`` objective uses, and the single
    dispatch every selection rule in the tree goes through. The sparsity step
    and the Pareto axis must not disagree about what makes a term unstable: a
    search that prunes by one statistic and then scores the survivors by
    another is optimizing against its own regularizer.
    ``objectives.instability_metric`` therefore picks both, and a new
    selection rule (see ``subset_selection.KneeSparsity``) gets the same
    estimator by calling this rather than choosing one of its own.

    ``'vcoef'`` and ``'cv'`` read the score off the Gram setup that produced
    them (``'cv'`` through ``cv_reducer``, the caller's reduction of the
    window stack); the basis-free estimators are computed here from the
    active columns. Alignment matters -- callers index the result by position
    within ``active_mask``, and the intercept is the last column -- so the
    ones column is appended EXPLICITLY and the estimators are told
    ``fit_intercept=False``: they drop the intercept they add themselves,
    which would leave the score vector one entry short.
    """
    if metric == 'vcoef':
        return gram_setup.score(active_mask)
    if metric == 'cv':
        return cv_reducer(weights)
    cols = np.where(active_mask)[0]
    feat_cols = cols[cols < n_features]
    Xa = X[:, feat_cols]
    if cols.size and cols[-1] == n_features:
        Xa = np.hstack([Xa, np.ones((X.shape[0], 1))])
    return _KEEP_RULE_ESTIMATORS[metric](Xa, y, sw, grid_shape,
                                         fit_intercept=False)


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
        # ``full_coef_`` IS the unified coefficient layout (see
        # ``Equation._validate_weight_layout``): one entry per feature column in
        # order, then the augmented intercept, zeros retained. ``coef_`` /
        # ``intercept_`` are the sklearn-shaped views of it, kept for
        # compatibility; callers that store weights on an Equation want
        # ``full_coef_``.
        self.full_coef_ = None  # Includes the intercept

    def _soft_threshold(self, x, lambda_):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)

    def get_cv(self, weights):
        """This estimator's view of :func:`cv_scores`; see there."""
        return cv_scores(weights)

    # def get_cv(self, weights):
    #     weights_arr = np.asarray(weights)
    #     q1, q3 = np.percentile(weights_arr, [25, 75], axis=0)
    #     spread = (q3 - q1) / 1.349  # IQR/1.349 ≈ σ for Gaussian
    #     center = np.median(weights_arr, axis=0)
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         cv = spread ** 2 / (center ** 2 + spread ** 2)
    #     return np.nan_to_num(cv)

    @_loop_stats.timed('PhysicsInformedLasso.fit')
    def fit(self, X, y, sample_key=None, grid_shape=None, sample_weights=None, gram_setup=None):
        # ``grid_shape`` is the trajectory's INNER GRID shape (e.g. (nt, nx)),
        # which the caller reads from ``samples_manager.inner_shapes``. Falling
        # back to ``sample_weights.shape`` only when nothing was supplied: the
        # g_func weights arrive already boundary-masked and FLATTENED
        # (``Domain.g_func_masked_val``), so that fallback yields (N_interior,)
        # and silently collapses every multi-dimensional grid to one axis --
        # correct only for 1-D (ODE) data.
        if grid_shape is None and sample_weights is not None:
            grid_shape = np.asarray(sample_weights).shape

        n_samples, n_features = X.shape

        # 1. AUGMENTATION: Treat intercept as a constant physical term C
        # X_aug = np.column_stack((X, np.ones(n_samples)))
        total_features = n_features + 1

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
        # ONE statistic drives the L1 threshold below and the Instability
        # Pareto axis; the Gram is built only when that statistic is computed
        # from one (``None`` for the basis-free estimators).
        objectives_cfg = active_config().objectives
        metric = objectives_cfg.instability_metric
        gram_mode = objectives_cfg.gram_mode
        # Hoisted out of the outer RFE loop below: it used to be re-read per
        # iteration, per trajectory, per equation, per generation.
        anchor_on_residual = objectives_cfg.anchor_on_residual
        if gram_setup is None and gram_mode is not None:
            if gram_mode == 'vcoef':
                # ``sample_key`` MUST be threaded through: it selects the
                # trajectory whose field the Taylor-microscale basis resolution
                # reads. Without it ``resolve_vc_modes_from_input`` indexed the
                # per-sample dict with None, raised, was swallowed, and every
                # equation silently fell back to the fixed K=3 basis.
                gram_setup = VaryingCoefSetup(
                    X, y, sample_weights, grid_shape,
                    main_var=self.main_var, sample_key=sample_key)
            else:  # 'axis' backup
                gram_setup = GramSetup(X, y, sample_weights, grid_shape)
        # ``None`` when the metric is basis-free ('chi2', 'het', 'tile',
        # 'survival'): those score straight from the active columns, so
        # neither the varying-coefficient mode solve nor the sliding-window
        # stack is built at all.

        is_vcoef = getattr(gram_setup, 'is_vcoef', False)

        outer_iteration = 0
        max_outer_iters = total_features  # Max possible eliminations
        outer_iters_executed = 0

        # =================================================================
        # OUTER LOOP: Library Stabilization & RFE (Recursive Feature Elimination)
        # =================================================================
        while outer_iteration < max_outer_iters:
            outer_iters_executed += 1

            # The per-window weight stack is the 'cv' metric's raw material
            # (``get_cv`` reduces it to var/mu^2). Every other metric scores
            # without it, so the solve is skipped rather than computed and
            # thrown away.
            weights = (gram_setup.solve(active_mask) if metric == 'cv'
                       else None)

            # Slice the (weighted) precomputed Gram by the current active set.
            active_idx = np.where(active_mask)[0]
            Gua = G_w[np.ix_(active_idx, active_idx)]   # active x active (weighted)
            Gya = Gy_w[active_idx]
            norm_sq_active = norm_sq_features[active_idx]

            # Anchor the penalty to the max WEIGHTED correlation on the surviving
            # subspace. 'anchor_on_residual': X_active^T W (y - X_aug @ full_coef_)
            # = Gy_w[active] - G_w[active, :] @ full_coef_ (no N; full_coef_ = 0 on
            # the first pass so this reduces to max|X^T W y| then).
            if anchor_on_residual:
                max_corr = np.max(np.abs(Gya - G_w[active_idx, :] @ self.full_coef_))
            else:
                max_corr = np.max(np.abs(X_T_y[active_mask]))

            # The instability score performs as an adaptive alpha: it prunes
            # weak / zero / unstable / spuriously-varying terms hardest. Which
            # statistic says "unstable" is the search-wide instability metric,
            # the one the Pareto axis reads.
            active_cv = self._keep_rule_scores(metric, gram_setup, weights, Xf, y,
                                            sw, grid_shape, active_mask,
                                            n_features)

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
                active_coef = (weights.mean(axis=0) if weights is not None
                               else np.zeros(active_idx.size))

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

    def _keep_rule_scores(self, metric, gram_setup, weights, X, y, sw,
                       grid_shape, active_mask, n_features):
        """This estimator's view of :func:`instability_scores` -- the L1
        threshold's per-column scale. Kept as a method because the ``'cv'``
        branch reduces the window stack through :meth:`get_cv`, which is
        this class's own reduction."""
        return instability_scores(metric, X, y, sw, grid_shape, active_mask,
                                  n_features, gram_setup=gram_setup,
                                  cv_reducer=self.get_cv, weights=weights)

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

    #: Interval the per-equation ``('sparsity', var)`` metaparameter -- this
    #: operator's ``alpha`` -- is SEEDED from when a population is created
    #: (log-uniform between the ends; see
    #: ``moeadd.population_constr.SystemsPopulationConstructor.create``). It is
    #: an INITIAL range only: MetaparameterMutation and MetaparamerCrossover
    #: move alpha freely afterwards, outside the interval included.
    #:
    #: It lives here, and not in the search-space configuration, because it is
    #: a parameter of this estimator rather than a property of the space of
    #: equations being searched -- ``LASSOSparsity.apply`` below is the only
    #: reader of the value in the whole tree.
    initial_sparsity_interval = (1e-4, 2.5)

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
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        # ``fit_intercept=False`` + an explicitly appended ones column: the
        # free coefficient becomes an ORDINARY penalized column, so ``alpha``
        # can drive it to exactly zero the same way PhysicsInformedLasso's
        # augmented intercept can (sparsity.py: "Treats the intercept as a
        # penalized feature"). sklearn's own ``fit_intercept=True`` never
        # penalizes it, which left the two pipelines disagreeing about what
        # "this equation has no constant term" means.
        estimator = Lasso(alpha=objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
                          copy_X=True, fit_intercept=False, max_iter=1000,
                          positive=False, precompute=False, random_state=None,
                          selection='random', tol=0.0001, warm_start=False)

        targets, features = objective.evaluate()

        # Multisample: the legacy path fits ONE Lasso on all trajectories
        # stacked row-wise -- the same convention
        # ``LinRegBasedCoeffsEquation._legacy_evaluate_nonzero`` uses for the
        # follow-up refit, so both steps see identical row ordering.
        keys = list(global_var.samples_manager.inner_shapes.keys())
        target = (np.concat([np.asarray(targets[key]).reshape(-1) for key in keys], axis=0)
                  if targets is not None else None)
        features = (np.concat([np.asarray(features[key]) for key in keys], axis=0)
                    if features is not None else None)
        self.g_fun_vals = np.concat(
            [np.asarray(global_var.samples_manager.gFunc('dm')[key]).reshape(-1)
             for key in keys], axis=0)

        # Legacy LASSO step: min-max-rescale target + each feature
        # column to [-1, 1] before the L1 fit, so the alpha penalty is
        # comparable across features whose physical magnitudes span
        # many orders. The downstream ``LinRegBasedCoeffsEquation``
        # refit (see coeff_calculation.py) re-evaluates the surviving
        # terms with ``term.evaluate()`` to recover physically-
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

        # One slot per non-target term plus the trailing intercept -- the
        # layout every producer now emits (Equation._validate_weight_layout).
        n_slots = len(objective.structure)
        if features is None or not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
            # Degenerate features (e.g. constant column triggering divide-by-zero
            # in objective.evaluate's min-max normalisation). Fall back to a
            # zero-weight assignment so the candidate is treated as "empty"
            # rather than aborting the whole optimisation run.
            coef = np.zeros(n_slots)
        else:
            # The ones column goes on AFTER the min-max rescale: run through it
            # and ``_minmax_normalize_columns`` would map the constant column to
            # all-zeros (its cmax == cmin branch), so the intercept would die
            # trivially instead of being judged on its own contribution.
            features_aug = np.hstack([features, np.ones((features.shape[0], 1))])
            estimator.fit(features_aug, target, self.g_fun_vals)
            coef = estimator.coef_
        objective.weights_internal = coef
        objective.weights_internal_evald = True
        # A SELECTION decision only: this fit ran on min-max-rescaled features
        # and target, so ``coef[-1]`` says whether a free constant is needed,
        # not how big it is. LinRegBasedCoeffsEquation supplies the physical
        # magnitudes below and honours the zero/non-zero verdict recorded here.
        objective.weights_final = coef.copy()
        objective.weights_final_evald = True
        # Flag the equation for the un-normalised LinearRegression refit
        # performed by ``LinRegBasedCoeffsEquation`` -- see
        # ``epde/operators/common/coeff_calculation.py``. VWSRSparsity
        # does NOT set this marker; only the LASSO path opts into the
        # legacy two-step (min-max LASSO + linreg-on-survivors) flow.
        objective._legacy_refit_pending = True
        # Note: _eval_cache is intentionally NOT wiped here. It stores only the
        # WIDE (target, features) pair, keyed on target_idx -- ``evaluate``
        # refuses to cache its ``active_only`` branch precisely because that one
        # reads the weights this operator just updated. Structural mutations
        # call ``Equation.reset_state``, which performs the wipe at the right
        # moment.


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

    #: PhysicsInformedLasso derives one penalty per feature from the data, so
    #: this operator never reads the ``('sparsity', var)`` metaparameter. The
    #: degenerate interval says exactly that: seeding is a no-op and the
    #: metaparameter keeps the neutral 1.0 of ``main_structures``'s defaults
    #: instead of a random alpha nothing consumes.
    initial_sparsity_interval = (1.0, 1.0)

    @_loop_stats.timed('VWSRSparsity.apply')
    def apply(self, objective : Equation, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        estimator = PhysicsInformedLasso(main_var=objective.main_var_to_explain)

        # Tier 3 fast path: if the upstream EqRPS term-sweep has
        # precomputed a super-Gram (and the cached Z over all terms),
        # derive ``target`` / ``features`` plus the per-target
        # ``GramSetup`` by slicing -- skips both objective.evaluate's
        # vstack + transpose AND the windowed XTWX matmul. Everything is
        # keyed by trajectory ID: each sample owns its own Z, its own
        # super-Gram and hence its own per-target view.
        gram_super = getattr(objective, '_gram_super', None)
        if gram_super is not None:
            Z = gram_super['Z']
            t = objective.target_idx
            target = {key: Z_key[:, t] for key, Z_key in Z.items()}
            features = {key: Z_key[:, [i for i in range(Z_key.shape[1]) if i != t]]
                        for key, Z_key in Z.items()}
            super_mode = gram_super.get('mode')
            if super_mode == 'vcoef':
                gram_setups = VaryingCoefSetup.from_full(gram_super, t)
            elif super_mode == 'axis':
                gram_setups = GramSetup.from_full(gram_super, t)
            else:
                gram_setups = {}      # basis-free metric: Z only
        else:
            target, features = objective.evaluate()
            gram_setups = {}

        assert isinstance(target, dict) and isinstance(features, dict), (
            'Unexpected behavior: target and features, obtained from obj.evaluate '
            f'have to be dicts, instead got: {type(target)} and {type(features)}.')

        gfuncs = global_var.samples_manager.gFunc('dm')
        grid_shapes = global_var.samples_manager.inner_shapes

        sampled_full_coefs = []
        sampled_sw_weights = {}
        sampled_vc_scores = {}

        for key in target.keys():
            estimator.fit(features[key], target[key], sample_key=key,
                          grid_shape=grid_shapes[key],
                          sample_weights=gfuncs[key],
                          gram_setup=gram_setups.get(key))
            # Copy: ``fit`` allocates a fresh ``full_coef_`` per call, but
            # taking a copy keeps that an implementation detail rather than a
            # correctness dependency.
            sampled_full_coefs.append(np.array(estimator.full_coef_, dtype=float))
            sampled_sw_weights[key] = estimator.cached_weights_
            sampled_vc_scores[key] = estimator.cached_vc_score_

        # Averaging the FULL vector is arithmetically identical to the former
        # "mean the coefs, mean the intercepts separately" (the mean is linear),
        # but it keeps the intercept in its slot instead of re-appending it.
        weights = np.mean(np.stack(sampled_full_coefs, axis = 1), axis = 1)

        objective.weights_internal = weights
        objective.weights_internal_evald = True
        # VWSR runs no un-normalised refit -- PhysicsInformedLasso already fits
        # on the physical scale -- so the final magnitudes ARE the internal
        # ones. The former ``np.append([w for w in ... if w != 0], intercept)``
        # zero-filtering is gone: weights_final retains its zeros and stays
        # aligned to the full structure (Equation._validate_weight_layout).
        objective.weights_final = weights.copy()
        objective.weights_final_evald = True
        # Per-sample stability products are kept PER TRAJECTORY (dicts keyed
        # by sample ID, the tree-wide multisample convention) and reduced by
        # ``Instability.compute``. Assigning ``estimator.cached_*`` after the
        # loop kept only the LAST trajectory's values, so the instability
        # objective scored the equation on one sample out of n. They cannot be
        # averaged here: each sample converges to its OWN active mask, so the
        # vectors legitimately differ in length. ``None`` entries (axis path,
        # or every feature died) survive as ``None`` for the consumer to skip.
        objective._cached_sw_weights = (None if all(v is None for v in sampled_sw_weights.values())
                                        else sampled_sw_weights)
        objective._cached_vc_score = (None if all(v is None for v in sampled_vc_scores.values())
                                      else sampled_vc_scores)
        # See LASSOSparsity.apply: _eval_cache survives a weights update;
        # only structural resets via ``Equation.reset_state`` should wipe it.

    def use_default_tags(self):
        self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}


def build_sparsity_operator(sparsity_cls=None, sparsity_kwargs=None):
    """Instantiate the sparse-regression operator and apply its settings.

    ``sparsity_kwargs`` comes from the ``objectives`` config group, already
    validated against this class by
    ``epde.interface.search_config.validate_sparsity_kwargs`` -- so every key
    names an attribute the operator really has.
    """
    operator = (sparsity_cls if sparsity_cls is not None else VWSRSparsity)()
    for name, value in (sparsity_kwargs or {}).items():
        setattr(operator, name, value)
    return operator


def initial_sparsity_interval(sparsity_cls) -> tuple:
    """The interval a new population seeds ``('sparsity', var)`` from.

    The value is the sparse-regression operator's own parameter -- only
    :class:`LASSOSparsity` reads it -- so the operator class in use is asked
    for it rather than the search-space configuration carrying a setting that
    the default (VWSR) pipeline ignores. A class that declares nothing gets the
    neutral, degenerate ``(1.0, 1.0)``; ends that are equal are the convention
    for "this operator does not tune a sparsity constant".
    """
    interval = getattr(sparsity_cls, 'initial_sparsity_interval', None)
    if interval is None:
        return (1.0, 1.0)
    return tuple(interval)
