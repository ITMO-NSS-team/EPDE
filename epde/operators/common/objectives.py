#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Objective "fillers" for the fitness host operators.

A *filler* is a single-responsibility metric component (NOT a
``CompoundOperator`` -- it carries no EA params / suboperators). A fitness
host (``SolverFreeFitness`` / ``SolverBasedFitness`` in
``epde.operators.common.fitness``) runs the shared scaffolding once
(sparsity -> coefficient fit -> context) and asks each filler for one
scalar, which the filler writes to its own equation attribute
(``fitness_value`` / ``coefficients_stability`` / ...).

Two families:

* **Solver-free** (:class:`EquationObjective`): compute from the fitted
  feature matrix / target on the data grid. The *discrepancy* fillers
  double as the host's "primary" objective -- they own the
  right-part-selection hooks (``needs_sparsity`` / ``is_degenerate`` /
  normalization) and their value is what ``EqRightPartSelector`` ranks
  candidate targets by during its ``force_out_of_place`` term-sweep.
* **Solver-based**: the solver options of the same :class:`Discrepancy`
  family (``'solver_l2'`` / ``'pic'`` / ``'deepxde'``), computed from a
  solved field produced by a PDE solver backend via the
  ``compute(eq, eq_idx, sctx)`` protocol.

Every metric's logic lives in exactly ONE filler -- this is the
single-responsibility split that replaces the copy-pasted discrepancy /
stability blocks formerly duplicated across five fitness operators.
"""
from __future__ import annotations

from abc import ABC

from dataclasses import dataclass
from functools import partial, reduce

from typing import Union, Dict

import numpy as np

import epde.globals as global_var
from epde.operators.common.stability import calculate_weights, vc_stability_total_lr
from epde.operators.utils.template import CompoundOperator, dictApplyUFunc, dictZerosLike, dictFullLike, \
    dictAdd, dictSubtr
from epde.operators.common.survival import (
    chi2_scores, heterogeneity_scores, survival_scores, tile_scores,
)

LOSS_NAN_VAL = 1e7


def _extract_coefs_intercept(equation):
    """Reconstruct ``(coefs, intercept)`` from ``weights_final`` under the
    VWSR convention: ``weights_internal[-1]`` is the intercept-presence flag.
    A zero intercept was dropped from ``weights_final`` (it is then
    nnz-length); a non-zero one is its trailing entry. ``weights_final`` is
    sparsity-truncated -- NOT position-indexed like ``weights_internal``.
    Pinned by TestSparsityWeightsFinalConventions."""
    if equation.weights_internal[-1]:
        return equation.weights_final[:-1], equation.weights_final[-1]
    return equation.weights_final, 0.0


def _degenerate_excluding_intercept(filler, equation) -> bool:
    """Shared ``is_degenerate`` for the VWSR-paired discrepancy fillers:
    degenerate iff every non-intercept weight is zero, or the post-fit
    discrepancy exceeds the filler's ``degenerate_threshold``."""
    fv = getattr(equation, 'fitness_value', None)
    return bool(np.all(equation.weights_internal[:-1] == 0)
                or (fv is not None and fv > filler.degenerate_threshold))


# --------------------------------------------------------------------------- #
#  Solver-free fillers                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class FitContext:
    """Shared products of one solver-free fitness ``apply`` pass."""
    g_fun_vals : Dict[int, np.ndarray] # or even Union[np.ndarray, Dict[int, np.ndarray]] - weighting vector (flat) or None
    data_shape : Dict[int, np.ndarray] # object        # grid inner shape (tuple) or None
    penalty_coeff : float
    for_rps : bool = False     # True during EqRPS force_out_of_place sweep


class EquationObjective:
    """Base class for a solver-free objective filler.

    Subclasses set ``name`` / ``value_attr`` / ``flag_attr`` and implement
    :meth:`compute`. Discrepancy subclasses also override the RPS
    scaffolding hooks below; non-primary fillers (e.g. instability) never
    have those hooks consulted.
    """
    name = 'objective'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'

    # Whether the host's failure stamps (the unfitted-equation stamp and the
    # post-compute degeneracy verdict, see ``SolverFreeFitness.apply``)
    # overwrite this filler's value with LOSS_NAN_VAL. ``Complexity`` opts
    # out of BOTH: its value is structure-derived, and before it was a
    # filler the lazy Pareto reader reported the real count for degenerate
    # and RPS-exhausted forms alike -- stamping would change the legacy
    # Pareto geometry (two stamped forms would tie on the complexity axis
    # instead of dominance-comparing on their real counts). A skipped filler
    # keeps its flag down, so the ``equation_complexity`` reader falls back
    # to the lazy cores -- the exact pre-filler semantics.
    stamped_on_failure = True

    # -- right-part-selection scaffolding hooks (consulted on the primary) -- #
    def needs_sparsity(self, equation, for_rps: bool) -> bool:
        return bool(for_rps or not getattr(equation, 'weights_internal_evald', False))

    # A candidate target whose post-fit discrepancy exceeds this is declined
    # during the RPS term-sweep (a degenerate target fits too poorly to be the
    # right part). Checked AFTER compute() has set ``fitness_value`` -- see
    # ``SolverFreeFitness.apply``; reading it before compute() gives the stale /
    # ``None`` default and the check silently never fires.
    degenerate_threshold = 1.0

    def is_degenerate(self, equation) -> bool:
        fv = getattr(equation, 'fitness_value', None)
        return bool(np.all(equation.weights_internal == 0)
                    or (fv is not None and fv > self.degenerate_threshold))

    def compute(self, equation, ctx: FitContext) -> float:
        raise NotImplementedError


class Discrepancy(EquationObjective):
    """The DISCREPANCY objective family: one filler containing every residual
    metric as an option, BUILT FROM THE SEARCH CONFIGURATION -- ``Discrepancy()``
    constructs bare and resolves its option at compute time from
    ``epde.globals.resolve_discrepancy_metric()``, which ``EpdeSearch``
    writes from its ``discrepancy_metric`` kwarg (the same late-dispatch
    convention as :class:`Instability` and :class:`Complexity`). An explicit
    constructor metric exists only as internal wiring for fixed-role hosts.

    Options (canonical):

    * ``'wape'`` (default) -- normalised absolute residual, the legacy
      ``L2LRFitness`` core: ``sum|target - fit| / sum|target|``. Normalised
      features in-place, un-normalised during the RPS sweep, the
      ``weights_internal[-1]`` intercept-presence test, no ``g_func``
      weighting and no penalty division.
    * ``'l2'`` -- weighted L2 norm of the residual, the legacy ``L2Fitness``
      core: un-normalised features, the three-way ``weights_internal`` /
      ``weights_final`` reconstruction, ``g_func`` weighting, and the
      all-zero penalty division. NO target-scale normalisation.
    * ``'l2_relative'`` -- ``||target - fit||_2 / ||target||_2``: the L2
      analogue of WAPE (which is the L1 form of the same relative residual).
      Like WAPE, no ``g_func`` weighting.
    * ``'scale_invariant'`` -- the pointwise cancellation residual
      ``mean_x |sum_k c_k phi_k(x)| / sum_k |c_k phi_k(x)|`` over every term
      (target + fitted features + intercept). Invariant under multiplying
      the whole equation by any field ``g(x)`` and under target choice;
      bounded in [0, 1] by the triangle inequality, so it does NOT reward
      rewriting ``E = 0`` as the degenerate ``T*E = 0``.

    SOLVER-BASED options (a solver error IS a discrepancy -- it measures
    the solved field against the data instead of the fitted feature matrix
    against the target; bodies moved verbatim from the former
    ``SolverL2Discrepancy`` / ``PICError`` / ``DeepXDEError`` classes):

    * ``'solver_l2'`` -- L2 of (solved field - data), ``g_func``-weighted,
      plus the PINN residual loss, penalty division on all-zero weights
      (the legacy ``SolverBasedFitness`` core; named ``solver_l2`` because
      plain ``'l2'`` is the solver-FREE legacy metric above).
    * ``'pic'`` -- PIC p-loss: mean squared (solved field - data) weighted
      by ``g_func`` plus the PINN residual loss, no penalty division.
    * ``'deepxde'`` -- error of (DeepXDE solution - data) under a
      configurable inner metric (``error_metric``: 'rmse' default / 'l2' /
      'mae'); the host packs per-eq masked ``(solution, data)`` pairs and
      re-syncs ``error_metric`` / ``penalty_coeff`` from its params before
      each solve (``SolverBasedFitness._apply_deepxde``).

    PROTOCOL: solver-free options are computed as ``compute(equation,
    ctx: FitContext)`` by ``SolverFreeFitness``; solver options as
    ``compute(eq, eq_idx, sctx: SolverContext)`` by ``SolverBasedFitness``
    (``'solver_l2'`` / ``'pic'`` read the full-field ``solution[..., idx]``
    + the ``tensor_cache`` reference; ``'deepxde'`` reads the per-eq masked
    pair). ``compute`` dispatches on the call shape and fails loudly when
    an option is driven through the wrong host.

    Aliases: ``'l2_scaled'`` / ``'l2_rel'`` / ``'residual'`` ->
    ``'l2_relative'``; ``'scale_inv'`` / ``'sinv'`` / ``'cancellation'`` ->
    ``'scale_invariant'``. Unknown names raise ``ValueError`` -- this
    replaces the silent typo-to-WAPE catch-all the strategy switch had.

    Every solver-free option owns the full primary-filler contract
    (right-part-selection scaffolding): ``needs_sparsity`` is the shared
    base rule; ``is_degenerate`` branches -- ``'l2'`` keeps its legacy
    all-zero-``weights_internal`` test (no threshold), the other three use
    the intercept-excluding test plus the post-fit ``degenerate_threshold``.
    The RPS hooks are never consulted on solver options (the solver host
    wires a lightweight solver-free fitness for the RPS term-sweep).
    """
    name = 'discrepancy'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'
    # norm_prtl = partial(np.linalg.norm, ord=2)

    OPTIONS = ('wape', 'l2', 'l2_relative', 'scale_invariant',
               'solver_l2', 'pic', 'deepxde')
    SOLVER_OPTIONS = ('solver_l2', 'pic', 'deepxde')
    ALIASES = {'l2_scaled': 'l2_relative', 'l2_rel': 'l2_relative',
               'residual': 'l2_relative', 'scale_inv': 'scale_invariant',
               'sinv': 'scale_invariant', 'cancellation': 'scale_invariant'}

    def __init__(self, metric: str = None, error_metric: str = 'rmse',
                 penalty_coeff: float = 0.2):
        """``Discrepancy()`` -- the normal construction -- carries NO metric:
        the option is resolved at compute time from
        ``epde.globals.resolve_discrepancy_metric()``, which ``EpdeSearch``
        writes from its own configuration. An explicit ``metric`` is
        INTERNAL wiring only, for hosts whose metric is fixed by their role
        (the RPS-sweep's ``'l2_relative'`` lightweight fitness; the solver
        branch's backend-implied ``'solver_l2'`` / ``'deepxde'``)."""
        if metric is not None:
            metric = self.ALIASES.get(metric, metric)
            if metric not in self.OPTIONS:
                raise ValueError(
                    f'discrepancy metric must be one of {self.OPTIONS} '
                    f'(or aliases {tuple(self.ALIASES)}); got {metric!r}')
        self.metric = metric
        # 'deepxde'-option inner state (the host re-syncs both from its
        # params before each solve); harmless carried on other options.
        self.error_metric = error_metric
        self.penalty_coeff = penalty_coeff

    def _resolved_metric(self) -> str:
        """Instance override if wired, else the search-level configuration."""
        return (self.metric if self.metric is not None
                else global_var.resolve_discrepancy_metric())

    # needs_sparsity: the shared base ``for_rps or not weights_internal_evald``
    # rule (the legacy L2Discrepancy override was identical to the base). In
    # MOEA/D the in-place pass always has weights_internal_evald=True (RPS set
    # it); in single-objective mode the ``not weights_internal_evald`` clause
    # lets the in-place fitness trigger sparsity itself -- and also repairs
    # the latent L2LRFitness crash when an RPS-exhausted equation arrived
    # unfitted.

    def is_degenerate(self, equation) -> bool:
        if self._resolved_metric() == 'l2':
            # Legacy L2Fitness contract: all-zero weights only, no threshold.
            return bool(np.all(equation.weights_internal == 0))
        return _degenerate_excluding_intercept(self, equation)

    def compute(self, equation, *args) -> float:
        """Protocol-dispatching compute.

        Solver-free options: ``compute(equation, ctx: FitContext)`` (the
        ``SolverFreeFitness`` filler-loop shape). Solver options:
        ``compute(eq, eq_idx: int, sctx: SolverContext)`` (the
        ``SolverBasedFitness`` per-equation shape). A mismatch means the
        option was wired to the wrong host -- fail loudly.
        """
        metric = self._resolved_metric()
        if metric in self.SOLVER_OPTIONS:
            if len(args) != 2:
                raise TypeError(
                    f"solver discrepancy option {metric!r} is computed "
                    "as compute(eq, eq_idx, sctx) by SolverBasedFitness; it "
                    "cannot serve a solver-free host")
            eq_idx, sctx = args
            return getattr(self, f'_compute_{metric}')(equation, eq_idx, sctx)
        if len(args) != 1:
            raise TypeError(
                f"solver-free discrepancy option {metric!r} is computed "
                "as compute(equation, ctx) by SolverFreeFitness; it cannot "
                "serve a solver host")
        return getattr(self, f'_compute_{metric}')(equation, args[0])

    def _compute_wape(self, equation, ctx: FitContext) -> float:
        # L2LRFitness used un-normalised features only on the RPS sweep
        # (force_out_of_place), normalised features for the in-place pass.
        normalize = not ctx.for_rps
        _, targets, features = equation.evaluate(normalize = normalize, return_val = False)
        if features is None:
            discr = dictSubtr(targets, {key: tg.mean() for key, tg in targets.items()})
        else:
            coefs, intercept = _extract_coefs_intercept(equation)
            discr = dictSubtr(targets, dictAdd(dictApplyUFunc(np.dot, features, 
															  equation.weights_final[:-1]),
											   intercept))
			# discr = target - (np.dot(features, coefs) + intercept)
        rl_error = np.sum([np.sum(np.abs(discr[key])) / np.sum(np.abs(targets[key]))
						   for key in targets.keys()])
        return float(rl_error)

    def _compute_l2(self, equation, ctx: FitContext) -> float:
        _, targets, features = equation.evaluate(normalize=False, return_val=False)

       #  _, targets, features = equation.evaluate(normalize = False, return_val = False)
        if all([feature is None for feature in features.values()]):
            discr_feats = 0
        else:
            n_cols = [feature.shape[1] if feature.ndim > 1 else 1
					  for feature in features.values()][0]
            mask = equation.weights_internal != 0
            if n_cols == len(mask):
                discr_feats = dictApplyUFunc(np.dot, features, equation.weights_final[:-1])
            elif n_cols == int(mask.sum()):
                discr_feats = dictApplyUFunc(np.dot, features, equation.weights_final[:-1])
            else:
                discr_feats = dictZerosLike(features, 0)

        discr = dictSubtr(dictAdd(discr_feats, dictFullLike(targets.shape, equation.weights_final[-1])), targets)
        
        g = ctx.g_fun_vals

        if g is not None:
            discr = dictApplyUFunc(np.multiply, discr, self.g_fun_vals)
        
        rl_error = dictApplyUFunc(self.norm_prtl, discr)
        rl_error = reduce(lambda x, y: x+y, rl_error.values(), 0) # [value for value in rl_error.values()]

        if np.sum(equation.weights_final[:-1]) == 0:
            rl_error /= ctx.penalty_coeff

        return float(rl_error)

    def _compute_l2_relative(self, equation, ctx: FitContext) -> float:
        normalize = not ctx.for_rps
        _, targets, features = equation.evaluate(normalize=normalize, return_val=False)
        # target = np.asarray(target, dtype=float)
        if features is None:
            discr = dictSubtr(targets, {key: target.mean() for key, target in targets.items()})
        else:
            coeffs, intercept = _extract_coefs_intercept(equation)
            discr = dictSubtr(targets, dictAdd(dictApplyUFunc(np.dot, features, coeffs), intercept))
        den = float(np.mean([np.linalg.norm(target) for target in targets]))
        if np.isclose(den, 0.0):
            return float(LOSS_NAN_VAL)
        return float(np.mean([np.linalg.norm(discrepancy) for discrepancy in discr.values()]) / den)

    def _compute_scale_invariant(self, equation, ctx: FitContext) -> float:
        normalize = not ctx.for_rps
        _, targets, features = equation.evaluate(normalize=normalize, return_val=False)
        if features is None:
            # only the target term survives -> a single term cannot cancel.
            return 1.0
        coeffs, intercept = _extract_coefs_intercept(equation)
        contribs = dictApplyUFunc(np.multiply, features, np.asarray(coeffs)[None, :])          # per-term weighted values
        resid = dictSubtr(targets, dictAdd(contribs.sum(axis=1), intercept))        # |sum_k c_k phi_k|
        term_mass = dictAdd(dictApplyUFunc(np.abs, targets),
                            dictAdd(dictApplyUFunc(lambda x: np.abs(x).sum(axis=1), contribs), abs(intercept)))
        rho = dictApplyUFunc(np.divide, dictApplyUFunc(np.abs, resid), term_mass)
        return float(np.mean([value for value in dictApplyUFunc(np.mean, rho).values()]))

    # -- solver-based options (SolverBasedFitness per-equation protocol) -- #

    def _compute_solver_l2(self, eq, eq_idx, sctx):
        if _loss_is_nan(sctx.loss_add):
            return 2 * LOSS_NAN_VAL
        # ref = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))
        sol = {key: sol[..., eq_idx] for key, sol in sctx.solution.items()}
        ref = global_var.samples_manager.get((eq.main_var_to_explain, (1.0,)))
        ref = {key: reference.reshape(sol[key].shape) for key, reference in ref.items()}

        discr = dictApplyUFunc(np.multiply, dictSubtr(sol, ref), 
                               {gfunc_vals.reshape(discr[key].shape) for key, gfunc_vals in sctx.g_fun_vals.values()})
        # discr = np.multiply(discr, sctx.g_fun_vals.reshape(discr.shape))
        # rl_error = np.mean([item for item in dictApplyUFunc(lambda x: np.linalg.norm(x, ord=2), discr).values()])
        rl_error = np.mean(list(dictApplyUFunc(lambda x: np.linalg.norm(x, ord=2), discr).values()))
        
        fitness = rl_error + sctx.pinn_loss_mult * float(sctx.loss_add)
        if np.sum(eq.weights_final[:-1]) == 0:
            fitness /= sctx.penalty_coeff
        return float(fitness)

    def _compute_pic(self, eq, eq_idx, sctx):
        if _loss_is_nan(sctx.loss_add):
            return 2 * LOSS_NAN_VAL
        # ref = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))
        # sol = sctx.solution[..., eq_idx]
        # discr = sol - ref.reshape(sol.shape)
        # discr = np.multiply(discr, sctx.g_fun_vals.reshape(discr.shape))
        sol = {key: sol[..., eq_idx] for key, sol in sctx.solution.items()}
        ref = global_var.samples_manager.get((eq.main_var_to_explain, (1.0,)))
        ref = {key: reference.reshape(sol[key].shape) for key, reference in ref.items()}

        discr = dictApplyUFunc(np.multiply, dictSubtr(sol, ref), 
                               {gfunc_vals.reshape(discr[key].shape) for key, gfunc_vals in sctx.g_fun_vals.values()})

        rl_error = np.mean(list(dictApplyUFunc(lambda x: np.mean(x ** 2), discr).values()))
        return float(rl_error + sctx.pinn_loss_mult * float(sctx.loss_add))

    def _compute_deepxde(self, eq, eq_idx, sctx):
        # sctx.solution[eq_idx] = masked solution, sctx.g_fun_vals[eq_idx] =
        # masked data (packed by SolverBasedFitness's deepxde branch).
        masked_solution = {key: sol[eq_idx] for key, sol in sctx.solution.items()} # [eq_idx]
        masked_data = {key: gfunc_val[eq_idx] for key, gfunc_val in sctx.g_fun_vals.items()} 
        metric = self.error_metric
        # if metric == 'l2':
        #     err = np.linalg.norm(masked_solution - masked_data, ord=2)
        # elif metric == 'mae':
        #     err = np.mean(np.abs(masked_solution - masked_data))
        # else:  # 'rmse' default
        #     err = np.sqrt(np.mean((masked_solution - masked_data) ** 2))

        if metric == 'l2':
            err_func = lambda x: np.linalg.norm(x - masked_data, ord=2)
        elif metric == 'mae':
            err_func = lambda x: np.mean(np.abs(x))
        else:  # 'rmse' default
            err_func = lambda x: np.sqrt(np.mean(x**2))
            # err = np.sqrt(np.mean((masked_solution - masked_data) ** 2))
        err = np.mean(list(dictApplyUFunc(err_func, dictSubtr(masked_solution, masked_data)).values()))

        if np.sum(eq.weights_final) == 0:
            err /= self.penalty_coeff
        return float(err)

class WAPEDiscrepancy(EquationObjective):
    """Normalised absolute residual (WAPE), the ``L2LRFitness`` core.

    ``sum|target - fit| / sum|target|``. Reproduces ``L2LRFitness.apply``:
    normalised features in-place, un-normalised during the RPS sweep, the
    ``weights_internal[-1]`` intercept-presence test, no ``g_func``
    weighting and no penalty division.
    """
    name = 'discrepancy'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'

    # needs_sparsity: inherits the base ``for_rps or not weights_internal_evald``.
    # In MOEA/D the in-place pass always has weights_internal_evald=True (RPS
    # set it), so this is equivalent to the legacy L2LRFitness "sparsity only
    # on the RPS sweep" behaviour. In single-objective mode (RandomRHPSelector
    # never runs sparsity) the ``not weights_internal_evald`` clause lets the
    # in-place fitness trigger sparsity itself -- and also repairs the latent
    # L2LRFitness crash when an RPS-exhausted equation arrived unfitted.

    def is_degenerate(self, equation) -> bool:
        return bool(np.all(equation.weights_internal[:-1] == 0))

    def compute(self, equation, ctx: FitContext) -> float:
        # L2LRFitness used un-normalised features only on the RPS sweep
        # (force_out_of_place), normalised features for the in-place pass.
        # normalize = not ctx.for_rps
        _, targets, features = equation.evaluate(normalize = False, return_val = False)

        if features is None:
            discr = dictApplyUFunc(lambda x, y: x - y, targets, dictApplyUFunc(np.mean(), targets)) # target - target.mean()
        else:
            if equation.weights_internal[-1]:
                discr_feats = dictApplyUFunc(np.dot, features, equation.weights_final[:-1])
                discr_feats = dictApplyUFunc(lambda x, y: x+y, discr_feats, equation.weights_final[-1])
            else:
                discr_feats = dictApplyUFunc(np.dot, features, equation.weights_final[:-1])
            discr = dictApplyUFunc(lambda x, y: x-y, targets, discr_feats)

        rl_error = reduce(lambda x, y: x + float(np.sum(np.abs(discr[y])) / np.sum(np.abs(targets[y]))), range(len(discr)), 0.) 
        return rl_error


class Instability(EquationObjective):
    """The instability objective, dispatching on the estimator selected by
    ``epde.globals.resolve_instability_metric()``:

    * ``'vcoef'``: fast path sums the per-term ``_cached_vc_score``
      produced by ``PhysicsInformedLasso`` (``VWSRSparsity``); fallback
      (LASSO path, no cache) is ``vc_stability_total_lr``.
    * ``'cv'``: axis-aligned sliding-window CV via ``calculate_weights``.
    * ``'survival'`` / ``'tile'`` / ``'het'`` / ``'chi2'``: the estimators
      from ``epde.operators.common.survival`` (``'het'`` = Q-calibrated
      excess-variance heterogeneity, tau^2/(tau^2+mean^2); ``'chi2'`` =
      per-term Nyblom-Hansen cumulative-score-path constancy: global-OLS
      Theta, one path per grid axis, each bulge measured against the
      term's own signal energy -- the resolver DEFAULT), memoized per
      equation as ``_cached_alt_instability = (metric, value)``.

    The estimator choice affects ONLY this objective; the sparsity
    keep-rule keeps following ``gram_mode``. Fails loudly: any exception
    propagates -- a silent fallback value would corrupt the Pareto front
    invisibly.
    """
    name = 'instability'
    value_attr = 'coefficients_stability'
    flag_attr = 'stability_calculated'

    def compute(self, equation, ctx: FitContext) -> float:
        metric = global_var.resolve_instability_metric()
        if metric == 'vcoef':
            cached = getattr(equation, '_cached_vc_score', None)
            if cached is not None:
                return float(np.sum(cached))
        elif metric in ('survival', 'tile', 'het', 'chi2'):
            cached = getattr(equation, '_cached_alt_instability', None)
            if cached is not None and cached[0] == metric:
                return float(cached[1])
        data_shape = ctx.data_shape
        _, targets, features = equation.evaluate(normalize = True, return_val = False)

        if features is None:
            return 1.0
        fit_intercept = bool(equation.weights_internal[-1] != 0)
        if metric == 'vcoef':
            return float(vc_stability_total_lr(
                features, targets, ctx.g_fun_vals, data_shape,
                main_var=equation.main_var_to_explain,
                fit_intercept=fit_intercept))
        if metric in ('survival', 'tile', 'het', 'chi2'):
            estimator = {'survival': survival_scores, 'tile': tile_scores,
                         'het': heterogeneity_scores, 'chi2': chi2_scores}[metric]
            estim = partial(estimator, fit_intercept=fit_intercept)
            scores = dictApplyUFunc(estim, features, targets, ctx.g_fun_vals, data_shape) #estimator(features, targets, ctx.g_fun_vals, data_shape,
                     #          fit_intercept=fit_intercept)
            value = np.mean([float(np.sum(score)) for score in scores.values()])
            equation._cached_alt_instability = (metric, value)
            return value
        # metric == 'cv': the axis-aligned sliding-window CV.
        sw = getattr(equation, '_cached_sw_weights', None)
        if sw is None:
            sw = calculate_weights(
                features, targets, ctx.g_fun_vals, data_shape, fit_intercept,
                gram_cls=None, gram_kwargs=None)
        sw_arr = np.array(sw)
        mu = sw_arr.mean(axis=0)
        std = sw_arr.std(axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = (std ** 2) / (mu ** 2)
            cv[mu == 0] = 0.0
        return float(np.sum(np.nan_to_num(cv)) / len(data_shape))


class Complexity(EquationObjective):
    """The COMPLEXITY objective family: parsimony of the fitted structure,
    with the option selected per instance (or, when constructed without one,
    from the process-level ``epde.globals.complexity_metric``).

    Options:

    * ``'factors'`` (the unset default) -- the legacy factor-count: 0.5 per
      non-derivative factor, derivative order per derivative factor, summed
      over the target term and every non-zero-weight term. Bit-compatible
      with every existing legacy-pipeline artifact (known quirks included --
      see ``complexity_deriv``).
    * ``'terms'`` -- active-term count: non-zero non-target
      ``weights_internal`` slots + 1 when the fitted intercept is non-zero,
      UNIFORM across the LASSO / VWSR sparsity pairings (see
      ``_terms_of_equation``).

    A non-primary filler like :class:`Instability` (the RPS scaffolding
    hooks are never consulted on it). ``stamped_on_failure = False``: the
    value is deterministic from structure + weights and carries no fit
    information, so neither host failure stamp touches it -- matching the
    pre-filler behavior where the lazy Pareto reader reported the true
    count for degenerate and RPS-exhausted forms alike.
    """
    name = 'complexity'
    value_attr = 'complexity_value'
    flag_attr = 'complexity_calculated'
    stamped_on_failure = False

    OPTIONS = ('factors', 'terms')

    def __init__(self, metric: str = None):
        if metric is not None and metric not in self.OPTIONS:
            raise ValueError(
                f'complexity metric must be one of {self.OPTIONS} or None '
                f'(= follow globals.complexity_metric); got {metric!r}')
        self.metric = metric

    def compute(self, equation, ctx: FitContext) -> float:
        # Local import: the per-equation cores live in eq_mo_objectives (the
        # readers' module), which main_structures also imports locally -- the
        # same cycle-dodging idiom.
        from epde.eq_mo_objectives import (_complexity_of_equation,
                                           _terms_of_equation)
        metric = (self.metric if self.metric is not None
                  else global_var.resolve_complexity_metric())
        if metric == 'terms':
            return float(_terms_of_equation(equation))
        return float(_complexity_of_equation(equation))

#            # print(dictApplyUFunc(vc_stab_prtl, features, targets, ctx.g_fun_vals, data_shape), len(features),
#            #       reduce(lambda x, y: x + float(y), dictApplyUFunc(vc_stab_prtl, features, targets, ctx.g_fun_vals, data_shape).values()) / len(features))
#            # raise RuntimeError('VCOEF CASE!')
#            print(f'data_shapes are {data_shape}')
#            return reduce(lambda x, y: x + float(y), 
#                          dictApplyUFunc(vc_stab_prtl, features, targets, ctx.g_fun_vals, data_shape).values()) / len(features)

#        sw = getattr(equation, '_cached_sw_weights', None)
#        if sw is None:
#            calc_weights_prtl = partial(calculate_weights, fit_intercept = fit_intercept, gram_cls = None, gram_kwargs = None)
#            sw = dictApplyUFunc(calculate_weights, features, targets, ctx.g_fun_vals, data_shape) # , fit_intercept,
#                # gram_cls=None, gram_kwargs=None)

#        # sw_arr = np.array(sw)
#        mu = dictApplyUFunc(partial(np.mean, axis = 0), sw) # sw_arr.mean(axis=0)
#        std = dictApplyUFunc(partial(np.std, axis = 0, ddof=1), sw) # sw_arr.std(axis=0, ddof=1)
#        with np.errstate(divide='ignore', invalid='ignore'):
#            cv = dictApplyUFunc(lambda x, y: x / y, 
#                                dictApplyUFunc(lambda x: x**2, std),
#                                dictApplyUFunc(lambda x: x**2, std))  # (std ** 2) / (mu ** 2)
#            cv = dictApplyUFunc(lambda x, y: np.where(x, 0., y), dictApplyUFunc(partial(np.isclose, b = 0.), mu), cv) # [mu == 0] = 0.0

        # print(f'cv in Instability: {cv}')
        # raise RuntimeError('...')

#        return float(np.sum(dictApplyUFunc(np.nan_to_num, cv)) / len(data_shape))
        # except: # Exception
        #     return 1.0


# --------------------------------------------------------------------------- #
#  Solver-based fillers                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class SolverContext:
    """Shared products of one ``SolverBasedFitness`` solve."""
    solution: object          # ndarray (..., n_eqs) NN solution on the grid
    loss_add: object          # torch scalar / float -- PINN residual loss
    g_fun_vals: object        # weighting vector (backend-specific masking)
    penalty_coeff: float
    pinn_loss_mult: float


def _loss_is_nan(loss_add) -> bool:
    try:
        import torch
        if torch.is_tensor(loss_add):
            return bool(torch.isnan(loss_add))
    except Exception:
        pass
    try:
        return bool(np.isnan(float(loss_add)))
    except Exception:
        return False


# The solver-based discrepancy options live INSIDE the ``Discrepancy``
# family above (``'solver_l2'`` / ``'pic'`` / ``'deepxde'``); this section
# keeps only the solver context/plumbing shared with the hosts. The former
# ``SolverObjective`` base and its three subclasses are gone -- a solver
# error is a discrepancy.
