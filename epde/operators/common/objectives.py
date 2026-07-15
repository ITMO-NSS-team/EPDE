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
* **Solver-based** (:class:`SolverObjective`): compute from a solved field
  produced by a PDE solver backend.

Every metric's logic lives in exactly ONE filler -- this is the
single-responsibility split that replaces the copy-pasted discrepancy /
stability blocks formerly duplicated across five fitness operators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import epde.globals as global_var
from epde.operators.common.stability import calculate_weights, vc_stability_total_lr

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
    g_fun_vals: object        # weighting vector (flat) or None
    data_shape: object        # grid inner shape (tuple) or None
    penalty_coeff: float
    for_rps: bool = False     # True during EqRPS force_out_of_place sweep


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


class L2Discrepancy(EquationObjective):
    """Weighted L2 norm of the residual (the legacy ``L2Fitness`` core).

    Reproduces ``L2Fitness.apply`` verbatim: un-normalised features, the
    three-way ``weights_internal`` / ``weights_final`` reconstruction, the
    ``g_func`` weighting, and the all-zero penalty division.
    """
    name = 'discrepancy'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'

    def needs_sparsity(self, equation, for_rps: bool) -> bool:
        return bool(for_rps or not getattr(equation, 'weights_internal_evald', False))

    def is_degenerate(self, equation) -> bool:
        return bool(np.all(equation.weights_internal == 0))

    def compute(self, equation, ctx: FitContext) -> float:
        _, target, features = equation.evaluate(normalize=False, return_val=False)
        if features is None:
            discr_feats = 0
        else:
            n_cols = features.shape[1] if features.ndim > 1 else 1
            mask = equation.weights_internal != 0
            if n_cols == len(mask):
                discr_feats = np.dot(features, equation.weights_internal)
            elif n_cols == int(mask.sum()):
                discr_feats = np.dot(features, equation.weights_final[:-1])
            else:
                discr_feats = np.zeros(features.shape[0])

        # ``weights_final[-1]`` is read as the intercept UNCONDITIONALLY --
        # coherent only under the LASSOSparsity pairing, whose weights_final
        # always carries a trailing intercept slot (nnz+1, even when 0.0).
        # VWSRSparsity drops a zero intercept (weights_final is then
        # nnz-length); its fillers (WAPE/ScaleInvariant/L2Relative) branch on
        # the ``weights_internal[-1]`` presence flag instead. Pinned by
        # TestSparsityWeightsFinalConventions.
        discr = (discr_feats + np.full(target.shape, equation.weights_final[-1]) - target)
        g = ctx.g_fun_vals
        if g is not None and getattr(g, 'shape', None) == discr.shape:
            discr = np.multiply(discr, g)
        rl_error = np.linalg.norm(discr, ord=2)

        fitness_value = rl_error
        if np.sum(equation.weights_final) == 0:
            fitness_value /= ctx.penalty_coeff
        return float(fitness_value)


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

    is_degenerate = _degenerate_excluding_intercept

    def compute(self, equation, ctx: FitContext) -> float:
        # L2LRFitness used un-normalised features only on the RPS sweep
        # (force_out_of_place), normalised features for the in-place pass.
        normalize = not ctx.for_rps
        _, target, features = equation.evaluate(normalize=normalize, return_val=False)
        if features is None:
            discr = target - target.mean()
        else:
            coefs, intercept = _extract_coefs_intercept(equation)
            discr = target - (np.dot(features, coefs) + intercept)
        rl_error = np.sum(np.abs(discr)) / np.sum(np.abs(target))
        return float(rl_error)


class ScaleInvariantDiscrepancy(EquationObjective):
    """Scale-invariant discrepancy: the pointwise cancellation residual

        mean_x  |sum_k c_k phi_k(x)|  /  sum_k |c_k phi_k(x)|

    summed over every term of the equation (target + fitted features + intercept).
    Unlike WAPE (``sum|target-fit| / sum|target|``) this is invariant under
    multiplying the whole equation by any field ``g(x)`` -- numerator and
    denominator both pick up ``|g(x)|`` pointwise and it cancels -- and under
    target choice (the term *set* is the same whichever term is the right part).
    Bounded in [0, 1] by the triangle inequality, so it does NOT reward rewriting
    ``E = 0`` as the degenerate ``T*E = 0``.

    Reconstructs ``(coefs, intercept)`` from ``weights_final`` exactly as
    ``WAPEDiscrepancy`` (same ``weights_internal[-1]`` intercept test and the same
    ``normalize = not for_rps`` term set), so it is a drop-in primary filler.
    """
    name = 'discrepancy'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'

    is_degenerate = _degenerate_excluding_intercept

    def compute(self, equation, ctx: FitContext) -> float:
        normalize = not ctx.for_rps
        _, target, features = equation.evaluate(normalize=normalize, return_val=False)
        if features is None:
            # only the target term survives -> a single term cannot cancel.
            return 1.0
        coefs, intercept = _extract_coefs_intercept(equation)
        contribs = features * np.asarray(coefs)[None, :]          # per-term weighted values
        resid = target - contribs.sum(axis=1) - intercept        # |sum_k c_k phi_k|
        term_mass = np.abs(target) + np.abs(contribs).sum(axis=1) + abs(intercept)
        rho = np.abs(resid) / term_mass
        return float(np.mean(rho))


class Instability(EquationObjective):
    """Varying-coefficient instability (the metric removed from all five
    operators, consolidated here once).

    Fast path: sum the per-term ``_cached_vc_score`` produced by
    ``PhysicsInformedLasso`` (``VWSRSparsity``). Fallback (LASSO / axis
    paths, where no cache exists): ``vc_stability_total_lr`` for
    ``gram_mode='vcoef'``, else the sliding-window CV via
    ``calculate_weights``. On any failure returns ``1.0`` (matching the
    old ``try/except`` guards).
    """
    name = 'instability'
    value_attr = 'coefficients_stability'
    flag_attr = 'stability_calculated'

    def compute(self, equation, ctx: FitContext) -> float:
        cached = getattr(equation, '_cached_vc_score', None)
        if cached is not None:
            return float(np.sum(cached))
        try:
            data_shape = ctx.data_shape
            _, target, features = equation.evaluate(normalize=True, return_val=False)
            if features is None:
                return 1.0
            fit_intercept = bool(equation.weights_internal[-1] != 0)
            if global_var.gram_mode == 'vcoef':
                return float(vc_stability_total_lr(
                    features, target, ctx.g_fun_vals, data_shape,
                    main_var=equation.main_var_to_explain,
                    fit_intercept=fit_intercept))
            sw = getattr(equation, '_cached_sw_weights', None)
            if sw is None:
                sw = calculate_weights(
                    features, target, ctx.g_fun_vals, data_shape, fit_intercept,
                    gram_cls=None, gram_kwargs=None)
            sw_arr = np.array(sw)
            mu = sw_arr.mean(axis=0)
            std = sw_arr.std(axis=0, ddof=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = (std ** 2) / (mu ** 2)
                cv[mu == 0] = 0.0
            return float(np.sum(np.nan_to_num(cv)) / len(data_shape))
        except Exception:
            return 1.0


class L2RelativeDiscrepancy(EquationObjective):
    """L2 relative residual -- an L2 analogue of :class:`WAPEDiscrepancy`,
    selectable as a primary ``discrepancy_metric`` alongside ``'wape'`` and
    ``'scale_invariant'``:

        ||target - sum_j c_j phi_j||_2 / ||target||_2

    WAPE is the L1 (``sum|.| / sum|.|``) form of the same relative residual;
    this is the L2 (Euclidean-norm) form. ``L2Discrepancy`` computes the same
    residual norm but WITHOUT the ``/ ||target||`` normalisation (and weights it
    by ``g_func``); this one normalises by the right-part norm and -- like WAPE /
    ScaleInvariant -- does not ``g_func``-weight.

    A primary discrepancy filler: it owns the right-part-selection scaffolding
    (un-normalised term set during the ``force_out_of_place`` sweep, normalised
    in-place; ``is_degenerate`` on the ``weights_internal[:-1]`` all-zero test
    and the post-fit ``degenerate_threshold``) exactly as
    :class:`WAPEDiscrepancy`, so it can drive ``EqRightPartSelector`` and the
    Pareto quality axis as a drop-in for WAPE.
    """
    name = 'discrepancy'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'

    is_degenerate = _degenerate_excluding_intercept

    def compute(self, equation, ctx: FitContext) -> float:
        normalize = not ctx.for_rps
        _, target, features = equation.evaluate(normalize=normalize, return_val=False)
        target = np.asarray(target, dtype=float)
        if features is None:
            discr = target - target.mean()
        else:
            coefs, intercept = _extract_coefs_intercept(equation)
            discr = target - (np.dot(features, coefs) + intercept)
        den = float(np.linalg.norm(target))
        if den <= 0.0:
            return float(LOSS_NAN_VAL)
        return float(np.linalg.norm(discr) / den)


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


class SolverObjective:
    """Base class for a solver-based objective filler."""
    name = 'objective'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'

    def compute(self, eq, eq_idx: int, sctx: SolverContext) -> float:
        raise NotImplementedError


class SolverL2Discrepancy(SolverObjective):
    """L2 of (solved field - data), weighted by ``g_func``, plus the PINN
    residual loss (the legacy ``SolverBasedFitness`` core)."""
    name = 'discrepancy'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'

    def compute(self, eq, eq_idx, sctx):
        if _loss_is_nan(sctx.loss_add):
            return 2 * LOSS_NAN_VAL
        ref = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))
        sol = sctx.solution[..., eq_idx]
        discr = sol - ref.reshape(sol.shape)
        discr = np.multiply(discr, sctx.g_fun_vals.reshape(discr.shape))
        rl_error = np.linalg.norm(discr, ord=2)
        fitness = rl_error + sctx.pinn_loss_mult * float(sctx.loss_add)
        if np.sum(eq.weights_final) == 0:
            fitness /= sctx.penalty_coeff
        return float(fitness)


class PICError(SolverObjective):
    """PIC p-loss: mean squared (solved field - data) weighted by ``g_func``
    plus the PINN residual loss (the legacy ``PIC`` p-loss core)."""
    name = 'discrepancy'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'

    def compute(self, eq, eq_idx, sctx):
        if _loss_is_nan(sctx.loss_add):
            return 2 * LOSS_NAN_VAL
        ref = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))
        sol = sctx.solution[..., eq_idx]
        discr = sol - ref.reshape(sol.shape)
        discr = np.multiply(discr, sctx.g_fun_vals.reshape(discr.shape))
        rl_error = np.mean(discr ** 2)
        return float(rl_error + sctx.pinn_loss_mult * float(sctx.loss_add))


class DeepXDEError(SolverObjective):
    """Error of (DeepXDE solution - data) under a configurable metric
    (the legacy ``DeepXDEBasedFitness._compute_error`` core).

    The host supplies, per equation, an already-masked ``(solution, data)``
    pair on ``sctx`` via the special ``solution`` carrying the masked
    solution and ``g_fun_vals`` carrying the masked target.
    """
    name = 'discrepancy'
    value_attr = 'fitness_value'
    flag_attr = 'fitness_calculated'

    def __init__(self, error_metric: str = 'rmse', penalty_coeff: float = 0.2):
        self.error_metric = error_metric
        self.penalty_coeff = penalty_coeff

    def compute(self, eq, eq_idx, sctx):
        # sctx.solution[eq_idx] = masked solution, sctx.g_fun_vals[eq_idx] =
        # masked data (packed by SolverBasedFitness's deepxde branch).
        masked_solution = sctx.solution[eq_idx]
        masked_data = sctx.g_fun_vals[eq_idx]
        metric = self.error_metric
        if metric == 'l2':
            err = np.linalg.norm(masked_solution - masked_data, ord=2)
        elif metric == 'mae':
            err = np.mean(np.abs(masked_solution - masked_data))
        else:  # 'rmse' default
            err = np.sqrt(np.mean((masked_solution - masked_data) ** 2))
        if np.sum(eq.weights_final) == 0:
            err /= self.penalty_coeff
        return float(err)
