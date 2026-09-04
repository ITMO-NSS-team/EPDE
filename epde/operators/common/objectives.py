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
  right-part-selection hooks (``is_degenerate`` /
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
from epde.interface.search_config import active_config
from epde.operators.common.stability import (calculate_weights, cv_scores,
                                             vc_stability_total_lr)
from epde.operators.utils.template import CompoundOperator, dictApplyUFunc, dictZerosLike, dictFullLike, \
    dictAdd, dictSubtr
from epde.operators.common.survival import _BASIS_FREE_METRICS

LOSS_NAN_VAL = 1e7


def _extract_coefs_intercept(equation, features=None):
    """Reconstruct ``(coefs, intercept)`` from ``weights_final``.

    Under the unified layout (``Equation._validate_weight_layout``)
    ``weights_final`` is always ``[*one coef per non-target term, intercept]``
    with zeros retained, so the split is purely positional.

    Pass ``features`` whenever the coefficients are about to be dotted with a
    feature matrix: ``Equation.evaluate`` emits TWO widths from one structure --
    the default returns every non-target term (m columns) while
    ``active_only=True`` returns only the terms with a non-zero
    ``weights_internal`` slot (nnz columns) -- and ``coefs`` is always m-long.
    The narrow case is masked by ``equation.active_mask``.

    Before the unification ``weights_final`` was itself zero-filtered to nnz+1
    by the sparsity operators, which lined up with the narrow width and
    silently mismatched the wide one unless ``remove_zero_terms`` had already
    collapsed nnz onto m. The older ``if equation.weights_internal[-1]:`` guard
    was worse still: it read a presence FLAG that never existed."""
    coefs = np.asarray(equation.weights_final[:-1])
    intercept = equation.weights_final[-1]
    if features is None:
        return coefs, intercept

    widths = {np.asarray(value).shape[1] for value in features.values()
              if value is not None and np.asarray(value).ndim > 1}
    if not widths or widths == {coefs.size}:
        return coefs, intercept
    if len(widths) > 1:
        raise ValueError(
            f'feature matrices disagree on width across trajectories: {sorted(widths)}')
    width = widths.pop()
    mask = equation.active_mask
    if width != int(mask.sum()):
        raise ValueError(
            f'feature matrix has {width} columns, which matches neither the '
            f'{coefs.size} non-target terms nor the {int(mask.sum())} active '
            'ones; weights and structure have desynced')
    return coefs[mask], intercept


def _term_weights(equation):
    """The per-term coefficients of ``weights_internal``, positionally aligned
    to the non-target terms of ``equation.structure`` -- i.e. everything but
    the trailing intercept slot, which the unified layout always carries
    (``Equation._validate_weight_layout``).

    Named rather than inlined because the distinction is load-bearing: a
    degeneracy test must ask whether every TERM weight is zero, and the raw
    ``np.all(weights_internal == 0)`` would additionally demand a zero
    intercept."""
    return np.asarray(equation.weights_internal[:-1])


def _mean_of_sums(per_sample) -> float:
    """Reduce a per-trajectory dict of per-term score vectors to one scalar:
    sum within a trajectory, mean across trajectories. ``None`` entries (a
    sample whose estimator produced nothing) are skipped; an all-``None`` dict
    reduces to 0.0."""
    totals = [float(np.sum(v)) for v in per_sample.values() if v is not None]
    return float(np.mean(totals)) if totals else 0.0


def _relative_norm(residual, reference) -> float:
    """``||residual|| / ||reference||`` -- the solver discrepancy, made a ratio.

    Falls back to the bare norm when the reference is identically zero (there
    is no scale to divide by, and 0/0 must not reach the Pareto front).
    """
    scale = float(np.linalg.norm(reference, ord=2))
    value = float(np.linalg.norm(residual, ord=2))
    return value / scale if scale > 0. else value


def _relative_mean_square(residual, reference) -> float:
    """``mean(residual^2) / mean(reference^2)`` -- the 'pic' counterpart."""
    scale = float(np.mean(np.asarray(reference) ** 2))
    value = float(np.mean(np.asarray(residual) ** 2))
    return value / scale if scale > 0. else value


def _degenerate_excluding_intercept(filler, equation) -> bool:
    """Shared ``is_degenerate`` for the VWSR-paired discrepancy fillers:
    degenerate iff every non-intercept weight is zero, or the post-fit
    discrepancy exceeds the filler's ``degenerate_threshold``."""
    fv = getattr(equation, 'fitness_value', None)
    return bool(np.all(_term_weights(equation) == 0)
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

    # The best attainable value on this axis, i.e. this objective's
    # contribution to MOEA/D's ideal point. It belongs to the objective, not
    # to whoever assembles the front: ``[0., 1.]`` was never "the ideal for
    # not-use_pic", it is COMPLEXITY's ideal, because the least complex
    # equation has one factor. Deriving the ideal point from these attributes
    # (see ``ideal_point``) is what keeps the three lockstep sites -- filler
    # assembly, SoEq axis registration, and the ideal point -- from drifting
    # apart, and means a new second-axis objective only has to declare its own
    # value here.
    ideal_value = 0.0

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
    # A candidate target whose post-fit discrepancy exceeds this is declined
    # during the RPS term-sweep (a degenerate target fits too poorly to be the
    # right part). Checked AFTER compute() has set ``fitness_value`` -- see
    # ``SolverFreeFitness.apply``; reading it before compute() gives the stale /
    # ``None`` default and the check silently never fires.
    degenerate_threshold = 1.0

    def is_degenerate(self, equation) -> bool:
        fv = getattr(equation, 'fitness_value', None)
        # ``_term_weights``, not the raw vector: the unified layout carries a
        # trailing intercept slot, and an equation is degenerate when its TERMS
        # are all zero regardless of whether a constant survived.
        return bool(np.all(_term_weights(equation) == 0)
                    or (fv is not None and fv > self.degenerate_threshold))

    def compute(self, equation, ctx: FitContext) -> float:
        raise NotImplementedError


class Discrepancy(EquationObjective):
    """The DISCREPANCY objective family: one filler containing every residual
    metric as an option, BUILT FROM THE SEARCH CONFIGURATION -- ``Discrepancy()``
    constructs bare and resolves its option at compute time from
    ``active_config().objectives.discrepancy_metric``, which ``EpdeSearch``
    writes from its ``discrepancy_metric`` kwarg (the same late-dispatch
    convention as :class:`Instability` and :class:`Complexity`). An explicit
    constructor metric exists only as internal wiring for fixed-role hosts.

    Options (canonical):

    * ``'wape'`` (default) -- normalised absolute residual, the legacy
      ``L2LRFitness`` core: ``sum|target - fit| / sum|target|``. Normalised
      features in-place, un-normalised during the RPS sweep, no ``g_func``
      weighting and no penalty division.
    * ``'l2'`` -- weighted L2 norm of the residual, the legacy ``L2Fitness``
      core: un-normalised features, ``g_func`` weighting, and the all-zero
      penalty division. NO target-scale normalisation.
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
    (right-part-selection scaffolding): ``is_degenerate`` branches --
    ``'l2'`` keeps its legacy
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
        ``active_config().objectives.discrepancy_metric``, which ``EpdeSearch``
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
                else active_config().objectives.discrepancy_metric)

    def is_degenerate(self, equation) -> bool:
        if self._resolved_metric() == 'l2':
            # Legacy L2Fitness contract: all-zero TERM weights only, no
            # threshold (see the base is_degenerate on the intercept slot).
            return bool(np.all(_term_weights(equation) == 0))
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
        # L2LRFitness restricted the design matrix to the currently ACTIVE
        # support on the RPS sweep (force_out_of_place) and used every
        # non-target term on the in-place pass. ``active_only`` selects columns;
        # nothing here is normalised (see Equation.evaluate).
        targets, features = equation.evaluate(active_only=ctx.for_rps)
        if features is None:
            discr = dictSubtr(targets, {key: tg.mean() for key, tg in targets.items()})
        else:
            coefs, intercept = _extract_coefs_intercept(equation, features)
            discr = dictSubtr(targets,
                              dictAdd(dictApplyUFunc(np.dot, features, coefs),
                                      intercept))
        # MEAN over trajectories, not sum: ``degenerate_threshold`` is a
        # PER-SAMPLE WAPE bound (1.0), and summing made it n_samples times
        # stricter -- with three trajectories every RPS candidate above a mean
        # WAPE of 0.33 was declined, the sweep hit inf fitness and rerolled the
        # whole equation. Identical to the legacy value when len(targets) == 1.
        rl_error = np.mean([np.sum(np.abs(discr[key])) / np.sum(np.abs(targets[key]))
                            for key in targets.keys()])
        return float(rl_error)

    def _compute_l2(self, equation, ctx: FitContext) -> float:
        targets, features = equation.evaluate(active_only=True)

        if features is None or all([feature is None for feature in features.values()]):
            discr_feats = dictZerosLike(targets, 0)
        else:
            # ``_extract_coefs_intercept`` now performs the width match that
            # this ad-hoc ``n_cols == len(mask) or n_cols == mask.sum()`` test
            # only checked -- and it NARROWS the coefficients rather than
            # assuming ``weights_final`` was already zero-filtered to fit.
            coefs, intercept = _extract_coefs_intercept(equation, features)
            discr_feats = dictApplyUFunc(np.dot, features, coefs)

        # ``dictFullLike`` takes the REFERENCE DICT, not a ``.shape`` (targets is
        # a per-trajectory dict and has no such attribute).
        discr = dictSubtr(dictAdd(discr_feats,
                                  dictFullLike(targets, equation.weights_final[-1])),
                          targets)

        g = ctx.g_fun_vals
        if g is not None:
            # ``ctx.g_fun_vals`` IS the weighting; there is no ``self.g_fun_vals``
            # on a filler (that attribute belonged to the old operator classes).
            discr = dictApplyUFunc(np.multiply, discr, g)

        # Weighted L2 norm per trajectory, summed -- the legacy L2Fitness core.
        # ``self.norm_prtl`` never existed on the class (the definition is
        # commented out at the top of Discrepancy).
        rl_error = dictApplyUFunc(lambda x: np.linalg.norm(x, ord=2), discr)
        rl_error = reduce(lambda x, y: x + y, rl_error.values(), 0.)

        if np.sum(equation.weights_final[:-1]) == 0:
            rl_error /= ctx.penalty_coeff

        return float(rl_error)

    def _compute_l2_relative(self, equation, ctx: FitContext) -> float:
        targets, features = equation.evaluate(active_only=ctx.for_rps)
        if features is None:
            discr = dictSubtr(targets, {key: target.mean() for key, target in targets.items()})
        else:
            coeffs, intercept = _extract_coefs_intercept(equation, features)
            discr = dictSubtr(targets, dictAdd(dictApplyUFunc(np.dot, features, coeffs), intercept))
        # ``.values()``: iterating the dict itself yields KEYS, so the norm was
        # taken of the integer sample ID -- 0.0 for a single-sample run, which
        # made this metric return LOSS_NAN_VAL unconditionally.
        den = float(np.mean([np.linalg.norm(target) for target in targets.values()]))
        if np.isclose(den, 0.0):
            return float(LOSS_NAN_VAL)
        return float(np.mean([np.linalg.norm(discrepancy) for discrepancy in discr.values()]) / den)

    def _compute_scale_invariant(self, equation, ctx: FitContext) -> float:
        targets, features = equation.evaluate(active_only=ctx.for_rps)
        if features is None:
            # only the target term survives -> a single term cannot cancel.
            return 1.0
        coeffs, intercept = _extract_coefs_intercept(equation, features)
        contribs = dictApplyUFunc(np.multiply, features, np.asarray(coeffs)[None, :])          # per-term weighted values
        # ``contribs`` is a per-trajectory dict; the term sum has to be taken
        # INSIDE each sample's array, not on the dict.
        resid = dictSubtr(targets, dictAdd(dictApplyUFunc(lambda x: x.sum(axis=1), contribs),
                                           intercept))                          # |sum_k c_k phi_k|
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

        # The old form was a SET comprehension that unpacked two names from
        # ``.values()`` and read ``discr`` before it existed.
        discr = dictSubtr(sol, ref)
        gvals = {key: np.asarray(gfunc_vals).reshape(discr[key].shape)
                 for key, gfunc_vals in sctx.g_fun_vals.items()}
        discr = dictApplyUFunc(np.multiply, discr, gvals)
        # RELATIVE to the reference, not the bare Euclidean norm. The old
        # unnormalised form grew like sqrt(n_points), so it was neither
        # grid-independent nor commensurable with the solver-free options
        # (``wape`` / ``l2_relative``) or with the PINN term it is summed
        # with. Measured on Allen-Cahn it spanned only 26-62 over truth and
        # junk alike -- no discrimination -- while ``wape`` on the same
        # candidates separated 0.003-0.009 (near-truth) from 0.38-1.02 (junk).
        # As a ratio it is also self-diagnosing: a solve that did not
        # reproduce the data lands at or above 1.
        rl_error = float(np.mean([_relative_norm(value, ref[key] * gvals[key])
                                  for key, value in discr.items()]))

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

        # See _compute_solver_l2: set comprehension -> per-sample dict.
        discr = dictSubtr(sol, ref)
        gvals = {key: np.asarray(gfunc_vals).reshape(discr[key].shape)
                 for key, gfunc_vals in sctx.g_fun_vals.items()}
        discr = dictApplyUFunc(np.multiply, discr, gvals)

        # Relative, for the same reason as ``_compute_solver_l2`` -- the two
        # solver options were on different scales (bare 2-norm there, bare
        # mean-square here), so the family could not be compared with itself.
        rl_error = float(np.mean([_relative_mean_square(value, ref[key] * gvals[key])
                                  for key, value in discr.items()]))
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
            # ``dictSubtr`` below already forms (solution - data); subtracting
            # ``masked_data`` again here double-counted it.
            err_func = lambda x: np.linalg.norm(x, ord=2)
        elif metric == 'mae':
            err_func = lambda x: np.mean(np.abs(x))
        else:  # 'rmse' default
            err_func = lambda x: np.sqrt(np.mean(x**2))
            # err = np.sqrt(np.mean((masked_solution - masked_data) ** 2))
        err = np.mean(list(dictApplyUFunc(err_func, dictSubtr(masked_solution, masked_data)).values()))

        if np.sum(eq.weights_final) == 0:
            err /= self.penalty_coeff
        return float(err)

class Instability(EquationObjective):
    """The instability objective, dispatching on the estimator selected by
    ``active_config().objectives.instability_metric``:

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
        metric = active_config().objectives.instability_metric
        if metric == 'vcoef':
            cached = getattr(equation, '_cached_vc_score', None)
            if cached is not None:
                # Per-trajectory dict written by VWSRSparsity: sum each
                # sample's per-term scores, then average across trajectories --
                # the same reduction the survival/chi2 branch below uses.
                return _mean_of_sums(cached)
        elif metric in _BASIS_FREE_METRICS:
            cached = getattr(equation, '_cached_alt_instability', None)
            if cached is not None and cached[0] == metric:
                return float(cached[1])
        data_shape = ctx.data_shape
        targets, features = equation.evaluate()

        if features is None:
            return 1.0
        # THE INTERCEPT RULE: a regularized-away intercept is not a column of
        # any later model. ``weights_internal[-1]`` is the sparsity step's
        # SUPPORT decision (``weights_final[-1]`` agrees under the unified
        # layout, but this is the vector that decides). The flag propagates
        # unchanged into vc_stability_total_lr / calculate_weights / the
        # survival.py estimators, each of which appends its own ones column
        # only when it is set.
        fit_intercept = bool(equation.weights_internal[-1] != 0)
        if metric == 'vcoef':
            # Forward the TRAJECTORY KEY. ``VaryingCoefSetup`` resolves its
            # cosine basis from that sample's own field via the Taylor
            # microscale, and ``sample_key=None`` silently means trajectory 0
            # (``stability.resolve_vc_modes_from_input``), so without this
            # every trajectory was scored on trajectory 0's basis while the
            # keep-rule (``sparsity.py``, ``subset_selection.py``) already
            # passed each key -- the two sides resolving one statistic on
            # different bases. ``dictApplyUFunc`` fans out over dict args
            # positionally and does not hand the callee its key, so the keys
            # are passed AS a dict.
            def _vc_total(feats, tgt, g_vals, shape, key):
                return vc_stability_total_lr(
                    feats, tgt, g_vals, shape,
                    main_var=equation.main_var_to_explain,
                    fit_intercept=fit_intercept, sample_key=key)

            totals = dictApplyUFunc(_vc_total, features, targets,
                                    ctx.g_fun_vals, data_shape,
                                    {key: key for key in features})
            return float(np.mean([float(v) for v in totals.values()]))
        if metric in _BASIS_FREE_METRICS:
            estimator = _BASIS_FREE_METRICS[metric]
            estim = partial(estimator, fit_intercept=fit_intercept)
            scores = dictApplyUFunc(estim, features, targets, ctx.g_fun_vals, data_shape)
            value = np.mean([float(np.sum(score)) for score in scores.values()])
            equation._cached_alt_instability = (metric, value)
            return value
        # metric == 'cv': the axis-aligned sliding-window CV, per trajectory.
        sw = getattr(equation, '_cached_sw_weights', None)
        if sw is None:
            estim = partial(calculate_weights, fit_intercept=fit_intercept,
                            gram_cls=None, gram_kwargs=None)
            sw = dictApplyUFunc(estim, features, targets, ctx.g_fun_vals, data_shape)
        per_sample = []
        for key, sw_key in sw.items():
            if sw_key is None:
                continue
            # THE reduction of the window stack, not a copy of it. This block
            # used to re-derive ``var / mu^2`` inline, so the keep-rule and
            # this axis ran two implementations of one statistic and nothing
            # kept them in step -- exactly the drift the shared estimator
            # table now prevents for the basis-free metrics.
            cv = cv_scores(sw_key)
            # Divide by that trajectory's GRID DIMENSIONALITY (the number of
            # per-dim window batches ``calculate_weights`` stacks), not by the
            # trajectory count.
            n_dims = max(len(np.atleast_1d(data_shape[key])), 1)
            per_sample.append(float(np.sum(np.nan_to_num(cv))) / n_dims)
        if not per_sample:
            return 1.0
        return float(np.mean(per_sample))


class Complexity(EquationObjective):
    """The COMPLEXITY objective family: parsimony of the fitted structure,
    with the option selected per instance (or, when constructed without one,
    from the search-level ``objectives.complexity_metric``).

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
    # One factor is the least complex an equation can be under either option,
    # so 1.0 -- not 0.0 -- is the attainable optimum on this axis.
    ideal_value = 1.0

    OPTIONS = ('factors', 'terms')

    def __init__(self, metric: str = None):
        if metric is not None and metric not in self.OPTIONS:
            raise ValueError(
                f'complexity metric must be one of {self.OPTIONS} or None '
                f'(= follow objectives.complexity_metric); got {metric!r}')
        self.metric = metric

    def compute(self, equation, ctx: FitContext) -> float:
        # Local import: the per-equation cores live in eq_mo_objectives (the
        # readers' module), which main_structures also imports locally -- the
        # same cycle-dodging idiom.
        from epde.eq_mo_objectives import (_complexity_of_equation,
                                           _terms_of_equation)
        metric = (self.metric if self.metric is not None
                  else active_config().objectives.complexity_metric)
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


# ---------------------------------------------------------------------------
# The ideal point, owned by the objectives
# ---------------------------------------------------------------------------
# MOEA/D needs the best attainable value on each axis. That used to be a
# literal in ``EpdeSearch._create_optimizer`` selected by the ``use_pic``
# bool -- so the ideal point could disagree with the axis actually in use
# (a 'complexity' second objective with ``use_pic=True`` left the ideal
# at the instability value), and a third second-axis objective would have
# silently inherited complexity's ideal. Now each objective declares its own
# ``ideal_value`` and the point is assembled from the names of the objectives
# in play.

#: Objective family by the name it is selected under. Keys match
#: ``EquationObjective.name`` and the values accepted by
#: ``objectives.second_objective`` / ``objectives.instability_metric``.
OBJECTIVE_REGISTRY = {
    'discrepancy': Discrepancy,
    'instability': Instability,
    'complexity': Complexity,
}


def ideal_point(objective_names):
    """The ideal objective vector for the axes named in ``objective_names``.

    Args:
        objective_names: ordered names of the objectives on the front, e.g.
            ``('discrepancy', 'instability')``. The first axis is always the
            discrepancy fitness.

    Returns:
        list of float -- one ``ideal_value`` per axis, in the same order.
    """
    unknown = [name for name in objective_names if name not in OBJECTIVE_REGISTRY]
    if unknown:
        raise ValueError(
            'No objective registered under {0}; known objectives are {1}. A '
            'new objective must be added to OBJECTIVE_REGISTRY and declare '
            'its own ideal_value.'.format(unknown, sorted(OBJECTIVE_REGISTRY)))
    return [float(OBJECTIVE_REGISTRY[name].ideal_value)
            for name in objective_names]
