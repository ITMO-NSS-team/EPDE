#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:20:59 2021

@author: mike_ubuntu

Fitness host operators. Each owns ONE responsibility: SCORE an
already-fitted equation, delegating every scalar objective to a pluggable
*filler* from ``epde.operators.common.objectives``.

The hosts have no suboperators. Support selection (``sparsity``) and the
coefficient refit (``coeff_calc``) belong to ``EqRightPartSelector``, which
runs both before every call into a host -- per candidate target during its
term-sweep, and once for the winner -- and is the only operator that prunes
zero-weight terms.

* :class:`SolverFreeFitness` -- gene-level; hosts solver-free fillers
  (``Discrepancy`` / ``Instability`` / ``Complexity`` / ...).
  Replaces the former ``L2Fitness`` and ``L2LRFitness``.
* :class:`SolverBasedFitness` -- chromosome-level; solves the system once
  via a PDE backend (autograd ``SolverAdapter`` or DeepXDE) and hosts
  solver-based fillers (``SolverL2Discrepancy`` / ``PICError`` /
  ``DeepXDEError``) plus an optional ``Instability`` r-loss. Replaces the
  former ``SolverBasedFitness``, ``PIC`` and ``DeepXDEBasedFitness``.

The metric logic lives in the fillers (single responsibility); the hosts
only build the scoring context and write each filler's value to its attribute.
"""
import warnings
from copy import deepcopy

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import cm

from epde.integrate import SolverAdapter
# DeepXDEAdapter is imported lazily inside the deepxde backend to avoid
# triggering deepxde's import-time backend banner when no DeepXDE solver
# is used (the solver-free path never imports it).
from epde.structure.main_structures import SoEq, Equation
from epde.operators.utils.template import CompoundOperator
import epde.globals as global_var
from epde.interface.search_config import active_config
# Re-exported so ``from epde.operators.common.fitness import
# vc_stability_total_lr`` keeps working for external callers
# (e.g. projects/thesis/_vc_cache_gate.py).
from epde.operators.common.stability import (calculate_weights, vc_stability_total_lr)  # noqa: F401
from epde.operators.common.objectives import (
    FitContext, SolverContext, EquationObjective, # SolverObjective,
    Discrepancy, Instability, # L2Discrepancy, 
    LOSS_NAN_VAL, #  SolverL2Discrepancy, PICError, DeepXDEError, 
)
from epde import _loop_stats


class SolverFreeFitness(CompoundOperator):
    """Solver-free fitness host (gene level).

    A PURE SCORER: it has no suboperators, fits nothing and changes no
    structure. Given an equation whose support and coefficients are already
    set, it evaluates each configured objective filler and stores that
    filler's scalar on the equation. The ``primary`` filler (a discrepancy)
    drives the right-part-selection scaffolding and is the value returned for
    ``EqRightPartSelector``'s ``force_out_of_place`` term-sweep.

    Fitting is :class:`~epde.operators.common.right_part_selection.EqRightPartSelector`'s
    job: it applies ``sparsity`` then ``coeff_calc`` before every call into
    here, and is the only operator that prunes (``remove_zero_terms``, once,
    after its sweep). Scoring an equation with no ``weights_internal``
    raises rather than quietly fitting it.

    Parameters
    ----------
    param_keys : list
        Operator parameter names (``['penalty_coeff']``).
    objectives : list of EquationObjective
        Fillers to evaluate in-place; each writes its own attribute.
    primary : EquationObjective, optional
        The discrepancy filler used for RPS / force_out_of_place. Defaults
        to ``objectives[0]``.
    """
    key = 'SolverFreeFitness'

    def __init__(self, param_keys: list = None, objectives: list = None,
                 primary: EquationObjective = None):
        super().__init__(param_keys if param_keys is not None else ['penalty_coeff'])
        self.objectives = list(objectives) if objectives else []
        self.primary = primary if primary is not None else (
            self.objectives[0] if self.objectives else None)

    @_loop_stats.timed('SolverFreeFitness.apply')
    def apply(self, objective: Equation, arguments: dict, force_out_of_place: bool = False):
        penalty_coeff = self.params['penalty_coeff']
        if not (penalty_coeff > 0. and penalty_coeff < 1.):
            raise ValueError('Incorrect penalty coefficient set, value shall be in (0, 1).')

        primary = self.primary
        # THIS HOST SCORES; IT DOES NOT FIT. The support decision (sparsity)
        # and the coefficient refit belong to EqRightPartSelector, which runs
        # both before every call into here -- once per candidate target during
        # its term-sweep, and once more for the winner. Owning them here is
        # what let the in-place pass silently re-sparsify an already-fitted
        # equation and then prune its structure, which in turn made the
        # post-RPS structural label unreliable for the MOEA/D history dedup.
        if not getattr(objective, 'weights_internal_evald', False):
            raise RuntimeError(
                'SolverFreeFitness: scoring an equation with no support '
                'decision. Sparsity must run before fitness -- '
                'EqRightPartSelector owns it (suboperators "sparsity" and '
                f'"coeff_calc"). target={objective.main_var_to_explain!r}')
        # During the RPS term-sweep a degenerate (all-zero-weight) candidate is
        # skipped by returning None; in-place we always fall through to a
        # finite value.
        if force_out_of_place and primary.is_degenerate(objective):
            return None

        g_fun_vals = global_var.samples_manager.gFunc('dmf')
        data_shapes = global_var.samples_manager.inner_shapes
        # print(data_shapes)
        # try:
        #     g_fun_vals = global_var.samples_manager.gFunc('dmf')
        # except AttributeError:
        #     g_fun_vals = None
        # try:
        #     data_shapes = global_var.samples_manager.inner_shapes
        # except AttributeError:
        #     data_shapes = None
        # raise NotImplementedError('Implement me!')
        # TODO: FIX!
        ctx = FitContext(g_fun_vals=g_fun_vals, data_shape=data_shapes,
                         penalty_coeff=penalty_coeff, for_rps=force_out_of_place)

        if force_out_of_place:
            val = primary.compute(objective, ctx)
            objective.fitness_value = val
            # Post-fit degeneracy: decline a candidate target that still fits
            # poorly (discrepancy above the filler's degenerate_threshold) OR
            # that only reaches the target through an amplified near-null
            # combination (guard ratio above ``rps_amplification_cap``).
            # Either way return None -- exactly like the all-features-zeroed
            # sparsity case above -- so the RPS sweep never considers the
            # candidate at all.
            if primary.is_degenerate(objective) or _amplification_trips(objective):
                return None
            return val

        # NOTHING IS PRUNED HERE. The single prune in the pipeline is the
        # ``remove_zero_terms`` at the end of EqRightPartSelector.apply, once
        # its sweep has chosen a winner; the in-place counterpart that used to
        # live at this point is gone. Leaving zero-weight terms in the
        # structure is safe under the unified coefficient layout
        # (Equation._validate_weight_layout): both weight vectors are
        # structure-aligned with zeros retained, so text_form / latex_form
        # index correctly, ``_extract_coefs_intercept`` reconciles the wide and
        # active_only column widths, and the complexity cores count
        # weights_internal non-zeros rather than structure length. The comment
        # that used to defend the prune described the retired nnz+1 layout.

        # An equation that RPS left unfitted -- e.g. every candidate target in its
        # term-sweep was declined by the degeneracy check (discrepancy over the
        # threshold) -- has no final weights. Score it as maximally degenerate so
        # MOEA/D selects it out, instead of crashing on weights_final access.
        if not getattr(objective, 'weights_final_evald', False):
            for filler in self.objectives:
                # Complexity opts out (stamped_on_failure=False): flag stays
                # down, so the equation_complexity reader falls back to the
                # lazy structure-derived cores -- the exact pre-filler
                # legacy semantics for RPS-exhausted equations.
                if not getattr(filler, 'stamped_on_failure', True):
                    continue
                setattr(objective, filler.value_attr, LOSS_NAN_VAL)
                setattr(objective, filler.flag_attr, True)
            objective.aic = None
            objective.aic_calculated = True
            return

        for filler in self.objectives:
            setattr(objective, filler.value_attr, filler.compute(objective, ctx))
            setattr(objective, filler.flag_attr, True)
        # The RPS-time is_degenerate check (fitness.py force_out_of_place branch)
        # only declines candidate TARGETS; it cannot evict an already-fitted form
        # that is non-dominated via an artificially-low instability (e.g. a single
        # -term SInv cancellation residual of 1.0 with stability ~0). Apply the
        # primary's degeneracy verdict here too -- now that fitness_value holds the
        # in-place discrepancy -- so a degenerate form is dominated out of the front.
        # An amplified fit (``_amplification_trips``: e.g. the LV
        # ``Lambda*(du+dv-alpha*u+gamma*v) = d^2u`` parasite) is degenerate
        # NO MATTER which path installed its target: the term-sweep already
        # returns None for such candidates, but the sweep's exit-guarantee
        # fallback can still install one -- this backstop keeps the amplified
        # fit off the front regardless of how its weights arose.
        if primary is not None and (primary.is_degenerate(objective)
                                    or _amplification_trips(objective)):
            for filler in self.objectives:
                # Complexity opts out (stamped_on_failure=False): its value
                # is structure-derived and stays real -- stamping it would
                # change the legacy Pareto geometry for degenerate forms.
                if getattr(filler, 'stamped_on_failure', True):
                    setattr(objective, filler.value_attr, LOSS_NAN_VAL)
        # AIC is not produced by the solver-free path; expose the default
        # the legacy WAPE operator set so downstream readers don't assert.
        objective.aic = None
        objective.aic_calculated = True

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'gene level', 'no suboperators', 'inplace'}


def _amplification_trips(objective) -> bool:
    """True when the CURRENT fit only reaches its target by amplifying the
    residual of a near-null feature combination: guard ratio
    A = sum|c|*||col|| / ||target|| above ``search_space.rps_amplification_cap``
    (None disables). Shared by the RPS-sweep decline (out-of-place -> return
    None, exactly like the all-features-zeroed case) and the in-place
    condemnation stamp, so the verdict is identical no matter which path
    installed the target. Deferred import: the ratio lives with the RPS
    machinery."""
    cap = active_config().search_space.rps_amplification_cap
    if cap is None:
        return False
    from epde.operators.common.right_part_selection import amplification_ratio
    if amplification_ratio(objective) > cap:
        _loop_stats.record('EqRPS.amplification_decline', 1, 1)
        return True
    return False


class SolverBasedFitness(CompoundOperator):
    """Solver-based fitness host (chromosome level).

    Solves the candidate system once with a PDE backend, then scores each
    equation with the configured solver fillers. Subsumes the former
    ``SolverBasedFitness`` (``backend='autograd'``, ``masked=False``,
    ``Discrepancy('solver_l2')``), ``PIC`` (``backend='autograd'``,
    ``masked=True``, ``Discrepancy('pic')`` + ``Instability``) and
    ``DeepXDEBasedFitness`` (``backend='deepxde'``,
    ``Discrepancy('deepxde')`` + ``Instability``).

    The right-part-selection term-sweep never solves: the director wires a
    lightweight :class:`SolverFreeFitness` as the RPS fitness instead.
    """
    key = 'SolverBasedFitness'

    def __init__(self, param_keys: list, objectives: list = None,
                 primary: Discrepancy = None, instability: Instability = None,
                 backend: str = 'autograd', masked: bool = False):
        super().__init__(param_keys)
        self.adapter = None
        #: Resolved in ``set_adapter``; the device the net and the grids share.
        self.solver_device = 'cpu'
        self.backend = backend
        self.masked = masked
        self.objectives = list(objectives) if objectives else []
        # objectives[0] fallback: see SolverFreeFitness.__init__ -- pass
        # ``primary`` explicitly with a discrepancy filler.
        self.primary = primary if primary is not None else (
            self.objectives[0] if self.objectives else None)
        # ``instability=`` is a legacy convenience channel: the filler is
        # folded into the uniform ``objectives`` list, and the host computes
        # every non-primary filler with the solver-free FitContext protocol
        # after the solver pass -- exactly like SolverFreeFitness consumes
        # its second-objective filler.
        if instability is not None and all(instability is not obj for obj in self.objectives):
            self.objectives.append(instability)

    @staticmethod
    def _resolved_device() -> str:
        """The device the solver runs on.

        ``torch.cuda.is_available`` used to be read as a FUNCTION OBJECT and
        never called -- always truthy -- next to a hardcoded
        ``explicit_cpu = False``, so this said 'cuda' unconditionally and
        ignored ``solver.device`` outright. The solver then put its net on cuda
        while the grids stayed on cpu.
        """
        device = active_config().solver.device
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            warnings.warn(
                f"solver.device={device!r} requested but CUDA is not "
                "available; falling back to cpu.",
                global_var.EPDEUsageWarning, stacklevel=3)
            return 'cpu'
        return device

    def set_adapter(self, net=None, pretrained_net=None):
        if self.backend == 'deepxde':
            if self.adapter is None:
                from epde.integrate.deepxde_integration import DeepXDEAdapter
                cfg = self.params.get('deepxde_config', {})
                self.adapter = DeepXDEAdapter(pretrained_net=pretrained_net, **cfg)
            return
        # Resolved unconditionally, not just when the adapter is (re)built:
        # ``_apply_autograd`` has to put its grid stack on the same device the
        # trained net ends up on.
        self.solver_device = self._resolved_device()
        if self.adapter is None or net is not None:
            compiling_params = {'mode': 'autograd', 'tol': 0.01, 'lambda_bound': 100}
            optimizer_params = {}
            training_params = {'epochs': 1e3, 'info_string_every': 1e3}
            early_stopping_params = {'patience': 4, 'no_improvement_patience': 250}
            self.adapter = SolverAdapter(net=net, use_cache=False,
                                         device=self.solver_device)
            self.adapter.set_compiling_params(**compiling_params)
            self.adapter.set_optimizer_params(**optimizer_params)
            self.adapter.set_early_stopping_params(**early_stopping_params)
            self.adapter.set_training_params(**training_params)

    def apply(self, objective: SoEq, arguments: dict, force_out_of_place: bool = False):
        # No suboperators: like SolverFreeFitness this host only scores, and
        # the system it is handed has already been fitted by
        # EqRightPartSelector. (The ``force_out_of_place`` sparsity call that
        # used to sit here was unreachable -- the RPS sweep dispatches to
        # ``fitness_calculation``, which on the solver path is a separate
        # lightweight SolverFreeFitness, never this host.)
        unfitted = [eq.main_var_to_explain for eq in objective.vals
                    if not getattr(eq, 'weights_internal_evald', False)]
        if unfitted:
            raise RuntimeError(
                'SolverBasedFitness: solving a system whose equations have no '
                'support decision. Sparsity must run before fitness -- '
                'EqRightPartSelector owns it (suboperators "sparsity" and '
                f'"coeff_calc"). unfitted={unfitted}')
        if self.backend == 'deepxde':
            return self._apply_deepxde(objective, force_out_of_place)
        return self._apply_autograd(objective, force_out_of_place)

    def _build_fit_context(self):
        try:
            g_fun_vals = global_var.samples_manager.gFunc('dmf')
        except AttributeError:
            g_fun_vals = None
        try:
            data_shape = global_var.samples_manager.inner_shapes
        except AttributeError:
            data_shape = None
        return g_fun_vals, data_shape

    @staticmethod
    def _pretrained_net():
        """The data-representation net, when one was trained.

        ``global_var.solution_guess_nn`` is written only by
        ``reset_data_repr_nn``, whose two call sites in ``interface.py`` are
        commented out -- so the name normally does NOT exist, and reading a
        missing module attribute raises ``AttributeError``. The old guard here
        caught ``NameError``, which is what a missing *local* raises, so every
        autograd solve died on its first line.
        """
        return deepcopy(getattr(global_var, 'solution_guess_nn', None))

    @staticmethod
    def _grid_stack(grids):
        """One trajectory's coordinates as a single ``(n_points, n_dims)``
        float tensor.

        Fed the ``mode='solver'`` grids: the reference data
        (``samples_manager.get``) and everything ``Equation.evaluate`` produces
        live on the INNER domain -- the boundary is pruned away -- so sampling
        the solution on the FULL grid, as this did, compared 100 points against
        80 values.
        """
        columns = [torch.as_tensor(np.asarray(grid).reshape(-1)) for grid in grids]
        return torch.stack(columns, dim=1).float()

    # NOTE: the caller moves the stack onto ``self.solver_device`` -- the net
    # is put there by SolverAdapter.solve, and the original code never moved
    # the grids at all, so a cuda run died in the first matmul.

    def _apply_autograd(self, objective, force_out_of_place):
        self.set_adapter(net=self._pretrained_net())

        print('solving equation:')
        print(objective.text_form)

        samples = global_var.samples_manager
        grids = samples.grids(mode='solver')
        # Both branches take the inner-domain weighting now: ``self.masked``
        # used to pick between the full-grid g and the masked one, but the
        # reference data has been inner-domain-only since the caches were
        # pruned, so the full-grid option no longer lines up with anything to
        # compare against.
        g_fun_vals = samples.gFunc('dmf')

        # One solve per trajectory: each carries its own domain, and the solver
        # fillers (Discrepancy's 'solver_l2' / 'pic' options) index
        # ``sctx.solution`` and ``sctx.g_fun_vals`` by trajectory key.
        solutions, losses = {}, []
        for domain_key in samples.trajecatoryIDs:
            loss_add, solution_nn = self.adapter.solve_epde_system(
                system=objective, domain_key=domain_key, grids=None,
                boundary_conditions=None, use_fourier=True)
            losses.append(loss_add)
            grid_stack = self._grid_stack(grids[domain_key]).to(self.solver_device)
            solutions[domain_key] = solution_nn(grid_stack).detach().cpu().numpy()
        # Mean, not sum: the per-trajectory losses are the same quantity
        # measured on different samples, and ``pinn_loss_mult`` scales it.
        loss_add = sum(losses) / len(losses)

        sctx = SolverContext(solution=solutions, loss_add=loss_add, g_fun_vals=g_fun_vals,
                             penalty_coeff=self.params['penalty_coeff'],
                             pinn_loss_mult=self.params['pinn_loss_mult'])
        sw_g, data_shape = self._build_fit_context()
        fit_ctx = FitContext(g_fun_vals=sw_g, data_shape=data_shape,
                             penalty_coeff=self.params['penalty_coeff'], for_rps=False)

        sum_err = 0.0
        for eq_idx, eq in enumerate(objective.vals):
            err = self.primary.compute(eq, eq_idx, sctx)
            if force_out_of_place:
                sum_err += err
                continue
            setattr(eq, self.primary.value_attr, err)
            setattr(eq, self.primary.flag_attr, True)
            for filler in self.objectives:
                if filler is self.primary:
                    continue
                eq.aic_calculated = True
                setattr(eq, filler.value_attr, filler.compute(eq, fit_ctx))
                setattr(eq, filler.flag_attr, True)
        if force_out_of_place:
            return sum_err

    def _apply_deepxde(self, objective, force_out_of_place):
        self.set_adapter(pretrained_net=self._pretrained_net())

        # Keep the 'deepxde' family option's config in sync with host params
        # (the legacy DeepXDEBasedFitness read these from self.params).
        if getattr(self.primary, 'metric', None) == 'deepxde':
            self.primary.error_metric = self.params.get('error_metric', 'rmse')
            self.primary.penalty_coeff = self.params.get('penalty_coeff', 0.2)

        samples = global_var.samples_manager
        grids = samples.grids()
        masks = samples.gFunc('m')

        if isinstance(objective, SoEq):
            eqs = [objective.vals[v] for v in objective.vars_to_describe]
        else:
            eqs = [objective]
        # ``evaluate`` returns a per-trajectory dict; the old code called
        # ``.reshape(-1)`` straight on it.
        targets = [eq.evaluate(active_only=True)[0] for eq in eqs]

        # One solve per trajectory: DeepXDE builds a single geometry from a
        # single grid, and the fillers index sctx by trajectory key.
        solutions, per_sample_data, losses = {}, {}, []
        try:
            for domain_key in samples.trajecatoryIDs:
                data_list = [np.asarray(target[domain_key]).reshape(-1)
                             for target in targets]
                solution_list, loss = self.adapter.solve(
                    equation_or_system=objective, grids=grids[domain_key],
                    data=data_list, domain_key=domain_key)
                if np.isnan(loss):
                    raise ValueError('NaN loss')
                flat_mask = np.asarray(masks[domain_key]).reshape(-1)
                solutions[domain_key] = [np.asarray(sol).reshape(-1)[flat_mask]
                                         for sol in solution_list]
                per_sample_data[domain_key] = data_list
                losses.append(float(loss))
        except Exception as exc:
            print(f'[SolverBasedFitness/deepxde] DeepXDE solve failed: {exc}')
            if force_out_of_place:
                return LOSS_NAN_VAL
            for eq in eqs:
                eq.fitness_value = LOSS_NAN_VAL
                eq.fitness_calculated = True
            return
        loss = float(np.mean(losses))

        sw_g, data_shape = self._build_fit_context()
        fit_ctx = FitContext(g_fun_vals=sw_g, data_shape=data_shape,
                             penalty_coeff=self.params.get('penalty_coeff', 0.2),
                             for_rps=False)
        # DeepXDEError reads sctx.solution[key][eq_idx] against
        # sctx.g_fun_vals[key][eq_idx] -- masked solution against the
        # inner-domain data, per trajectory.
        sctx = SolverContext(solution=solutions, loss_add=loss,
                             g_fun_vals=per_sample_data,
                             penalty_coeff=self.params.get('penalty_coeff', 0.2),
                             pinn_loss_mult=0.0)

        total_err = 0.0
        for eq_idx, eq in enumerate(eqs):
            err = self.primary.compute(eq, eq_idx, sctx)
            if force_out_of_place:
                total_err += err
                continue
            setattr(eq, self.primary.value_attr, err)
            setattr(eq, self.primary.flag_attr, True)
            for filler in self.objectives:
                if filler is self.primary:
                    continue
                setattr(eq, filler.value_attr, filler.compute(eq, fit_ctx))
                setattr(eq, filler.flag_attr, True)
        if force_out_of_place:
            return total_err / max(len(eqs), 1)

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'chromosome level', 'no suboperators', 'inplace'}


def plot_data_vs_solution(grid, data, solution):
    if grid.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(grid[:, 0].reshape(-1), grid[:, 1].reshape(-1),
                        solution.reshape(-1), cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()
        plt.close(fig)
    if grid.shape[1] == 1:
        fig = plt.figure()
        plt.scatter(grid.reshape(-1), solution.reshape(-1), color='r')
        plt.scatter(grid.reshape(-1), data.reshape(-1), color='k')
        plt.show()
        plt.close(fig)
    else:
        raise Exception('Infeasible dimensionality of the input dataset.')
