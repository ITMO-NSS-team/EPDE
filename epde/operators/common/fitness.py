#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:20:59 2021

@author: mike_ubuntu

Fitness host operators. Each owns ONE responsibility: run the shared
scaffolding (sparsity -> coefficient fit -> context) and delegate every
scalar objective to a pluggable *filler* from
``epde.operators.common.objectives``.

* :class:`SolverFreeFitness` -- gene-level; hosts solver-free fillers
  (``L2Discrepancy`` / ``WAPEDiscrepancy`` / ``Instability`` / ...).
  Replaces the former ``L2Fitness`` and ``L2LRFitness``.
* :class:`SolverBasedFitness` -- chromosome-level; solves the system once
  via a PDE backend (autograd ``SolverAdapter`` or DeepXDE) and hosts
  solver-based fillers (``SolverL2Discrepancy`` / ``PICError`` /
  ``DeepXDEError``) plus an optional ``Instability`` r-loss. Replaces the
  former ``SolverBasedFitness``, ``PIC`` and ``DeepXDEBasedFitness``.

The metric logic lives in the fillers (single responsibility); the hosts
only orchestrate the fit and write each filler's value to its attribute.
"""
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
# Re-exported so ``from epde.operators.common.fitness import
# vc_stability_total_lr`` keeps working for external callers
# (e.g. projects/thesis/_vc_cache_gate.py).
from epde.operators.common.stability import (calculate_weights, vc_stability_total_lr)  # noqa: F401
from epde.operators.common.objectives import (
    FitContext, SolverContext, EquationObjective, SolverObjective,
    L2Discrepancy, WAPEDiscrepancy, Instability,
    SolverL2Discrepancy, PICError, DeepXDEError, LOSS_NAN_VAL,
)
from epde import _loop_stats


class SolverFreeFitness(CompoundOperator):
    """Solver-free fitness host (gene level).

    Runs ``sparsity`` (when the primary filler asks) and ``coeff_calc``,
    then evaluates each configured objective filler and stores its scalar
    on the equation. The ``primary`` filler (a discrepancy) drives the
    right-part-selection scaffolding and is the value returned for
    ``EqRightPartSelector``'s ``force_out_of_place`` term-sweep.

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
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)

        penalty_coeff = self.params['penalty_coeff']
        if not (penalty_coeff > 0. and penalty_coeff < 1.):
            raise ValueError('Incorrect penalty coefficient set, value shall be in (0, 1).')

        primary = self.primary
        # Sparsity is run when the primary (discrepancy) filler asks for it:
        # always during the RPS sweep, and as a fallback in-place when the
        # equation lacks a valid weights_internal state (see L2Discrepancy /
        # WAPEDiscrepancy.needs_sparsity, lifted from the old operators).
        if primary.needs_sparsity(objective, force_out_of_place):
            self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
            # During the RPS term-sweep a degenerate (all-zero-weight)
            # candidate is skipped by returning None; in-place we always
            # fall through to a finite value.
            if force_out_of_place and primary.is_degenerate(objective):
                return None
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])

        try:
            g_fun_vals = global_var.grid_cache.g_func[
                global_var.grid_cache.g_func_mask].reshape(-1)
        except AttributeError:
            g_fun_vals = None
        try:
            data_shape = global_var.grid_cache.inner_shape
        except AttributeError:
            data_shape = None
        ctx = FitContext(g_fun_vals=g_fun_vals, data_shape=data_shape,
                         penalty_coeff=penalty_coeff, for_rps=force_out_of_place)

        if force_out_of_place:
            return primary.compute(objective, ctx)

        # In-place finalization: weights_internal has selected the sparse
        # support, so physically prune the now-dead zero-weight terms before
        # objectives are scored and the structure is stored/rendered -- the
        # standing in-place counterpart to the remove_zero_terms that
        # EqRightPartSelector runs after its own sweep. Applied
        # UNCONDITIONALLY (not just on the sparsity fallback): an equation can
        # reach the front carrying weights_internal zeros that were never
        # pruned (a survivor whose support tightened without a following
        # prune), which otherwise desyncs text_form (it indexes the compacted
        # weights_final by full-structure position) -- the spurious
        # cross-equation "leak" render artefact. Safe now that
        # remove_zero_terms compacts weights_internal in lockstep; scoring is
        # unaffected (the fillers read weights_final / evaluate()). The RPS
        # sweep returned above, so this never runs out-of-place (which must
        # keep every candidate term).
        if getattr(objective, 'weights_internal_evald', False):
            objective.remove_zero_terms()

        for filler in self.objectives:
            setattr(objective, filler.value_attr, filler.compute(objective, ctx))
            setattr(objective, filler.flag_attr, True)
            if filler.compute(objective, ctx) == 0:
                print()
        # AIC is not produced by the solver-free path; expose the default
        # the legacy WAPE operator set so downstream readers don't assert.
        objective.aic = None
        objective.aic_calculated = True

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'gene level', 'contains suboperators', 'inplace'}


class SolverBasedFitness(CompoundOperator):
    """Solver-based fitness host (chromosome level).

    Solves the candidate system once with a PDE backend, then scores each
    equation with the configured solver fillers. Subsumes the former
    ``SolverBasedFitness`` (``backend='autograd'``, ``masked=False``,
    ``SolverL2Discrepancy``), ``PIC`` (``backend='autograd'``,
    ``masked=True``, ``PICError`` + ``Instability``) and
    ``DeepXDEBasedFitness`` (``backend='deepxde'``, ``DeepXDEError`` +
    ``Instability``).

    The right-part-selection term-sweep never solves: the director wires a
    lightweight :class:`SolverFreeFitness` as the RPS fitness instead.
    """
    key = 'SolverBasedFitness'

    def __init__(self, param_keys: list, objectives: list = None,
                 primary: SolverObjective = None, stability: Instability = None,
                 backend: str = 'autograd', masked: bool = False):
        super().__init__(param_keys)
        self.adapter = None
        self.backend = backend
        self.masked = masked
        self.objectives = list(objectives) if objectives else []
        self.primary = primary if primary is not None else (
            self.objectives[0] if self.objectives else None)
        # Optional solver-free r-loss filler (instability) reused as a
        # second objective, mirroring PIC / DeepXDE.
        self.stability = stability

    def set_adapter(self, net=None, pretrained_net=None):
        if self.backend == 'deepxde':
            if self.adapter is None:
                from epde.integrate.deepxde_integration import DeepXDEAdapter
                cfg = self.params.get('deepxde_config', {})
                self.adapter = DeepXDEAdapter(pretrained_net=pretrained_net, **cfg)
            return
        if self.adapter is None or net is not None:
            compiling_params = {'mode': 'autograd', 'tol': 0.01, 'lambda_bound': 100}
            optimizer_params = {}
            training_params = {'epochs': 1e3, 'info_string_every': 1e3}
            early_stopping_params = {'patience': 4, 'no_improvement_patience': 250}
            explicit_cpu = False
            device = 'cuda' if (torch.cuda.is_available and not explicit_cpu) else 'cpu'
            self.adapter = SolverAdapter(net=net, use_cache=False, device=device)
            self.adapter.set_compiling_params(**compiling_params)
            self.adapter.set_optimizer_params(**optimizer_params)
            self.adapter.set_early_stopping_params(**early_stopping_params)
            self.adapter.set_training_params(**training_params)

    def apply(self, objective: SoEq, arguments: dict, force_out_of_place: bool = False):
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)
        if force_out_of_place:
            self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])

        if self.backend == 'deepxde':
            return self._apply_deepxde(objective, force_out_of_place)
        return self._apply_autograd(objective, force_out_of_place)

    def _build_fit_context(self):
        try:
            g_fun_vals = global_var.grid_cache.g_func[
                global_var.grid_cache.g_func_mask].reshape(-1)
        except AttributeError:
            g_fun_vals = None
        try:
            data_shape = global_var.grid_cache.inner_shape
        except AttributeError:
            data_shape = None
        return g_fun_vals, data_shape

    def _apply_autograd(self, objective, force_out_of_place):
        try:
            net = deepcopy(global_var.solution_guess_nn)
        except NameError:
            net = None
        self.set_adapter(net=net)

        print('solving equation:')
        print(objective.text_form)
        loss_add, solution_nn = self.adapter.solve_epde_system(
            system=objective, grids=None, boundary_conditions=None, use_fourier=True)

        _, grids = global_var.grid_cache.get_all(mode='torch')
        if self.masked:
            g_mask = global_var.grid_cache.g_func_mask
            grids = [grid[g_mask] for grid in grids]
            g_fun_vals = global_var.grid_cache.g_func[g_mask]
        else:
            g_fun_vals = global_var.grid_cache.g_func
        grids = torch.stack([grid.reshape(-1) for grid in grids], dim=1).float()
        solution = solution_nn(grids).detach().cpu().numpy()

        sctx = SolverContext(solution=solution, loss_add=loss_add, g_fun_vals=g_fun_vals,
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
            if self.stability is not None:
                eq.aic_calculated = True
                setattr(eq, self.stability.value_attr, self.stability.compute(eq, fit_ctx))
                setattr(eq, self.stability.flag_attr, True)
        if force_out_of_place:
            return sum_err

    def _apply_deepxde(self, objective, force_out_of_place):
        try:
            pretrained_net = deepcopy(global_var.solution_guess_nn)
        except Exception:
            pretrained_net = None
        self.set_adapter(pretrained_net=pretrained_net)

        # Keep the DeepXDEError filler's config in sync with host params
        # (the legacy DeepXDEBasedFitness read these from self.params).
        if isinstance(self.primary, DeepXDEError):
            self.primary.error_metric = self.params.get('error_metric', 'rmse')
            self.primary.penalty_coeff = self.params.get('penalty_coeff', 0.2)

        keys, grids = global_var.grid_cache.get_all(mode='numpy')
        mask_flat = global_var.grid_cache.g_func_mask.flatten()

        if isinstance(objective, SoEq):
            eqs = [objective.vals[v] for v in objective.vars_to_describe]
        else:
            eqs = [objective]
        data_list = []
        for eq in eqs:
            _, target, _ = eq.evaluate(normalize=False, return_val=False)
            data_list.append(target.reshape(-1))

        try:
            solution_list, loss = self.adapter.solve(
                equation_or_system=objective, grids=grids, data=data_list)
            if np.isnan(loss):
                raise ValueError('NaN loss')
        except Exception as exc:
            print(f'[SolverBasedFitness/deepxde] DeepXDE solve failed: {exc}')
            if force_out_of_place:
                return LOSS_NAN_VAL
            for eq in eqs:
                eq.fitness_value = LOSS_NAN_VAL
                eq.fitness_calculated = True
            return

        sw_g, data_shape = self._build_fit_context()
        fit_ctx = FitContext(g_fun_vals=sw_g, data_shape=data_shape,
                             penalty_coeff=self.params.get('penalty_coeff', 0.2),
                             for_rps=False)
        # Pack per-eq masked (solution, data) for DeepXDEError.
        masked_solutions = [solution_list[i][mask_flat] for i in range(len(eqs))]
        masked_data = [data_list[i] for i in range(len(eqs))]
        sctx = SolverContext(solution=masked_solutions, loss_add=loss,
                             g_fun_vals=masked_data,
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
            if self.stability is not None:
                setattr(eq, self.stability.value_attr, self.stability.compute(eq, fit_ctx))
                setattr(eq, self.stability.flag_attr, True)
        if force_out_of_place:
            return total_err / max(len(eqs), 1)

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'chromosome level', 'contains suboperators', 'inplace'}


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
