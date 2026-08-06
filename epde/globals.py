#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:14:57 2021

@author: mike_ubuntu
"""

from dataclasses import dataclass
import warnings
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
import epde.globals as global_var
# device = torch.device('cpu') # TODO: make system-agnostic approach

from epde.cache.cache import Cache
from epde.cache.ctrl_cache import ControlNNContainer
from epde.supplementary import create_solution_net, AutogradDeriv
from epde.preprocessing.smoothers import NN


# Gram-construction configuration, read by VWSRSparsity.apply,
# PhysicsInformedLasso.fit, EqRightPartSelector._precompute_super_gram,
# and L2LRFitness.apply. ``mode='vcoef'`` (default) uses the
# varying-coefficient stability estimator (``VaryingCoefSetup``);
# ``mode='axis'`` is the legacy axis-aligned sliding-window backup
# (``GramSetup`` reduced by the var/mu^2 CV in
# ``PhysicsInformedLasso.get_cv``).
gram_mode: str = 'vcoef'

# Which objective the SINGLE-objective optimizer minimises: 'discrepancy'
# (default, the residual fitness) or 'instability' (vcoef coefficient
# stability). The fitness host always computes discrepancy; when this is
# 'instability' it ALSO computes instability and the objective reader
# (SoEq.use_default_singleobjective_function) points the optimizer at it.
# Set via ``set_single_objective_metric`` before ``build_search``.
single_objective_metric: str = 'discrepancy'

# Per-rep seed for additive Gaussian noise applied at ``cfg.load_data()``;
# rewritten each rep so every rep sees an independent noise realization.
noise_seed = None

# ``gram_mode='vcoef'`` varying-coefficient stability config (see
# ``epde.operators.common.stability.VaryingCoefSetup``). ``vc_modes_cache`` resolves the
# per-axis basis resolution once per ``(grid_shape, main_var)`` from the
# Taylor microscale and reuses it for every candidate so the basis is
# identical across individuals; cleared on ``set_gram_config``. ``vc_k_max``
# caps modes per axis; ``vc_freq_coef`` scales the frequency ridge that
# suppresses noise leakage into the non-constant energy.
vc_modes_cache: dict = {}
vc_k_max: int = 6
vc_freq_coef: float = 0.0

# When True, ``VaryingCoefSetup._solve_gammas`` solves the mode block
# PER-FEATURE (block-diagonal in feature index) instead of jointly: cross-
# feature mode collinearity is dropped so a true constant-coefficient term's
# region-variation B (=nc_deb/C) is not inflated by collinear grid-modulated
# cousins (``x*u_xx``/``sin*u_xx`` sharing ``u_xx``'s mode energy, which pushed
# the weak true term's L1 threshold above its signal -> the ac t0/3 collapse).
# Extends the existing Frisch-Waugh constant-block decoupling to the modes.
# Default True; set False for the legacy joint mode solve.
vc_mode_decouple: bool = True

# When True, ``PhysicsInformedLasso.fit`` (the 'max_corr' anchor mode) anchors
# the L1 threshold to the WORKING RESIDUAL ``max_k|X_k^T r|`` (r = y minus the
# previous outer-iteration fit) instead of the RAW target ``max_k|X_k^T y|``.
# On the first pass r = y (full_coef_ = 0) so it matches the legacy anchor;
# thereafter the scale tracks what is still UNEXPLAINED as RFE shrinks the
# library, instead of staying pinned to ||y|| (which the dominant term inflates
# and which masks weak terms). No effect in 'tstat' mode (no max_corr there).
anchor_on_residual: bool = False

# Which estimator the INSTABILITY OBJECTIVE (the ``Instability`` filler /
# ``equation_terms_stability`` Pareto axis) uses. Decoupled from
# ``gram_mode``, which keeps governing the sparsity keep-rule unchanged.
#   None         -> 'chi2' (the default; before Aug 2026 it resolved from
#                   gram_mode: 'vcoef' -> 'vcoef', 'axis' -> 'cv').
#   'vcoef'      -> varying-coefficient NC/gamma_0^2 (the pre-chi2 default).
#   'cv'         -> axis-aligned sliding-window CV (var/mu^2).
#   'survival'   -> block-resampled coefficient survival
#                   (sign-flip rate + MAD/|median| across refits).
#   'tile'       -> per-tile refits, between-tile dispersion MAD/|median|
#                   (basis-free spatial inhomogeneity).
#   'het'        -> calibrated heterogeneity: per-block refits, score =
#                   tau^2/(tau^2 + mean^2) with tau^2 the DerSimonian-Laird
#                   excess variance (between-block variance MINUS what the
#                   blocks' own standard errors predict) -- cv2 with the
#                   sampling part subtracted; bounded, singularity-free.
#   'chi2'       -> per-term coefficient constancy from the cumulative
#                   score path (Nyblom-Hansen): Theta from ONE global OLS,
#                   never re-estimated; the running score
#                   S_j = cumsum(w_i X_ij r_i) propagates once along EACH
#                   grid axis, pinned to zero at both ends by the normal
#                   equations, and its bulge is measured against the
#                   term's own fitted-signal energy (never the residual).
#                   No refits, cheapest estimator. THE DEFAULT: on the
#                   14-system clean panels it ties vcoef/cv on both the
#                   non-domination and strict lenses at zero refit cost.
# Set via ``set_instability_metric`` before ``build_search``.
instability_metric = None

# RPS amplified-identity guard: during the right-part term-sweep, a candidate
# target whose winning fit has amplification ratio
#   A = sum_j |c_j| * ||col_j|| / ||target col||   (nonzero terms + intercept)
# above this cap is DECLINED -- the parasitic ``Lambda * (near-null identity
# combination) = target`` shape, where huge mutually-cancelling coefficients
# stretch the residual of a VALID analytical identity (e.g. the LV sum
# identity du+dv = alpha*u - gamma*v) to imitate an unrelated target out of
# derivative noise. Evidence base (truth-anchor sweep, 14 equations incl.
# real data): true forms sit at A in [1.0, 6.65]; observed parasites at
# ~7e2..1.6e6 -- the cap of 100 leaves 15x headroom above the worst truth
# (a genuine stiff balance still passes) and 6x below the mildest parasite.
# Identity-form refits (du = -dv + alpha*u - gamma*v, A ~ 2) are UNAFFECTED:
# only the amplified-cancellation shape is declined, so valid identities stay
# credited. None disables the guard.
rps_amplification_cap = 100.0


def set_gram_config(mode: str = 'vcoef'):
    """Override the global Gram-construction mode before ``build_search``.

    Used by ``projects/thesis/thesis_runner.py`` / ``profile_loop_stats.py``
    to switch between the varying-coefficient default (``'vcoef'``) and the
    axis-aligned sliding-window backup (``'axis'``) via a single CLI flag.
    """
    global gram_mode
    if mode not in ('axis', 'vcoef'):
        raise ValueError(
            f'gram_mode must be "axis" or "vcoef"; got {mode!r}')
    gram_mode = mode
    # Stale per-axis basis resolution from a prior CLI/config must not bleed
    # into a new invocation -- the source data or grid_shape may have changed.
    vc_modes_cache.clear()


def set_single_objective_metric(metric: str = 'discrepancy'):
    """Override the single-objective optimizer's objective before
    ``build_search``. Mirrors ``set_gram_config``: a process-level global
    read at population construction by
    ``SoEq.use_default_singleobjective_function`` and the single-objective
    director's fitness assembly.
    """
    global single_objective_metric
    if metric not in ('discrepancy', 'instability'):
        raise ValueError(
            f'single_objective_metric must be "discrepancy" or "instability"; got {metric!r}')
    single_objective_metric = metric


def set_anchor_on_residual(flag: bool = False):
    """Override whether the 'max_corr' anchor uses the working residual
    (``max|X^T r|``) instead of the raw target (``max|X^T y|``), before
    ``build_search``.
    """
    global anchor_on_residual
    anchor_on_residual = bool(flag)


def set_instability_metric(metric=None):
    """Override the instability-objective estimator before ``build_search``.

    Mirrors ``set_gram_config``: a process-level global read by the
    ``Instability`` filler in ``epde.operators.common.objectives``. ``None``
    (default) resolves to ``'chi2'``; see :data:`instability_metric` for
    the estimator menu. The sparsity keep-rule keeps following
    ``gram_mode`` regardless.

    ``'chi'`` is accepted as an alias and normalised to ``'chi2'`` HERE, at
    set time -- ``Instability.compute`` dispatches on canonical names, and
    an un-normalised ``'chi'`` would silently fall through to the ``'cv'``
    branch.
    """
    global instability_metric
    if metric == 'chi':
        metric = 'chi2'
    valid = (None, 'vcoef', 'cv', 'survival', 'tile', 'het', 'chi2')
    if metric not in valid:
        raise ValueError(
            f"instability_metric must be one of {valid} (or the alias "
            f"'chi' for 'chi2'); got {metric!r}")
    instability_metric = metric


def set_rps_amplification_cap(cap=100.0):
    """Override the RPS amplified-identity guard before ``build_search``.

    ``cap`` is the maximum admissible amplification ratio ``A`` of a
    candidate right-part fit (see :data:`rps_amplification_cap`); ``None``
    disables the guard. Mirrors ``set_gram_config``: a process-level global
    read by ``EqRightPartSelector`` during the term-sweep.
    """
    global rps_amplification_cap
    if cap is not None:
        cap = float(cap)
        if not np.isfinite(cap) or cap <= 1.0:
            raise ValueError(
                'rps_amplification_cap must be a finite value > 1 (true '
                f'forms reach A ~ 6.7) or None to disable; got {cap!r}')
    rps_amplification_cap = cap


def resolve_instability_metric() -> str:
    """The effective instability-objective estimator: the explicit
    ``instability_metric`` override if set, else ``'chi2'`` (the default;
    'vcoef' / 'cv' stay available as explicit selections, and the sparsity
    keep-rule keeps following ``gram_mode`` regardless)."""
    if instability_metric is not None:
        return instability_metric
    return 'chi2'


# Which residual metric the DISCREPANCY OBJECTIVE family (the
# ``Discrepancy`` filler) uses when no internal per-instance override is
# given. This is THE search-level configuration point: ``EpdeSearch``
# writes it from its ``discrepancy_metric`` kwarg at construction, and the
# filler resolves it at compute time -- objectives BUILD FROM the search
# configuration instead of having metric strings threaded through the
# directors.
#   None              -> 'wape' (the interface-wide default).
#   'wape'            -> sum|resid| / sum|target| (L1 relative).
#   'l2'              -> raw g-weighted ||resid||_2 (the legacy metric).
#   'l2_relative'     -> ||resid||_2 / ||target||_2
#                        (aliases: 'l2_scaled', 'l2_rel', 'residual').
#   'scale_invariant' -> pointwise cancellation residual
#                        (aliases: 'scale_inv', 'sinv', 'cancellation').
# The solver-side options ('solver_l2' / 'pic' / 'deepxde') are NOT part of
# this menu: they are implied by the host backend (use_solver /
# solver_backend) and wired internally by the strategy. ``use_pic`` plays no
# role here -- it ONLY selects the second objective (instability in place of
# the baseline complexity).
# Set via ``set_discrepancy_metric`` (aliases normalised at set time).
discrepancy_metric = None

_DISCREPANCY_MENU = (None, 'wape', 'l2', 'l2_relative', 'scale_invariant')
_DISCREPANCY_ALIASES = {'l2_scaled': 'l2_relative', 'l2_rel': 'l2_relative',
                        'residual': 'l2_relative',
                        'scale_inv': 'scale_invariant',
                        'sinv': 'scale_invariant',
                        'cancellation': 'scale_invariant'}


def set_discrepancy_metric(metric=None):
    """Override the discrepancy-family metric before ``build_search``.

    Mirrors ``set_instability_metric``: a process-level global read at
    compute time by the ``Discrepancy`` filler. Aliases are normalised to
    canonical names HERE; unknown names raise. ``None`` (default) resolves
    to ``'wape'``. Normally written by ``EpdeSearch.__init__`` from its
    ``discrepancy_metric`` kwarg -- the last-constructed search wins, the
    standing process-global trade-off (same as ``gram_mode``).
    """
    global discrepancy_metric
    metric = _DISCREPANCY_ALIASES.get(metric, metric)
    if metric not in _DISCREPANCY_MENU:
        raise ValueError(
            f'discrepancy_metric must be one of {_DISCREPANCY_MENU} '
            f'(or aliases {tuple(_DISCREPANCY_ALIASES)}); got {metric!r}')
    discrepancy_metric = metric


def resolve_discrepancy_metric() -> str:
    """The effective discrepancy-family metric: the explicit override if
    set, else ``'wape'`` (the interface-wide default)."""
    return discrepancy_metric if discrepancy_metric is not None else 'wape'


# Which reader the COMPLEXITY OBJECTIVE family (the ``Complexity`` filler /
# the ``equation_complexity`` Pareto reader) uses when no per-instance
# override is given.
#   None      -> 'factors' (exact backward compatibility, the default).
#   'factors' -> legacy factor-count (the ``equation_complexity_by_factors``
#                semantics: 0.5 per non-derivative factor, derivative order
#                per derivative factor, over the target term and every
#                non-zero-weight term) -- bit-compatible with all existing
#                legacy-pipeline artifacts, known quirks included.
#   'terms'   -> active-term count: number of non-zero non-target
#                ``weights_internal`` slots + 1 when the fitted intercept is
#                non-zero. UNIFORM across the LASSO / VWSR sparsity pairings.
# Set via ``set_complexity_metric`` before ``build_search``.
complexity_metric = None

# Which objective occupies MOEA/D's SECOND Pareto axis (the first is always
# the discrepancy fitness; the front is strictly 2-axis).
#   None          -> derive from ``use_pic`` (True -> 'instability',
#                    False -> 'complexity') -- exact backward compatibility.
#   'instability' -> the ``Instability`` filler + ``equation_terms_stability``.
#   'complexity'  -> the ``Complexity`` filler + ``equation_complexity``.
# Consumed in lockstep at three sites: strategy filler assembly, SoEq axis
# registration and the MOEA/D ideal point. Set via ``set_second_objective``
# BEFORE ``EpdeSearch`` construction (fillers are assembled in __init__).
second_objective = None


def set_complexity_metric(metric=None):
    """Override the complexity-objective reader before ``build_search``.

    Mirrors ``set_instability_metric``: a process-level global read by the
    ``Complexity`` filler (when constructed without a per-instance override)
    and by the ``equation_complexity`` reader's lazy fallback. ``None``
    (default) resolves to ``'factors'`` for exact backward compatibility;
    see :data:`complexity_metric` for the menu.
    """
    global complexity_metric
    valid = (None, 'factors', 'terms')
    if metric not in valid:
        raise ValueError(
            f'complexity_metric must be one of {valid}; got {metric!r}')
    complexity_metric = metric


def resolve_complexity_metric() -> str:
    """The effective complexity-objective reader: the explicit override if
    set, else ``'factors'`` (the legacy semantics)."""
    return complexity_metric if complexity_metric is not None else 'factors'


def set_second_objective(objective=None):
    """Override which objective occupies MOEA/D's second Pareto axis;
    overrides the ``use_pic``-derived default when set.

    Mirrors ``set_instability_metric``; see :data:`second_objective`. Must
    be called BEFORE ``EpdeSearch`` construction -- the fitness fillers are
    assembled in ``__init__``, and a later change desyncs the computed
    filler set from the registered axis readers (which then fail loudly on
    their ``*_calculated`` asserts).
    """
    global second_objective
    valid = (None, 'instability', 'complexity')
    if objective not in valid:
        raise ValueError(
            f'second_objective must be one of {valid}; got {objective!r}')
    second_objective = objective


def resolve_second_objective(use_pic: bool) -> str:
    """The effective second Pareto axis: the explicit ``second_objective``
    override if set, else the one ``use_pic`` implies (True ->
    'instability', False -> 'complexity')."""
    if second_objective is not None:
        return second_objective
    return 'instability' if use_pic else 'complexity'


def init_caches(set_grids: bool = False, device = 'cpu'):
    """
    Initialization global variables for keeping input data, values of grid and useful tensors such as evaluated terms

    Args:
        set_grids (`bool`): flag about using grid data

    Returns:
        None
    """
    global tensor_cache, grid_cache, initial_data_cache
    tensor_cache = Cache(device = device)
    initial_data_cache = Cache(device = device)
    if set_grids:
        grid_cache = Cache(device = device)
    else:
        grid_cache = None


def set_time_axis(axis: int):
    """
    Setting global of time axis
    """
    global time_axis
    time_axis = axis


def init_eq_search_operator(operator):
    global eq_search_operator
    eq_search_operator = operator


def init_sys_search_operator(operator):
    global sys_search_operator
    sys_search_operator = operator


def delete_cache():
    global tensor_cache, grid_cache
    try:
        del tensor_cache
    except NameError:
        print('Failed to delete tensor cache due to its inexistance')
    try:
        del grid_cache
    except NameError:
        print('Failed to delete grid cache due to its inexistance')


class TrainHistory(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.history = []
        self._idx = 0

    def add(self, element):
        self.history.append((self._idx, element))
        self._idx += 1

def reset_hist():
    global history
    history = TrainHistory()

@dataclass
class VerboseManager:
    """
    Manager for output in text form
    """
    plot_DE_solutions : bool
    show_iter_idx : bool
    iter_fitness : bool
    iter_stats : bool
    show_ann_loss : bool
    show_warnings : bool
    candidate_objectives : bool
    
def init_verbose(plot_DE_solutions : bool = False, show_iter_idx : bool = True,
                 show_iter_fitness : bool = False, show_iter_stats : bool = False,
                 show_ann_loss : bool = False, show_warnings : bool = False,
                 candidate_objectives : bool = True):
    """
    Method for initialized of manager for output in text form

    Args:
        plot_DE_solutions (`bool`): optional 
            display solutions of a differential equation, default - False
        show_iter_idx (`bool`): optional
            display the index of each iteration EA, default - False
        show_iter_fitness (`bool`): optional
            display the fitness of each iteration EA, default - False
        show_iter_stats (`bool`): optional
            display statistical properties of the population in each iteration EA, default - False
        show_warnings (`bool`): optional
            display warnings arising during the operation of the algorithm, default - False
    """
    global verbose
    if not show_warnings:
        warnings.filterwarnings("ignore")
    verbose = VerboseManager(plot_DE_solutions, show_iter_idx, show_iter_fitness,
                             show_iter_stats, show_ann_loss, show_warnings,
                             candidate_objectives)

def reset_control_nn(n_control: int = 1, ann: torch.nn.Sequential = None, 
                     ctrl_args: list = [(0, [None,]),], device: str = 'cpu'):
    '''
    Use of bad practices, link control nn to the token family. 
    '''

    global control_nn
    control_nn = ControlNNContainer(output_num = n_control, args = ctrl_args,
                                    net = ann, device = device)


def reset_data_repr_nn(data: List[np.ndarray], grids: List[np.ndarray], train: bool = True,
                       derivs: List[Union[int, List, Union[np.ndarray]]] = None,
                       penalised_derivs: List[Union[int, List]] = None,
                       epochs_max=1e3, predefined_ann: torch.nn.Sequential = None,
                       batch_frac=0.5, val_frac=0.2, learning_rate=1e-4, device='cpu',
                       use_fourier: bool = True, fourier_params: dict = None,
                       deriv_weight=1, penalty_weight=1e3):
    '''
    Represent the data with ANN, suitable to be used as the initial guess of the candidate equations solutions
    during the equation search, employing solver-based fitness function.

    Possible addition: add optimization in Sobolev space, using passed derivatives, incl. higher orders.
    '''

    if fourier_params is None:
        fourier_params = {'L': [4,], 'M': [3,]}

    global solution_guess_nn

    if predefined_ann is None:
        model = create_solution_net(equations_num=len(data), domain_dim=len(grids), device=device,
                                    use_fourier=use_fourier, fourier_params=fourier_params)
        # model = NN(Num_Hidden_Layers=5, Neurons_Per_Layer=50, Input_Dim=len(grids), Activation_Function='Rational')


    else:
        model = predefined_ann

    if train:
        model = model.to(device)

        grids_tr = torch.from_numpy(np.array([subgrid[global_var.grid_cache.g_func != 0].reshape(-1) for subgrid in grids])).float().T
        data_tr = torch.from_numpy(np.array([data_var[global_var.grid_cache.g_func != 0].reshape(-1) for data_var in data])).float().T
        grids_tr = grids_tr.to(device)
        data_tr = data_tr.to(device)

        n_total = grids_tr.size()[0]
        n_val = max(1, int(n_total * val_frac))
        perm_split = torch.randperm(n_total)
        val_indices = perm_split[:n_val]
        train_indices = perm_split[n_val:]

        grids_val, data_val = grids_tr[val_indices], data_tr[val_indices]
        grids_train, data_train = grids_tr[train_indices], data_tr[train_indices]

        batch_size = int(data_train.size()[0] * batch_frac)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        epochs_max = int(epochs_max)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, epochs_max // 20), factor=0.5)
        deriv_calc = AutogradDeriv()

        best_state = None
        min_val_loss = np.inf
        val_patience = max(1, epochs_max // 10)
        val_no_improve = 0

        print(f'Training ANN to represent input data on {epochs_max} epochs:')
        for t in range(epochs_max):
            permutation = torch.randperm(grids_train.size()[0])

            loss_list = []

            for i in range(0, grids_train.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i+batch_size]
                batch_x = grids_train[indices].detach().requires_grad_(True)
                batch_y = data_train[indices]

                pred = model(batch_x)
                loss = F.mse_loss(pred, batch_y) / (torch.mean(batch_y ** 2) + 1e-8)
                # if derivs is not None:
                #     deriv_loss = 0
                #     for var_idx, deriv_axes, deriv_tensor in derivs:
                #         deriv_autograd = deriv_calc.take_derivative(model, batch_x, axes=deriv_axes, component=var_idx)
                #         flat_indices = train_indices[indices]
                #         batch_derivs = torch.from_numpy(deriv_tensor.reshape(-1))[flat_indices].reshape_as(deriv_autograd).float().to(device)
                #         deriv_loss += deriv_weight * F.mse_loss(deriv_autograd, batch_derivs) / (torch.mean(batch_derivs ** 2) + 1e-8)
                #
                # loss += deriv_loss / len(derivs)

                if penalised_derivs is not None:
                    for var_idx, deriv_axes in penalised_derivs:
                        deriv_autograd = deriv_calc.take_derivative(model, batch_x, axes=deriv_axes, component=var_idx)
                        loss += penalty_weight * torch.mean(torch.abs(deriv_autograd))

                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            train_loss_mean = np.mean(loss_list)

            with torch.no_grad():
                val_loss = (F.mse_loss(model(grids_val), data_val) / (torch.mean(data_val ** 2) + 1e-8)).item()

            # if derivs is not None:
            #     grids_val_grad = grids_val.detach().requires_grad_(True)
            #     deriv_loss = 0
            #     for var_idx, deriv_axes, deriv_tensor in derivs:
            #         deriv_autograd = deriv_calc.take_derivative(model, grids_val_grad, axes=deriv_axes, component=var_idx)
            #         batch_derivs = torch.from_numpy(deriv_tensor.reshape(-1))[val_indices].reshape_as(deriv_autograd).float().to(device)
            #         deriv_loss += deriv_weight * F.mse_loss(deriv_autograd, batch_derivs).item() / (torch.mean(batch_derivs ** 2).item() + 1e-8)
            #
            #     val_loss += deriv_loss / len(derivs)

            scheduler.step(val_loss)

            if val_loss < min_val_loss:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                min_val_loss = val_loss
                val_no_improve = 0
            else:
                val_no_improve += 1

            if t % 100 == 0 and t != 0:
                print(f"Epoch {t:4d} | Train Loss: {train_loss_mean:.6e} | Val Loss: {val_loss:.6e}")

            if val_no_improve >= val_patience:
                print(f"Early stopping at epoch {t}, best val loss: {min_val_loss:.6e}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        print(f'Best val loss: {min_val_loss:.6e}, final train loss: {train_loss_mean:.6e}')
        solution_guess_nn = model
    else:
        solution_guess_nn = model
