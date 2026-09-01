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
# import epde.globals as global_var
# device = torch.device('cpu') # TODO: make system-agnostic approach

from epde.cache.cache_refactored import Cache
from epde.structure.domain import TrajectoriesManager

from epde.cache.ctrl_cache import ControlNNContainer
from epde.supplementary import create_solution_net, AutogradDeriv
from epde.preprocessing.smoothers import NN


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
# This module holds the objects that exist WHILE A SEARCH RUNS -- the caches,
# the trajectory manager, the verbose manager, the training history and the
# networks. It holds no settings.
#
# It used to hold both. Seven objective settings lived here as mutable module
# scalars, each with a ``set_*`` writer and a ``resolve_*`` reader, duplicating
# the declaration, the validation, the documentation and the default already
# carried by ``epde/interface/parameters/default_search_config.json``. Settings
# are now read from the resolved configuration
# (``epde.interface.search_config.active_config()``), which ``EpdeSearch``
# publishes at construction; the setters are gone, and so is the risk of the
# two sources disagreeing.
#
# ``vc_modes_cache`` below is the one survivor of that group, because it is a
# cache and not a setting. The vcoef estimator's actual knobs (``k_max``,
# ``freq_coef``, ``mode_decouple``) now live on ``VaryingCoefSetup``.

# Basis resolution per ``(grid_shape, main_var)`` for the varying-coefficient
# stability estimator, resolved once from the Taylor microscale and reused for
# every candidate so the basis is identical across individuals. Cleared by
# ``init_caches``: a stale per-axis resolution from a previous search must not
# bleed into a new one, since the source data or grid shape may both differ.
vc_modes_cache: dict = {}


def init_caches(set_grids: bool = False):
    """
    Initialization global variables for keeping input data, values of grid and useful tensors such as evaluated terms

    The caches are numpy-backed and therefore live on the CPU. They used to
    take the search's ``device``, which made ``EpdeSearch(device='cuda')``
    unconstructible: ``Cache.__init__`` rejects the cuda/numpy pairing
    outright. The GPU is only ever used by the solver, so the device is a
    ``solver`` config setting and reaches ``SystemSolverInterface`` instead.
    ``Cache`` keeps its own ``device``/``backend`` parameters for the cupy
    work sketched at cache_refactored.py:47.

    Args:
        set_grids (`bool`): flag about using grid data

    Returns:
        None
    """
    global tensor_cache, grid_cache, initial_data_cache, samples_manager

    samples_manager = TrajectoriesManager()
    tensor_cache = Cache()
    initial_data_cache = Cache()
    grid_cache = Cache() if set_grids else None
    # Per-search state, reset with the caches it is derived from.
    vc_modes_cache.clear()


# def init_eq_search_operator(operator):
#     global eq_search_operator
#     eq_search_operator = operator


# def init_sys_search_operator(operator):
#     global sys_search_operator
#     sys_search_operator = operator


def release_tensor_cache():
    """Drop the evaluated-term tensors, keeping the cache object itself.

    The memory-budget properties set by ``set_memory_properties`` survive, so a
    later evaluation refills the same configured cache.
    """
    if globals().get('tensor_cache') is not None:
        tensor_cache.initMemory()


def delete_cache():
    """Release every cached tensor: evaluated terms, grids, samples and the
    initial data.

    Rebinding to empty containers rather than ``del``-ing the names. The old
    version deleted ``tensor_cache`` and ``grid_cache`` outright -- which left
    the ~40 unguarded ``global_var.tensor_cache.…`` readers raising
    ``AttributeError`` rather than degrading -- and it missed
    ``samples_manager`` and ``initial_data_cache`` entirely, which is where the
    trajectory tensors (the actual bulk) are held. Nothing called it, so the
    breakage was latent.
    """
    global samples_manager

    for cache in (globals().get('tensor_cache'),
                  globals().get('grid_cache'),
                  globals().get('initial_data_cache')):
        if cache is not None:
            cache.initMemory()
    if globals().get('samples_manager') is not None:
        samples_manager = TrajectoriesManager()
    vc_modes_cache.clear()


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
    
class EPDEDeprecationWarning(DeprecationWarning):
    """An EPDE call that still works but is on its way out.

    A dedicated category so that ``init_verbose``'s blanket suppression can
    keep the framework's own migration diagnostics visible: they are addressed
    to the person writing the script, not to the search.
    """


class EPDEUsageWarning(UserWarning):
    """An argument EPDE accepted and will not act on."""


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
        # ...but never EPDE's own API diagnostics. The line above is
        # process-global, permanent and category-blind, so simply CONSTRUCTING
        # an EpdeSearch used to silence every later warning in the process --
        # including the deprecation notices that tell a script author their
        # call form is going away, which is the one audience that cannot act
        # on a warning it never sees. Registered after the ignore, hence ahead
        # of it in the filter list.
        for category in (EPDEDeprecationWarning, EPDEUsageWarning):
            warnings.filterwarnings("default", category=category)
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

        grids_tr = torch.from_numpy(np.array([subgrid[grid_cache.g_func != 0].reshape(-1) for subgrid in grids])).float().T
        data_tr = torch.from_numpy(np.array([data_var[grid_cache.g_func != 0].reshape(-1) for data_var in data])).float().T
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
