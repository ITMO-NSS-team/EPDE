#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:14:57 2021

@author: mike_ubuntu
"""

from dataclasses import dataclass
import copy
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
    
def init_verbose(plot_DE_solutions : bool = False, show_iter_idx : bool = True, 
                 show_iter_fitness : bool = False, show_iter_stats : bool = False, 
                 show_ann_loss : bool = False, show_warnings : bool = False):
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
                             show_iter_stats, show_ann_loss, show_warnings)

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
                best_state = copy.deepcopy(model.state_dict())
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
