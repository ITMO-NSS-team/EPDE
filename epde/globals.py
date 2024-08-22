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
# device = torch.device('cpu') # TODO: make system-agnostic approach

from epde.cache.cache import Cache
from epde.cache.ctrl_cache import ControlNNContainer
from epde.supplementary import create_solution_net, AutogradDeriv


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
                     ctrl_args: list = [(0, [None,]),]):
    '''
    Use of bad practices, link control nn to the token family. 
    '''

    global control_nn
    control_nn = ControlNNContainer(output_num=n_control, args=ctrl_args, net = ann)


def reset_data_repr_nn(data: List[np.ndarray], grids: List[np.ndarray], train: bool = True,
                       derivs: List[Union[int, List, Union[np.ndarray]]] = None, 
                       penalised_derivs: List[Union[int, List]] = None,
                       epochs_max=1e5, predefined_ann: torch.nn.Sequential = None,
                       batch_frac=0.5, learining_rate=1e-4, device = 'cpu'): # loss_mean=1000, 
    '''
    Represent the data with ANN, suitable to be used as the initial guess of the candidate equations solutions 
    during the equation search, employing solver-based fitness function.

    Possible addition: add optimization in Sobolev space, using passed derivatives, incl. higher orders.
    '''

    global solution_guess_nn

    if predefined_ann is None:
        model = create_solution_net(equations_num=len(data), domain_dim=len(grids), device = device)
    else:
        model = predefined_ann

    if train:
        model.to(device)

        grids_tr = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grids])).float().T
        data_tr = torch.from_numpy(np.array([data_var.reshape(-1) for data_var in data])).float().T
        grids_tr = grids_tr.to(device)
        data_tr = data_tr.to(device)

        # print('device ', device)
        # print(f'grds {grids_tr.get_device()} and data {data_tr.get_device()}')

        batch_size = int(data[0].size * batch_frac)
        optimizer = torch.optim.Adam(model.parameters(), lr = learining_rate)
        deriv_calc = AutogradDeriv()

        t = 0
        min_loss = np.inf
        loss_mean = np.inf
        while loss_mean > 1e-6 and t < epochs_max:

            permutation = torch.randperm(grids_tr.size()[0])

            loss_list = []

            for i in range(0, grids_tr.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i+batch_size]
                batch_x, batch_y = grids_tr[indices], data_tr[indices]

                # print(f'batch_y {batch_y.get_device()}, batch_x {batch_x.get_device()},, {next(model.parameters()).is_cuda}')
                # print(f'model(batch_x) {model(batch_x)}') 
                loss = torch.mean(torch.abs(batch_y - model(batch_x)))
                if derivs is not None:
                    for var_idx, deriv_axes, deriv_tensor in derivs:
                        deriv_autograd = deriv_calc.take_derivative(model, batch_x, axes = deriv_axes, component = var_idx)
                        batch_derivs = torch.from_numpy(deriv_tensor)[indices].reshape_as(deriv_autograd).to(device)
                        
                        loss_add = 1e2 * torch.mean(torch.abs(batch_derivs - deriv_autograd))
                        # print(loss, loss_add)
                        loss += loss_add

                if penalised_derivs is not None:
                    for var_idx, deriv_axes in derivs:
                        deriv_autograd = deriv_calc.take_derivative(model, batch_x, axes = deriv_axes, component = var_idx)
                        batch_derivs = torch.from_numpy(deriv_tensor)[indices].reshape_as(deriv_autograd).to(device)
                        higher_ord_penalty = 1e3 * torch.mean(torch.abs(deriv_autograd))

                        loss += higher_ord_penalty

                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            loss_mean = np.mean(loss_list)
            if loss_mean < min_loss:
                best_model = model
                min_loss = loss_mean
            t += 1
        model = best_model
        print(f'min loss is {min_loss}, in last epoch: {loss_list}, ')
        solution_guess_nn = best_model
    else:
        solution_guess_nn = model