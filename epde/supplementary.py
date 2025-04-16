#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:33:34 2020

@author: mike_ubuntu
"""

from abc import ABC
from typing import Callable, Union

import numpy as np
from functools import reduce
import copy
import torch
# device = torch.device('cpu')

import matplotlib.pyplot as plt

from epde.solver.data import Domain
from epde.solver.models import Fourier_embedding, mat_model


class BasicDeriv(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Trying to create abstract differentiation method')
    
    def take_derivative(self, u: torch.Tensor, args: torch.Tensor, axes: list):
        raise NotImplementedError('Trying to differentiate with abstract differentiation method')


class AutogradDeriv(BasicDeriv):
    def __init__(self):
        pass

    def take_derivative(self, u: Union[torch.nn.Sequential, torch.Tensor], args: torch.Tensor, 
                        axes: list = [], component: int = 0):
        if not args.requires_grad:
            args.requires_grad = True
        if axes == [None,]:
            return u(args)[..., component].reshape(-1, 1)
        if isinstance(u, torch.nn.Sequential):
            comp_sum = u(args)[..., component].sum(dim = 0)
        elif isinstance(u, torch.Tensor):
            raise TypeError('Autograd shall have torch.nn.Sequential as its inputs.')
        else:
            print(f'u.shape, {u.shape}')
            comp_sum = u.sum(dim = 0)
        for axis in axes:
            output_vals = torch.autograd.grad(outputs = comp_sum, inputs = args, create_graph=True)[0]
            comp_sum = output_vals[:, axis].sum()
        output_vals = output_vals[:, axes[-1]].reshape(-1, 1)
        return output_vals

class FDDeriv(BasicDeriv):
    def __init__(self):
        pass

    def take_derivative(self, u: np.ndarray, args: np.ndarray, 
                        axes: list = [], component: int = 0):
        
        if not isinstance(args, torch.Tensor):
            args = args.detach().cpu().numpy()

        output_vals = u[..., component].reshape(args.shape)
        if axes == [None,]:
            return output_vals
        for axis in axes:
            output_vals = np.gradient(output_vals, args.reshape(-1)[1] - args.reshape(-1)[0], axis = axis, edge_order=2)  
        return output_vals

def create_solution_net(equations_num: int, domain_dim: int, use_fourier = True, #  mode: str, domain: Domain 
                        fourier_params: dict = None, device = 'cpu'):
    '''
    fft_params have to be passed as dict with entries like: {'L' : [4,], 'M' : [3,]}
    '''
    L_default, M_default = 4, 10
    if use_fourier:
        if fourier_params is None:
            if domain_dim == 1:
                fourier_params = {'L' : [L_default],
                              'M' : [M_default]}
            else:
                fourier_params = {'L' : [L_default] + [None,] * (domain_dim - 1), 
                              'M' : [M_default] + [None,] * (domain_dim - 1)}
        fourier_params['device'] = device
        four_emb = Fourier_embedding(**fourier_params)
        if device == 'cuda':
            four_emb = four_emb.cuda()
        net_default = torch.nn.ModuleList([four_emb,])
    else:
        net_default = torch.nn.ModuleList([])
    linear_inputs = net_default[0].out_features if use_fourier else domain_dim
    
    if domain_dim == 1:            
        hidden_neurons = 128 # 64 #
    else:
        hidden_neurons = 112 # 54 #

    operators = net_default + torch.nn.ModuleList([torch.nn.Linear(linear_inputs, hidden_neurons, device=device),
                               torch.nn.Tanh(),
                               torch.nn.Linear(hidden_neurons, hidden_neurons, device=device),
                               torch.nn.Tanh(),
                               torch.nn.Linear(hidden_neurons, equations_num, device=device)])
    return torch.nn.Sequential(*operators)

def exp_form(a, sign_num: int = 4):
    if np.isclose(a, 0):
        return 0.0, 0
    exp = np.floor(np.log10(np.abs(a)))
    return np.around(a / 10**exp, sign_num), int(exp)


def rts(value, sign_num: int = 5):
    """
    Round to a ``sign_num`` of significant digits.
    """
    if value == 0:
        return 0
    magn_top = np.log10(value)
    idx = -(np.sign(magn_top)*np.ceil(np.abs(magn_top)) - sign_num)
    if idx - sign_num > 1:
        idx -= 1
    return np.around(value, int(idx))


def train_ann(args: list, data: np.ndarray, epochs_max: int = 500, batch_frac = 0.5, 
              dim = None, model = None, device = 'cpu'):
    if dim is None:
        dim = 1 if np.any([s == 1 for s in data.shape]) and data.ndim == 2 else data.ndim
    # assert len(args) == dim, 'Dimensionality of data does not match with passed grids.'
    data_size = data.size
    if model is None:
        model = torch.nn.Sequential(
                                    torch.nn.Linear(dim, 256, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(256, 256, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(256, 64, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(64, 1024, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(1024, 1, device=device)
                                    )
    
    model.to(device)
    data_grid = np.stack([arg.reshape(-1) for arg in args])
    grid_tensor = torch.from_numpy(data_grid).float().T.to(device)
    # grid_tensor.to(device)
    data = torch.from_numpy(data.reshape(-1, 1)).float().to(device)
    # print(data.size)
    # data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    batch_size = int(data_size * batch_frac)

    t = 0

    print('grid_flattened.shape', grid_tensor.shape, 'field.shape', data.shape)

    loss_mean = 1000
    min_loss = np.inf
    losses = []
    while loss_mean > 2e-3 and t < epochs_max:

        permutation = torch.randperm(grid_tensor.size()[0])

        loss_list = []

        for i in range(0, grid_tensor.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = grid_tensor[indices], data[indices]
            loss = torch.mean(torch.abs(batch_y-model(batch_x)))

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_mean = np.mean(loss_list)
        if loss_mean < min_loss:
            best_model = model
            min_loss = loss_mean
        losses.append(loss_mean)
        # if global_var.verbose.show_ann_loss:
        #     print('Surface training t={}, loss={}'.format(t, loss_mean))
        t += 1
    print_loss = True
    if print_loss:
        plt.plot(losses)
        plt.grid()
        plt.show()
    return best_model

def use_ann_to_predict(model, recalc_grids: list):
    data_grid = np.stack([grid.reshape(-1) for grid in recalc_grids])
    recalc_grid_tensor = torch.from_numpy(data_grid).float().T
    recalc_grid_tensor = recalc_grid_tensor #.to(device)

    return model(recalc_grid_tensor).detach().numpy().reshape(recalc_grids[0].shape)

def flatten(obj):
    '''
    Method to flatten list, passed as ``obj`` - the function parameter.
    '''
    assert type(obj) == list

    for idx, elem in enumerate(obj):
        if not isinstance(elem, (list, tuple)):
            obj[idx] = [elem,]
    return reduce(lambda x, y: x+y, obj)

def factor_params_to_str(factor, set_default_power=False, power_idx=0):
    param_label = np.copy(factor.params)
    if set_default_power:
        param_label[power_idx] = 1.
    return (factor.label, tuple(param_label))

def form_label(x, y):
    print(type(x), type(y.cache_label))
    return x + ' * ' + y.cache_label if len(x) > 0 else x + y.cache_label

def detect_similar_terms(base_equation_1, base_equation_2):   # Переделать!
    same_terms_from_eq1 = []
    same_terms_from_eq2 = []
    eq2_processed = np.full(
        shape=len(base_equation_2.structure), fill_value=False)

    similar_terms_from_eq1 = []
    similar_terms_from_eq2 = []

    different_terms_from_eq1 = []
    different_terms_from_eq2 = []
    for eq1_term in base_equation_1.structure:
        found_similar = False
        for idx, eq2_term in enumerate(base_equation_2.structure):
            if eq1_term == eq2_term and not eq2_processed[idx]:
                found_similar = True
                same_terms_from_eq1.append(eq1_term)
                same_terms_from_eq2.append(eq2_term)
                eq2_processed[idx] = True
                break
            elif ({token.label for token in eq1_term.structure} == {token.label for token in eq2_term.structure} and
                  len(eq1_term.structure) == len(eq2_term.structure) and not eq2_processed[idx]):
                found_similar = True
                similar_terms_from_eq1.append(eq1_term)
                similar_terms_from_eq2.append(eq2_term)
                eq2_processed[idx] = True
                break
        if not found_similar:
            different_terms_from_eq1.append(eq1_term)

    for idx, elem in enumerate(eq2_processed):
        if not elem:
            different_terms_from_eq2.append(base_equation_2.structure[idx])

    assert len(same_terms_from_eq1) + len(similar_terms_from_eq1) + \
        len(different_terms_from_eq1) == len(base_equation_1.structure)
    assert len(same_terms_from_eq2) + len(similar_terms_from_eq2) + \
        len(different_terms_from_eq2) == len(base_equation_2.structure)
    return [same_terms_from_eq1, similar_terms_from_eq1, different_terms_from_eq1], [same_terms_from_eq2, similar_terms_from_eq2, different_terms_from_eq2]


def filter_powers(gene):
    gene_filtered = []

    for token_idx in range(len(gene)):
        total_power = sum([factor.param(name = 'power') for factor in gene 
                           if gene[token_idx].partial_equlaity(factor)])#gene.count(gene[token_idx])
        powered_token = copy.deepcopy(gene[token_idx])
        
        power_idx = np.inf
        for param_idx, param_info in powered_token.params_description.items():
            if param_info['name'] == 'power':
                max_power = param_info['bounds'][1]
                power_idx = param_idx
                break
        powered_token.params[power_idx] = total_power if total_power < max_power else max_power
        if powered_token not in gene_filtered:
            gene_filtered.append(powered_token)
    return gene_filtered


def define_derivatives(var_name='u', dimensionality=1, max_order=2):
    """
    Method for generating derivative keys

    Args:
        var_name (`str`): name of input data dependent variable
        dimensionality (`int`): dimensionallity of data
        max_order (`int`|`list`): max order of delivative
    
    Returns:
        deriv_names (`list` with `str` values): keys for epde
        var_deriv_orders (`list` with `int` values): keys for enter to solver
    """
    deriv_names = []
    var_deriv_orders = []
    if isinstance(max_order, int):
        max_order = [max_order for dim in range(dimensionality)]
    for var_idx in range(dimensionality):
        for order in range(max_order[var_idx]):
            var_deriv_orders.append([var_idx,] * (order+1))
            if order == 0:
                deriv_names.append('d' + var_name + '/dx' + str(var_idx))
            else:
                deriv_names.append(
                    'd^'+str(order+1) + var_name + '/dx'+str(var_idx)+'^'+str(order+1))
    print('Deriv orders after definition', var_deriv_orders)
    return deriv_names, var_deriv_orders


def population_sort(input_population):
    individ_fitvals = [
        individual.fitness_value if individual.fitness_calculated else 0 for individual in input_population]
    pop_sorted = [x for x, _ in sorted(
        zip(input_population, individ_fitvals), key=lambda pair: pair[1])]
    return list(reversed(pop_sorted))


def normalize_ts(Input):
    matrix = np.copy(Input)
    if np.ndim(matrix) == 0:
        raise ValueError(
            
            'Incorrect input to the normalizaton: the data has 0 dimensions')
    elif np.ndim(matrix) == 1:
        return matrix
    else:
        for i in np.arange(matrix.shape[0]):
            std = np.std(matrix[i])
            if std != 0:
                matrix[i] = (matrix[i] - np.mean(matrix[i])) / std
            else:
                matrix[i] = 1
        return matrix
