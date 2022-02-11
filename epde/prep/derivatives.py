#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:41:16 2020

@author: mike_ubuntu
"""
import time
import numpy as np
import datetime
import multiprocessing as mp
from typing import Union

import torch
device = torch.device('cpu')

import epde.globals as global_var
from epde.prep.cheb import Process_Point_Cheb
from epde.prep.smoothing import Smoothing
from epde.supplementary import Define_Derivatives

# from TEDEouS.solver import apply_operator_set
# from TEDEouS.input_preprocessing import grid_prepare, operator_prepare

def scaling_test(field, steps = None, ff_name = None, output_file_name = None, smooth = True, sigma = 9,
                           mp_poolsize = 4, max_order = 2, polynomial_window = 9, poly_order = None, boundary = 1):
    assert field.ndim == 2, 'Test condition of 2D input field was not fullfilled'
    _, derivs_raw = Preprocess_derivatives(field, steps, ff_name = ff_name, output_file_name = output_file_name,
                       smooth=smooth, sigma = sigma, mp_poolsize=mp_poolsize, max_order = 1, polynomial_window=polynomial_window, poly_order=poly_order)
#    return derivs_fa
    derivs_raw = derivs_raw.reshape((int(np.sqrt(derivs_raw.shape[0])), int(np.sqrt(derivs_raw.shape[0])), derivs_raw.shape[1]))
    derivs_raw = derivs_raw[boundary:-boundary, boundary:-boundary]
    new_coords = np.empty_like(steps)
    for dim_idx in np.arange(new_coords.size):
        new_coords[dim_idx] = np.linalg.norm(derivs_raw[..., dim_idx])**(-1) * np.linalg.norm(derivs_raw[..., 0])
        print(dim_idx, new_coords[dim_idx], np.linalg.norm(derivs_raw[..., dim_idx]))    
    time.sleep(10)
    steps = np.array(steps) / new_coords
    print('new steps:', steps)
    time.sleep(5)
    
    _, derivs_scaled = Preprocess_derivatives(field, steps, ff_name = None, output_file_name = output_file_name,
                       smooth=smooth, sigma = sigma, mp_poolsize=mp_poolsize, max_order = max_order, polynomial_window=polynomial_window, poly_order=poly_order)    
    derivs_scaled = derivs_scaled.reshape((int(np.sqrt(derivs_scaled.shape[0])), int(np.sqrt(derivs_scaled.shape[0])), derivs_scaled.shape[1]))
    derivs_scaled = derivs_scaled[boundary:-boundary, boundary:-boundary]
    
    return derivs_raw, derivs_scaled
    

def Preprocess_derivatives_poly(field, grid = None, steps = None, data_name = None, output_file_name = None, smooth = True, sigma = 9,
                           mp_poolsize = 4, max_order = 2, polynomial_window = 9, poly_order = None, scaling = False):
    '''
    
    Main preprocessing function for the calculation of derivatives on uniform grid
    
    Parameters (old):
    ---------
    
    field : numpy.ndarray
        The values of studied field on uniform grid. The dimensionality of the tensor is not restricted;
        
    output_file_name : string, optional
        Name of the file, in which the tensors of caluclated derivatives will be saved; if it is not given, function returns the tensor
        
    mp_poolsize : integer, optional
        The number of workers for multiprocessing.pool, that would be created for derivative evaluation;
        
    max_order : integer, optional
        The maximum order of the derivatives to be calculated;
        
    polynomial_window : integer, optional
        The number of points, for which the polynmial will be fitted, in order to later analytically differentiate it and obtain the derivatives. 
        Shall be defined with odd number or if it is even, expect polynomial_window + 1 - number of points to be used.
    
    Returns:
    --------

    derivatives : np.ndarray
        If the output file name is not defined, or set as None, - tensor of derivatives, where the first dimentsion is the order 
        and the axis of derivative in such manner, that at first, all derivatives for first axis are returned, secondly, all 
        derivatives for the second axis and so on. The next dimensions match the dimensions of input field.
    
    '''
    t1 = datetime.datetime.now()

    polynomial_boundary = polynomial_window//2 + 1

    print('Executing on grid with uniform nodes:')
    if steps is None and grid is None:
        steps = np.ones(np.ndim(field))

    if smooth: field = Smoothing(field, 'gaussian', sigma = sigma)
    index_array = []

    if grid is None: 
        dim_coords = []
        for dim in np.arange(np.ndim(field)):
            dim_coords.append(np.arange(0, field.shape[dim] * steps[dim], steps[dim]))

        grid = np.meshgrid(*dim_coords, indexing = 'ij') 

    index_array = []

    
    for idx, _ in np.ndenumerate(field):
        index_array.append((idx, field, grid, polynomial_window, max_order, polynomial_boundary, poly_order))
    print(len(index_array))  
    
    if mp_poolsize > 1:
        pool = mp.Pool(mp_poolsize)
        derivatives = pool.map_async(Process_Point_Cheb, index_array)
        pool.close()
        pool.join()
        derivatives = derivatives.get()
    else:
        derivatives = list(map(Process_Point_Cheb, index_array))
    t2 = datetime.datetime.now()


    print('Start:', t1, '; Finish:', t2)
    print('Preprocessing runtime:', t2 - t1)
    
    if type(data_name) != type(None):
        np.save(data_name, field)
    if type(output_file_name) != type(None):
        if not '.npy' in output_file_name:
            output_file_name += '.npy'        
        np.save(output_file_name, derivatives)
    return field, np.array(derivatives)        

def init_ann(dim):
    # dim = global_var.tensor_cache.get(label = None).ndim
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 64),
        torch.nn.Tanh(),       
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1024),
        torch.nn.Tanh(),
        torch.nn.Linear(1024, 1)
    )
    return model

def differentiate(data, order : Union[int, list], mixed : bool = False, axis = None, *grids):
    print('order', order)
    if isinstance(order, int):
        order = [order,] * data.ndim
    if any([ord_ax != order[0] for ord_ax in order]) and mixed:
        raise Exception('Mixed derivatives can be taken only if the orders are same along all axes.')
    if data.ndim != len(grids) and (len(grids) != 1 or data.ndim != 2):
        print(data.ndim, len(grids))
        raise ValueError('Data dimensionality does not fit passed grids.')
        
    if len(grids) == 1 and data.ndim == 2:
        assert np.any(data.shape) == 1
        derivs = []
        dim_idx = 0 if data.shape[1] == 1 else 1
        grad = np.gradient(data, grids[0], axis = dim_idx)
        derivs.append(grad)
        ord_marker = order[dim_idx] if axis is None else order[axis]
        if ord_marker > 1:
            higher_ord_der_axis = None if mixed else dim_idx
            ord_reduced = [ord_ax - 1 for ord_ax in order]
            derivs.extend(differentiate(grad, ord_reduced, mixed, higher_ord_der_axis, *grids))
    else:
        derivs = []
        if axis == None:
            for dim_idx in np.arange(data.ndim):
                print(data.shape, grids[dim_idx].shape)
                grad = np.gradient(data, grids[dim_idx], axis = dim_idx)
                derivs.append(grad)
                ord_marker = order[dim_idx] if axis is None else order[axis]
                if ord_marker > 1:
                    higher_ord_der_axis = None if mixed else dim_idx
                    ord_reduced = [ord_ax - 1 for ord_ax in order]
                    derivs.extend(differentiate(grad, ord_reduced, mixed, higher_ord_der_axis, *grids))
        else:
            grad = np.gradient(data, grids[axis], axis = axis)
            derivs.append(grad)
            if order[axis] > 1:
                    ord_reduced = [ord_ax - 1 for ord_ax in order]
                    derivs.extend(differentiate(grad, ord_reduced, False, axis, *grids))
    return derivs

def Preprocess_derivatives_ANN(field, grid, max_order, test_output = False, 
                               epochs_max = 1e3, loss_mean = 1000):
    assert grid is not None, 'Grid needed for derivatives preprocessing with ANN'
    grid_unique = [np.unique(ax_grid) for ax_grid in grid]
    
    dim = 1 if np.any([s == 1 for s in field.shape]) and field.ndim == 2 else field.ndim
    print('dim', dim, field.shape, np.any([s == 1 for s in field.shape]), field.ndim == 2)
    model = init_ann(dim)
    grid_flattened = [subgrid.reshape(-1) for subgrid in grid]
    grid_flattened=torch.from_numpy(np.array(grid_flattened)).float().T
    dimensionality = field.ndim
    original_shape = field.shape
    test_output = True
    
    field = torch.from_numpy(field.reshape(-1, 1)).float()
    grid_flattened.to(device)
    field.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    batch_size = 128 # or whatever
    
    t=0 

    print('grid_flattened.shape', grid_flattened.shape, 'field.shape', field.shape)
    
    # loss_mean=1000
    min_loss=np.inf
    while loss_mean>1e-5 and t<epochs_max:
    
        # X is a torch Variable
        permutation = torch.randperm(grid_flattened.size()[0])
        
        loss_list=[]
        
        for i in range(0,grid_flattened.size()[0], batch_size):
            optimizer.zero_grad()
    
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = grid_flattened[indices], field[indices]
    
            # in case you wanted a semi-full example
            # outputs = model.forward(batch_x)
            loss = torch.mean(torch.abs(batch_y-model(batch_x)))
    
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_mean=np.mean(loss_list)
        if loss_mean<min_loss:
            best_model=model
            min_loss=loss_mean
        print('Surface trainig t={}, loss={}'.format(t,loss_mean))
        t+=1
    
    approximation = best_model(grid_flattened).detach().numpy().reshape(original_shape)
    print('shapes after & before', approximation.shape, original_shape)
    time.sleep(3)
    derivs = differentiate(approximation, max_order, False, None, *grid_unique)
    derivs = np.vstack([der.reshape(-1) for der in derivs]).T
    return approximation, derivs
    
    # derivs = []
    # _, operators = Define_Derivatives(dimensionality=dimensionality, max_order = max_order)
    # for operator in operators[1:]:
    #     temp_operator = [[1, operator, 1]]
    #     operator = operator_prepare(temp_operator, prepared_grid, subset=None, true_grid=grid, h=0.3)
    #     op_clean = apply_operator_set(best_model, operator)
    #     if test_output:
    #         derivs.append(op_clean)
    #     else:
    #         derivs.append(op_clean.detach().numpy().reshape(-1))
    # if test_output:
    #     return field, derivs
    # else:
        # return field, np.vstack(derivs).T

implemented_methods = {'ANN' : Preprocess_derivatives_ANN, 'poly' : Preprocess_derivatives_poly}

def Preprocess_derivatives(field, method, method_kwargs):
    if method not in implemented_methods.keys():
        raise NotImplementedError('Called preprocessing method has not been implemented yet. Use one of {implemented_methods}')
    return implemented_methods[method](field, **method_kwargs)
