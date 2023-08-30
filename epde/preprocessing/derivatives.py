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
from epde.preprocessing.cheb import process_point_cheb
from epde.preprocessing.smoothing import Smoothing
from epde.supplementary import define_derivatives, train_ann, use_ann_to_predict
    

def preprocess_derivatives_poly(field, grid=None, steps=None, data_name=None, output_file_name=None, smooth=True, sigma=9,
                                mp_poolsize=4, max_order=2, polynomial_window=9, poly_order=None, scaling=False):
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

    if smooth:
        field = Smoothing(field, 'gaussian', sigma=sigma)
    index_array = []

    if grid is None:
        dim_coords = []
        for dim in np.arange(np.ndim(field)):
            dim_coords.append(np.arange(0, field.shape[dim] * steps[dim], steps[dim]))

        grid = np.meshgrid(*dim_coords, indexing='ij')

    index_array = []

    for idx, _ in np.ndenumerate(field):
        index_array.append((idx, field, grid, polynomial_window,
                           max_order, polynomial_boundary, poly_order))
    print(len(index_array))

    if mp_poolsize > 1:
        pool = mp.Pool(mp_poolsize)
        derivatives = pool.map_async(process_point_cheb, index_array)
        pool.close()
        pool.join()
        derivatives = derivatives.get()
    else:
        derivatives = list(map(process_point_cheb, index_array))
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
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1024),
        torch.nn.Tanh(),
        torch.nn.Linear(1024, 512),
        torch.nn.Tanh(),        
        torch.nn.Linear(512, 1)
    )
    return model


def differentiate(data, order: Union[int, list], mixed: bool = False, axis=None, *grids):
    print('order', order)
    if isinstance(order, int):
        order = [order,] * data.ndim
    if any([ord_ax != order[0] for ord_ax in order]) and mixed:
        raise Exception(
            'Mixed derivatives can be taken only if the orders are same along all axes.')
    if data.ndim != len(grids) and (len(grids) != 1 or data.ndim != 2):
        print(data.ndim, len(grids))
        raise ValueError('Data dimensionality does not fit passed grids.')

    if len(grids) == 1 and data.ndim == 2:
        assert np.any(data.shape) == 1
        derivs = []
        dim_idx = 0 if data.shape[1] == 1 else 1
        grad = np.gradient(data, grids[0], axis=dim_idx)
        derivs.append(grad)
        ord_marker = order[dim_idx] if axis is None else order[axis]
        if ord_marker > 1:
            higher_ord_der_axis = None if mixed else dim_idx
            ord_reduced = [ord_ax - 1 for ord_ax in order]
            derivs.extend(differentiate(grad, ord_reduced,
                          mixed, higher_ord_der_axis, *grids))
    else:
        derivs = []
        if axis == None:
            for dim_idx in np.arange(data.ndim):
                print(data.shape, grids[dim_idx].shape)
                grad = np.gradient(data, grids[dim_idx], axis=dim_idx)
                derivs.append(grad)
                ord_marker = order[dim_idx] if axis is None else order[axis]
                if ord_marker > 1:
                    higher_ord_der_axis = None if mixed else dim_idx
                    ord_reduced = [ord_ax - 1 for ord_ax in order]
                    derivs.extend(differentiate(grad, ord_reduced,
                                  mixed, higher_ord_der_axis, *grids))
        else:
            grad = np.gradient(data, grids[axis], axis=axis)
            derivs.append(grad)
            if order[axis] > 1:
                ord_reduced = [ord_ax - 1 for ord_ax in order]
                derivs.extend(differentiate(grad, ord_reduced, False, axis, *grids))
    return derivs


def preprocess_derivatives_ANN(field, grid, max_order, test_output=False,
                               epochs_max=1e3, loss_mean=1000, batch_frac=0.5,
                               return_ann: bool = False):
    assert grid is not None, 'Grid needed for derivatives preprocessing with ANN'
    if isinstance(grid, np.ndarray):
        grid = [grid,]
    grid_unique = [np.unique(ax_grid) for ax_grid in grid]
    original_shape = field.shape

    best_model = train_ann(grids=grid, data=field, epochs_max=epochs_max)
    approximation = use_ann_to_predict(model=best_model, recalc_grids=grid)

    derivs = differentiate(approximation, max_order, False, None, *grid_unique)
    derivs = np.vstack([der.reshape(-1) for der in derivs]).T

    print(np.linalg.norm(approximation - field))
    print('shapes after & before', approximation.shape, original_shape)
    time.sleep(3)
    if return_ann:
        return approximation, derivs, best_model
    else:
        return approximation, derivs

implemented_methods = {'ANN': preprocess_derivatives_ANN,
                       'poly': preprocess_derivatives_poly}


def preprocess_derivatives(field, method, method_kwargs):
    if method not in implemented_methods.keys():
        raise NotImplementedError(
            'Called preprocessing method has not been implemented yet. Use one of {implemented_methods}')
    return implemented_methods[method](field, **method_kwargs)
