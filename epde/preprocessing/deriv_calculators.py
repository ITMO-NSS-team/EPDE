#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:11:05 2022

@author: maslyaev
"""

import numpy as np

import multiprocessing as mp
from typing import Union

from epde.preprocessing.cheb import Process_Point_Cheb


def differentiate(data : np.ndarray, order : Union[int, list], 
                  mixed : bool = False, axis = None, *grids) -> list:
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

def adaptive_finite_difference(data : np.ndarray, grid : list, max_order : Union[int, list, tuple]) -> np.ndarray:
    grid_unique = [np.unique(ax_grid) for ax_grid in grid]

    derivs = differentiate(data, max_order, False, None, *grid_unique)
    derivs = np.vstack([der.reshape(-1) for der in derivs]).T
    return derivs


def polynomial_diff(data : np.ndarray, grid : list, max_order : Union[int, list, tuple], 
                    mp_poolsize : int, polynomial_window : int, poly_order : int):
    polynomial_boundary = polynomial_window//2 + 1
    index_array = []

    
    for idx, _ in np.ndenumerate(data):
        index_array.append((idx, data, grid, polynomial_window, max_order, polynomial_boundary, poly_order))
    print(len(index_array))  
    
    if mp_poolsize > 1:
        pool = mp.Pool(mp_poolsize)
        derivatives = pool.map_async(Process_Point_Cheb, index_array)
        pool.close()
        pool.join()
        derivatives = derivatives.get()
    else:
        derivatives = list(map(Process_Point_Cheb, index_array))
        
    return np.array(derivatives)