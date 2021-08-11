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

from epde.prep.cheb import Process_Point_Cheb
from epde.prep.smoothing import Smoothing


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
    

def Preprocess_derivatives(field, grid = None, steps = None, data_name = None, output_file_name = None, smooth = True, sigma = 9,
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
    
#    raise TabError
    #np.save('ssh_field.npy', field)   
    if type(data_name) != type(None):
        np.save(data_name, field)
    if type(output_file_name) != type(None):
        if not '.npy' in output_file_name:
            output_file_name += '.npy'        
        np.save(output_file_name, derivatives)
    return field, np.array(derivatives)        
