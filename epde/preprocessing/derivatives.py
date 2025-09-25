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
# device = torch.device('cpu')

import epde.globals as global_var
from epde.preprocessing.cheb import process_point_cheb
from epde.preprocessing.smoothing import smoothing
from epde.supplementary import define_derivatives, train_ann, use_ann_to_predict
    

def preprocess_derivatives_poly(field, grid=None, steps=None, data_name=None, output_file_name=None, smooth=True, sigma=9,
                                mp_poolsize=4, max_order=2, polynomial_window=9, poly_order=None, scaling=False, include_time = False):
    """
    Main preprocessing function for calculating derivatives on a uniform grid using polynomial fitting.
    
        This function prepares the input data for the equation discovery process by calculating derivatives
        using local polynomial approximations. This approach allows to estimate derivatives even from noisy data.
    
        Args:
            field (numpy.ndarray): The values of the field on a uniform grid. The dimensionality is unrestricted.
            grid (numpy.ndarray, optional): The grid coordinates. If None, it's assumed to be a unit grid. Defaults to None.
            steps (numpy.ndarray, optional): The grid steps in each dimension. If None, it's assumed to be a unit step. Defaults to None.
            data_name (str, optional): Name for saving the input field data. Defaults to None.
            output_file_name (str, optional): Name for saving the calculated derivatives. If None, the derivatives are returned. Defaults to None.
            smooth (bool, optional): Whether to apply Gaussian smoothing to the field. Defaults to True.
            sigma (int, optional): The standard deviation for Gaussian smoothing. Defaults to 9.
            mp_poolsize (int, optional): The number of workers for multiprocessing. Defaults to 4.
            max_order (int, optional): The maximum order of derivatives to calculate. Defaults to 2.
            polynomial_window (int, optional): The number of points for polynomial fitting. Must be odd. Defaults to 9.
            poly_order (int, optional): The order of the polynomial to fit. If None, it defaults to max_order. Defaults to None.
            scaling (bool, optional): Whether to scale the data. Defaults to False.
            include_time (bool, optional): Whether to include time dimension. Defaults to False.
    
        Returns:
            tuple (numpy.ndarray, numpy.ndarray): A tuple containing the (potentially smoothed) input field and the calculated derivatives.
            The derivatives are structured such that the first dimension represents the order and axis of differentiation.
    """
    t1 = datetime.datetime.now()

    polynomial_boundary = polynomial_window//2 + 1

    print('Executing on grid with uniform nodes:')
    if steps is None and grid is None:
        steps = np.ones(np.ndim(field))

    if smooth:
        field = smoothing(field, 'gaussian', sigma=sigma, include_time = include_time)
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
    """
    Initializes an artificial neural network (ANN) model.
    
        This method constructs a sequential ANN model with several linear layers
        and Tanh activation functions. The architecture is designed to map an
        input of a given dimension to a single output value. This ANN serves as a component
        in the equation discovery process, specifically for learning coefficients or
        terms within the differential equation. The network's structure is designed
        to provide a flexible function approximation for representing these unknown elements.
    
        Args:
            dim: The input dimension of the ANN model. This corresponds to the number of
                 variables the equation depends on.
    
        Returns:
            torch.nn.Sequential: The initialized ANN model.
    """
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
    """
    Computes numerical derivatives of the input data.
    
        Calculates derivatives of the input data using the gradient method to enable equation discovery.
        It can compute mixed derivatives and handle data with multiple dimensions, which is essential
        for identifying complex relationships in multi-dimensional systems.
    
        Args:
            data: The input data to differentiate.
            order: The order of the derivative to compute. It can be an integer
                or a list of integers specifying the order along each axis.
            mixed: A boolean indicating whether to compute mixed derivatives.
                Defaults to False.
            axis: The axis along which to compute the derivative. Defaults to None.
            *grids: Grid spacing(s) for numerical differentiation.
    
        Returns:
            A list of arrays representing the computed derivatives. These derivatives are
            used to construct candidate differential equations.
    """
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
    """
    Preprocesses field derivatives using an Artificial Neural Network (ANN).
        
        This method leverages an ANN to create a smooth, differentiable representation
        of the input field. This is particularly useful when the original field data
        is noisy or discrete, as it allows for more accurate and stable derivative
        calculations, which are essential for discovering underlying differential
        equations.
        
        Args:
            field (np.ndarray): The field data to approximate.
            grid (list or np.ndarray): The grid coordinates corresponding to the field data.
            max_order (int): The maximum order of derivatives to calculate.
            test_output (bool, optional): A flag for testing purposes. Defaults to False.
            epochs_max (float, optional): The maximum number of epochs for ANN training. Defaults to 1e3.
            loss_mean (int, optional): Threshold for loss function. Defaults to 1000.
            batch_frac (float, optional): Fraction of data used for batch training. Defaults to 0.5.
            return_ann (bool, optional): Whether to return the trained ANN model. Defaults to False.
        
        Returns:
            tuple: A tuple containing the ANN approximation of the field and the
                calculated derivatives. If `return_ann` is True, the tuple also
                includes the trained ANN model. Specifically, it returns:
                - approximation (np.ndarray): The ANN approximation of the field.
                - derivs (np.ndarray): The calculated derivatives of the approximation.
                - best_model (tf.keras.Model, optional): The trained ANN model,
                  returned only if `return_ann` is True.
    """
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
    """
    Preprocesses a derivative field using a specified method.
    
        This method applies a preprocessing technique to a given derivative
        field, preparing it for subsequent equation discovery. Different preprocessing
        methods can enhance the signal or reduce noise in the derivative data,
        improving the accuracy and efficiency of the equation search process.
    
        Args:
            field: The derivative field to preprocess.
            method: The name of the preprocessing method to apply.
            method_kwargs: Keyword arguments to pass to the preprocessing method.
    
        Returns:
            The preprocessed derivative field.
    """
    if method not in implemented_methods.keys():
        raise NotImplementedError(
            'Called preprocessing method has not been implemented yet. Use one of {implemented_methods}')
    return implemented_methods[method](field, **method_kwargs)
