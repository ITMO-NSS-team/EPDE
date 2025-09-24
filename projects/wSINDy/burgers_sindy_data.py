#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:53:27 2023

@author: maslyaev
"""

import numpy as np
import pandas as pd

import torch
import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

import matplotlib.pyplot as plt
import matplotlib

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import lotka

import pysindy as ps

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens
from epde.interface.solver_integration import BoundaryConditions, BOPElement

def epde_discovery(x, t, u, use_ann = False): #(grids, data, use_ann = False):
    """
    Performs EPDE to find governing equations from data.
        
        This method sets up and executes the EPDE algorithm to identify the underlying equation that best describes the provided data. 
        It involves preprocessing, defining custom tokens, and fitting the EPDE search object to the data.
        The method aims to find a balance between model complexity and accuracy by searching for the most concise equation that adequately represents the data.
        
        Args:
            x: Spatial grid.
            t: Temporal grid.
            u: Data corresponding to the spatial and temporal grids.
            use_ann: Flag indicating whether to use an artificial neural network (ANN) preprocessor. Defaults to False.
        
        Returns:
            epde_search_obj: The fitted EPDE search object containing the discovered equations.
    """
    u = u.T
    grids = np.meshgrid(t, x, indexing = 'ij')
    print(u.shape, grids[0].shape, grids[1].shape)
    multiobjective_mode = True
    dimensionality = u.ndim - 1
    
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                          dimensionality = dimensionality, boundary = 10,
                                          coordinate_tensors = grids)    
    
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True
                                          preprocessor_kwargs={'epochs_max' : 2})
    popsize = 7
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=50)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize, 
                                                   training_epochs=50)
    # epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=50)
    
    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           token_labels = ['t', 'x'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)        
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    opt_val = 1e-1
    bounds = (1e-9, 1e0) if multiobjective_mode else (opt_val, opt_val)    
    print(u.shape, grids[0].shape)
    epde_search_obj.fit(data=u, variable_names=['u',], max_deriv_order=(2, 2),
                        equation_terms_max_number=5, data_fun_pow = 1, additional_tokens=[trig_tokens, custom_grid_tokens], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds)
    return epde_search_obj 


def sindy_examplar_search(x, t, u):
    """
    Performs SINDy exemplar search to identify governing equations.
    
        This method applies the SINDy algorithm to discover the underlying equations from the provided data.
        It leverages finite differences to estimate derivatives, constructs a library of potential terms
        using a PDE library, and then employs sparse regression to identify the most significant terms.
        This approach automates the identification of governing differential equations from data,
        allowing users to gain insights into the dynamics of the system.
    
        Args:
            x (np.ndarray): Spatial coordinates.
            t (np.ndarray): Time coordinates.
            u (np.ndarray): Input data representing the system's state.
    
        Returns:
            None. The method prints the identified model to the console.
    """
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    
    u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)
    
    u = u.reshape(len(x), len(t), 1)
    u_dot = u_dot.reshape(len(x), len(t), 1)
    
    print(u.shape, u_dot.shape)
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=3,
        spatial_grid=x,
        is_uniform=True,
    )
    
    print('STLSQ model: ')
    optimizer = ps.STLSQ(threshold=2, alpha=1e-5, normalize_columns=True)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    model.print()
        

if __name__ == '__main__':
    path = '/home/maslyaev/epde/EPDE_main/projects/wSINDy/data/burgers/'

    try:
        u_file = os.path.join(os.path.dirname( __file__ ), 'data/burgers/burgers.mat')
        # data = loadmat(u_file)
    except (FileNotFoundError, OSError):
        u_file = '/home/maslyaev/epde/EPDE_main/projects/wSINDy/data/burgers/burgers.mat'
    print(u_file)
    data = loadmat(u_file)

    # data = loadmat('burgers.mat')
    t = np.ravel(data['t'])
    x = np.ravel(data['x'])
    u = np.real(data['usol'])
    dt = t[1] - t[0]
    dx = x[1] - x[0]

    epde = True
    
    if epde:
        models = []
        for i in range(10):
            models.append(epde_discovery(x, t, u))
    else:
        model = sindy_examplar_search(x, t, u)#, d_t)