#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:53:27 2023

@author: maslyaev
"""

import numpy as np
import pandas as pd
from functools import reduce

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

def epde_discovery(grids, data, derivs, use_ann = False):
    """
    Performs equation discovery using the EPDE algorithm.
        
        This method orchestrates the search for governing equations by configuring and executing an EPDE search. 
        It leverages the provided data and derivative information to explore the space of possible equations, 
        aiming to identify the equation structures that best represent the underlying dynamics of the system. 
        The method configures the search space, optimization parameters, and token library before executing the search.
        
        Args:
            grids: Spatial and temporal grid coordinates.
            data: The data to fit the equation to.
            derivs: Precomputed derivatives of the data.
            use_ann: Flag to use artificial neural networks. Defaults to False.
        
        Returns:
            EpdeSearch: The fitted EPDE search object containing the discovered equations.
    """
    multiobjective_mode = True
    dimensionality = data.ndim - 1
    
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                          dimensionality = dimensionality, boundary = 10,
                                          coordinate_tensors = grids)    
    # epde_search_obj.set_preprocessor(default_preprocessor_type='spectral', # use_smoothing = True
    #                                  preprocessor_kwargs={})
    popsize = 8
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=35)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize, 
                                                   training_epochs=35)
    # epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=50)
    
    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           token_labels = ['t', 'x'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)        
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    opt_val = 1e-1
    bounds = (1e-10, 1e0) if multiobjective_mode else (opt_val, opt_val)    
    epde_search_obj.fit(data=data, variable_names=['u',], max_deriv_order=(2, 1),
                        equation_terms_max_number=4, data_fun_pow = 1, additional_tokens=[trig_tokens, ],  # custom_grid_tokens
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds)
    # sys = epde_search_obj.get_equations_by_complexity(2)
    return epde_search_obj #, sys[0] #, epde_search_obj.saved_derivaties 
    

def sindy_discovery(grids, u):#, u_dot):
    """
    Performs sparse identification of differential equations (SINDy) to discover governing equations from data.
        
        This method takes spatial and temporal grids and corresponding data 'u',
        constructs a library of candidate functions using `PDELibrary`, and then
        uses SINDy to identify a sparse model that describes the dynamics of the data.
        It automates the process of identifying governing differential equations from data.
        
        Args:
            grids (tuple): A tuple containing the temporal and spatial grids.
            u (numpy.ndarray): The data to be modeled.
        
        Returns:
            sindy.SINDy: The fitted SINDy model, representing the discovered differential equation.
    """
    t = np.unique(grids[0])
    x = np.unique(grids[1])    
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    
    u = u.T.reshape(len(x), len(t), 1)
    
    # задаем свои токены через лямбда выражения
    library_functions = [lambda x: x, lambda x: x * x]#, lambda x: np.cos(x)*np.cos(x)]#, lambda x: 1/x]
    library_function_names = [lambda x: x, lambda x: x + x]#, lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')']#, lambda x: '1/'+x]
    
    # ? проблема с использованием multiindices
    # multiindices=np.array([[0,1],[1,1],[2,0],[3,0]])
    
    pde_lib = ps.PDELibrary(library_functions=library_functions,
                            function_names=library_function_names,
                            derivative_order=3, spatial_grid=x,
                            # multiindices=multiindices,
                            implicit_terms=True, temporal_grid=t,
                            include_bias=True, is_uniform=True, include_interaction=True)
    feature_library = ps.feature_library.PolynomialLibrary(degree=3)
    optimizer = ps.SR3(threshold=0, max_iter=10000, tol=1e-15, nu=1e2,
                       thresholder='l0', normalize_columns=True)
    
    # optimizer = ps.STLSQ(threshold=50, alpha=1e-15, 
    #                       normalize_columns=True, max_iter=200)    
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)#, x_dot=u_dot)  
    return model

def translate_sindy_eq(equation: str):
    """
    Translates a SINDy equation into a human-readable format suitable for further processing within the EPDE framework.
        
        This method takes a SINDy equation string as input, replaces the
        shorthand notation with a more descriptive form, and appends " = du/dx1{power: 1.0}"
        to the end of the translated equation. It relies on a predefined
        correspondence dictionary to map the shorthand notations to their
        corresponding representations. This translation is crucial for ensuring that the discovered equations are interpretable and can be seamlessly integrated into subsequent analysis or simulation steps within the EPDE workflow.
    
        Args:
            equation (str): The SINDy equation string to translate.
    
        Returns:
            str: The translated SINDy equation.
    """
    correspondence = {"0" : "u{power: 1.0}",
                      "0_1" : "du/dx2{power: 1.0}",
                      "0_11" : "d^2u/dx2^2{power: 1.0}"}
    terms = [] # Check EPDE translator input format
    
    def replace(term):
        term = term.replace(' ', '').split('x')
        for idx, factor in enumerate(term[1:]):
            try:
                term[idx+1] = correspondence[factor]
            except KeyError:
                print(f'Key of term {factor} is missing')
                raise KeyError()
        return term
                
    for term in equation.split('+'):
        terms.append(reduce(lambda x, y: x + ' * ' + y, replace(term)))
    terms_comb = reduce(lambda x, y: x + ' + ' + y, terms) + ' = du/dx1{power: 1.0}'
    return terms_comb
    
def sindy_examplar_search(x, t, u, u_dot = None):
    """
    Identifies governing equations using SINDy exemplar search.
    
    This method employs the SINDy algorithm with a PDE library to discover
    the underlying differential equations that describe the provided data.
    It constructs a library of candidate functions and uses sparse regression
    to select the most relevant terms, effectively identifying the equation's structure.
    This approach helps automate the process of identifying governing differential equations from data.
    
    Args:
        x (np.ndarray): Spatial grid points.
        t (np.ndarray): Time points.
        u (np.ndarray): Input data representing the system's state.
        u_dot (np.ndarray, optional): Time derivatives of the input data. If None, it will be calculated using finite differences. Defaults to None.
    
    Returns:
        ps.SINDy: The fitted SINDy model, representing the identified governing equations.
    """
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    
    if u_dot is None:
        u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)
    
    u = u.reshape(len(x), len(t), 1)
    u_dot = u_dot.reshape(len(x), len(t), 1)
    
    print(u.shape, u_dot.shape)
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=x,
        is_uniform=True,
    )
    
    print('STLSQ model: ')
    optimizer = ps.STLSQ(threshold=2, alpha=1e-5, normalize_columns=True)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt, x_dot=u_dot)
    model.print()
    return model
        

if __name__ == '__main__':
    path = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Burgers/Data/'

    try:
        u_file = os.path.join(os.path.dirname( __file__ ), 'data/burgers_sln_256.csv')
        data = np.loadtxt(u_file, delimiter=',').T
    except (FileNotFoundError, OSError):
        u_file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Burgers/Data/burgers_sln_256.csv'
        data = np.loadtxt(u_file, delimiter=',').T

    derives = None
    dx = pd.read_csv(f'{path}burgers_sln_dx_256.csv', header=None)
    d_x = dx.values
    d_x = np.transpose(d_x)

    dt = pd.read_csv(f'{path}burgers_sln_dt_256.csv', header=None)
    d_t = dt.values
    d_t = np.transpose(d_t)

    dtt = pd.read_csv(f'{path}burgers_sln_dtt_256.csv', header=None)
    d_tt = dtt.values
    d_tt = np.transpose(d_tt)

    # derives = np.zeros(shape=(data.shape[0], data.shape[1], 3))
    # derives[:, :, 0] = d_t
    # derives[:, :, 1] = d_tt
    # derives[:, :, 2] = d_x

    derives = np.zeros(shape=(data.shape[0], data.shape[1], 3))
    derives[:, :, 0] = d_t
    derives[:, :, 1] = d_tt
    derives[:, :, 2] = d_x

    # raise NotImplementedError()
    

        
    # u = np.moveaxis(u, 1, 0)

    train_max = 150
    t = np.linspace(0, 4, data.shape[0])
    x = np.linspace(-4000, 4000, data.shape[1])
    grids = np.meshgrid(t, x, indexing = 'ij')
    
    derives = derives[:train_max, ...]
    derives = derives.reshape((-1, 3))
    
    dimensionality = data.ndim - 1; boundary = 20
    grids_training = (grids[0][:train_max, ...], grids[1][:train_max, ...])
    data_training = data[:train_max, ...]
    
    grids_test = (grids[0][train_max:, ...], grids[1][train_max:, ...])
    data_test = data[train_max:, ...]
    # x += np.random.normal(0, err_factor*np.min(x), size = x.size)
    # y += np.random.normal(0, err_factor*np.min(y), size = y.size)

        
    # t_max = 150
    # t_train = t[:t_max]; t_test = t[t_max:] 
    # x = data[:t_max, 0]; x_test = data[t_max:, 0]
    # y = data[:t_max, 1]; y_test = data[t_max:, 1]

    # magnitude = 0.5*1e-2
    # x_n = x #+ np.random.normal(scale = magnitude*x, size = x.shape)
    # y_n = y #+ np.random.normal(scale = magnitude*y, size = y.shape)
    # plt.plot(t_train, x_n)
    # plt.plot(t_train, y_n)
    # plt.show()
    epde = True
    
    if epde:
        test_launches = 1
        models = []
        for idx in range(test_launches):
            epde_search_obj = epde_discovery(grids_training, data_training,
                                             derives, False)
            
            # def get_ode_bop(key, var, grid_loc, value):
            #     bop = BOPElement(axis = 0, key = key, term = [None], power = 1, var = var)
            #     bop_grd_np = np.array([[grid_loc,]])
            #     bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
            #     bop.values = torch.from_numpy(np.array([[value,]])).float()
            #     return bop
                
            # bnd_t = torch.cartesian_prod(torch.from_numpy(np.array([t[train_max + 1]], dtype=np.float64)),
            #                              torch.from_numpy(x)).float()
            
            # bop_1 = BOPElement(axis = 0, key = 'u_t', term = [None], power = 1, var = 0)
            # bop_1.set_grid(bnd_t)
            # bop_1.values = torch.from_numpy(data_test[0, ...]).float()
            
            # t_der = epde_search_obj.saved_derivaties['u'][..., 0].reshape(grids_training[0].shape)
            # bop_2 = BOPElement(axis = 0, key = 'dudt', term = [0], power = 1, var = 0)
            # bop_2.set_grid(bnd_t)
            # bop_2.values = torch.from_numpy(t_der[-1, ...]).float()
            
            # bnd_x1 = torch.cartesian_prod(torch.from_numpy(t[train_max:]),
            #                               torch.from_numpy(np.array([x[0]], dtype=np.float64))).float()
            # bnd_x2 = torch.cartesian_prod(torch.from_numpy(t[train_max:]),
            #                               torch.from_numpy(np.array([x[-1]], dtype=np.float64))).float()            
            
            # bop_3 = BOPElement(axis = 1, key = 'u_x1', term = [None], power = 1, var = 0)
            # bop_3.set_grid(bnd_x1)
            # bop_3.values = torch.from_numpy(data_test[..., 0]).float()            

            # bop_4 = BOPElement(axis = 1, key = 'u_x2', term = [None], power = 1, var = 0)
            # bop_4.set_grid(bnd_x2)
            # bop_4.values = torch.from_numpy(data_test[..., -1]).float()            
            
            # # bop_grd_np = np.array([[,]])
            
            # # bop = get_ode_bop('u', 0, t_test[0], x_test[0])
            # # bop_y = get_ode_bop('v', 1, t_test[0], y_test[0])
            
            
            # pred_u_v = epde_search_obj.predict(system=sys, boundary_conditions=[bop_1(), bop_2(), bop_3(), bop_4()], 
            #                                     grid = grids_test, strategy='NN')
            # errs.append((np.mean(np.abs(x_test - pred_u_v[:, 0])), 
            #              np.mean(np.abs(y_test - pred_u_v[:, 1]))))
            # models.append((sys, epde_search_obj))
            models.append(epde_search_obj)
    else:
        # errs = []
        # models = []
        # test_launches = 10
        # for idx in range(test_launches):
        #     model = sindy_discovery(grids_training, data_training)#, d_t)
        #     # print('Initial conditions', np.array([x_test[0], y_test[0]]))
        #     # pred_u_v = model.simulate(np.array([x_test[0], y_test[0]]), t_test)
        #     # models.append(model)
            
        #     # plt.plot(t_test, x_test, '+', label = 'preys_odeint')
        #     # plt.plot(t_test, y_test, '*', label = "predators_odeint")
        #     # plt.plot(t_test, pred_u_v[:, 0], color = 'b', label='preys_NN')
        #     # plt.plot(t_test, pred_u_v[:, 1], color = 'r', label='predators_NN')
        #     # plt.xlabel('Time t, [days]')
        #     # plt.ylabel('Population')
        #     # plt.grid()
        #     # plt.legend(loc='upper right')       
        #     # plt.show()
        #     # errs.append((np.mean(np.abs(x_test - pred_u_v[:, 0])), 
        #     #              np.mean(np.abs(y_test - pred_u_v[:, 1]))))        
        #     models.append(model)        
        print(f'x.shape: {x.shape}, t.shape: {t.shape}, data.shape: {data.shape}')
        model = sindy_examplar_search(x, t, data.T, d_t.T)