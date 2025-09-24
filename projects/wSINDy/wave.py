#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:53:27 2023

@author: maslyaev
"""

import numpy as np

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

def write_pareto(dict_of_exp):
    """
    Writes Pareto front solutions to text files for each experiment.
    
        This function iterates through a dictionary of experimental results, extracting
        Pareto front solutions and writing them to separate text files. The filename
        is derived from the experiment parameters (dictionary key), and each iteration's
        Pareto front is written with equation strings separated by newlines. This
        facilitates the analysis and comparison of discovered equation structures
        across different experimental settings, aiding in the identification of robust
        and generalizable models.
    
        Args:
            dict_of_exp: Dictionary where keys are tuples representing experiment
                parameters and values are lists of lists of Pareto front objects.
    
        Returns:
            None. This method writes data to files and does not return any value.
    """
    for key, item in dict_of_exp.items():
        test_key = str(key[0]).replace('.', '_') + '__' + str(key[1]).replace('.', '_')
        with open('/home/maslyaev/epde/EPDE_main/projects/hunter-prey/param_var/'+test_key+'.txt', 'w') as f:
            for iteration in range(len(item)):
                f.write(f'Iteration {iteration}\n\n')
                for ind in [pareto.text_form for pareto in item[iteration][0]]:
                    f.write(ind + '\n\n')

def epde_discovery(grids, data, derivs, use_ann = False):
    """
    Performs symbolic regression to identify governing equations from data using an evolutionary algorithm.
        
        This method sets up and executes the search algorithm to discover
        equations that describe the provided data. It configures the search space,
        defines token types, and fits the model to the data by searching for the best equation structure.
        
        Args:
            grids: Spatial and temporal grid coordinates. These coordinates provide the independent variable values for the data.
            data: The data to fit the equations to. This is the dependent variable data that the algorithm attempts to model.
            derivs: Precomputed derivatives of the data. (Not used in the current implementation)
        
        Returns:
            tuple: A tuple containing:
                - The EPDE search object, which holds the state and results of the search.
                - The discovered equations with complexity 2, representing the simplest identified relationships.
                - The saved derivatives from the EPDE search object. These derivatives can be used for further analysis or validation.
    """
    multiobjective_mode = True
    dimensionality = data.ndim - 1
    
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                          dimensionality = dimensionality, boundary = 10,
                                          coordinate_tensors = grids)    
    # epde_search_obj.set_preprocessor(default_preprocessor_type='spectral', # use_smoothing = True
    #                                  preprocessor_kwargs={})
    popsize = 7
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=40)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize, 
                                                   training_epochs=40)
    # epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=50)
    
    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           token_labels = ['t', 'x'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)        
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    opt_val = 1e-1
    bounds = (1e-8, 1e0) if multiobjective_mode else (opt_val, opt_val)    
    epde_search_obj.fit(data=data, variable_names=['u',], max_deriv_order=(2, 2),
                        equation_terms_max_number=5, data_fun_pow = 1, additional_tokens=[trig_tokens, custom_grid_tokens], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds)
    sys = epde_search_obj.get_equations_by_complexity(2)
    return epde_search_obj, sys[0], epde_search_obj.saved_derivaties 
    

# def sindy_discovery(grids, u):
#     t = np.unique(grids[0])
#     x = np.unique(grids[1])
#     # u = u.T
#     # u_ax_arr = ps.AxesArray(u, {"ax_time":0, "ax_spatial":1})
    
#     library_functions = [
#         lambda x: x,
#         lambda x: x * x * x,
#         lambda x, y: x * y * y,
#         lambda x, y: x * x * y,
#     ]
#     library_function_names = [
#         lambda x: x,
#         lambda x: x + x + x,
#         lambda x, y: x + y + y,
#         lambda x, y: x + x + y,
#     ]
#     u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t = t[1] - t[0])
#     # print('u_ax_arr', u_ax_arr.shape, 'u_dot.shape', u_dot.shape)
#     pde_lib = ps.PDELibrary(
#         library_functions=library_functions,
#         function_names=library_function_names,
#         derivative_order=2,
#         spatial_grid=x,
#         include_bias=True,
#         is_uniform=True
#     ).fit([u])
#     # print(spatial_grid.shape)
#     print('STLSQ model: ')
#     optimizer = ps.STLSQ(threshold=50, alpha=1e-5, 
#                          normalize_columns=True, max_iter=200)
#     model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
#     print(u.shape, t.shape)
#     model.fit(u,  x_dot=u_dot)
#     model.print()
#     return model

def sindy_discovery(grids, u):
    """
    Discovers a sparse representation of the governing PDE from the given data using SINDy.
    
    This method constructs a SINDy model, fits it to the provided spatio-temporal data,
    and returns the fitted model. It leverages a PDELibrary for feature engineering,
    allowing for custom basis functions and derivative calculations, and SR3 for sparse
    regression. This enables the identification of the most relevant terms in the PDE.
    
    Args:
        grids: Grid points for the data. Assumed to contain time and space grids.
        u: Solution data to be modeled.
    
    Returns:
        The fitted SINDy model, representing the discovered PDE.
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
    model.fit(u, t=dt)  
    return model

if __name__ == '__main__':
    results = []
    shape = 80
        
    try:
        print(os.path.dirname( __file__ ))
        data_file = os.path.join(os.path.dirname( __file__ ), f'data/wave/wave_sln_{shape}.csv')
        data = np.loadtxt(data_file, delimiter = ',').T
    except FileNotFoundError:
        data_file = '/home/maslyaev/epde/EPDE_main/projects/wSINDy/data/wave/wave_sln_{shape}.csv'
        data = np.loadtxt(data_file, delimiter = ',').T

    t = np.linspace(0, 1, shape+1); x = np.linspace(0, 1, shape+1)
    train_max = 40
    grids = np.meshgrid(t, x, indexing = 'ij')
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
        derivs = None
        for idx in range(test_launches):
            epde_search_obj, sys, derivs = epde_discovery(grids_training, data_training,
                                                          derivs, False)
            
            # def get_ode_bop(key, var, grid_loc, value):
            #     bop = BOPElement(axis = 0, key = key, term = [None], power = 1, var = var)
            #     bop_grd_np = np.array([[grid_loc,]])
            #     bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
            #     bop.values = torch.from_numpy(np.array([[value,]])).float()
            #     return bop
                
            bnd_t = torch.cartesian_prod(torch.from_numpy(np.array([t[train_max + 1]], dtype=np.float64)),
                                         torch.from_numpy(x)).float()
            
            bop_1 = BOPElement(axis = 0, key = 'u_t', term = [None], power = 1, var = 0)
            bop_1.set_grid(bnd_t)
            bop_1.values = torch.from_numpy(data_test[0, ...]).float()
            
            t_der = epde_search_obj.saved_derivaties['u'][..., 0].reshape(grids_training[0].shape)
            bop_2 = BOPElement(axis = 0, key = 'dudt', term = [0], power = 1, var = 0)
            bop_2.set_grid(bnd_t)
            bop_2.values = torch.from_numpy(t_der[-1, ...]).float()
            
            bnd_x1 = torch.cartesian_prod(torch.from_numpy(t[train_max:]),
                                          torch.from_numpy(np.array([x[0]], dtype=np.float64))).float()
            bnd_x2 = torch.cartesian_prod(torch.from_numpy(t[train_max:]),
                                          torch.from_numpy(np.array([x[-1]], dtype=np.float64))).float()            
            
            bop_3 = BOPElement(axis = 1, key = 'u_x1', term = [None], power = 1, var = 0)
            bop_3.set_grid(bnd_x1)
            bop_3.values = torch.from_numpy(data_test[..., 0]).float()            

            bop_4 = BOPElement(axis = 1, key = 'u_x2', term = [None], power = 1, var = 0)
            bop_4.set_grid(bnd_x2)
            bop_4.values = torch.from_numpy(data_test[..., -1]).float()            
            
            # bop_grd_np = np.array([[,]])
            
            # bop = get_ode_bop('u', 0, t_test[0], x_test[0])
            # bop_y = get_ode_bop('v', 1, t_test[0], y_test[0])
            
            
            pred_u_v = epde_search_obj.predict(system=sys, boundary_conditions=[bop_1(), bop_2(), bop_3(), bop_4()], 
                                                grid = grids_test, strategy='NN')
            # errs.append((np.mean(np.abs(x_test - pred_u_v[:, 0])), 
            #              np.mean(np.abs(y_test - pred_u_v[:, 1]))))
            models.append((sys, epde_search_obj))
    else:
        errs = []
        models = []
        test_launches = 10
        for idx in range(test_launches):
            model = sindy_discovery(grids_training, data_training)
            # print('Initial conditions', np.array([x_test[0], y_test[0]]))
            # pred_u_v = model.simulate(np.array([x_test[0], y_test[0]]), t_test)
            # models.append(model)
            
            # plt.plot(t_test, x_test, '+', label = 'preys_odeint')
            # plt.plot(t_test, y_test, '*', label = "predators_odeint")
            # plt.plot(t_test, pred_u_v[:, 0], color = 'b', label='preys_NN')
            # plt.plot(t_test, pred_u_v[:, 1], color = 'r', label='predators_NN')
            # plt.xlabel('Time t, [days]')
            # plt.ylabel('Population')
            # plt.grid()
            # plt.legend(loc='upper right')       
            # plt.show()
            # errs.append((np.mean(np.abs(x_test - pred_u_v[:, 0])), 
            #              np.mean(np.abs(y_test - pred_u_v[:, 1]))))        
            models.append(model)        