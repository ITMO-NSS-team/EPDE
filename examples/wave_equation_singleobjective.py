#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

import numpy as np
import epde.interface.interface as epde_alg

from epde.interface.prepared_tokens import CustomTokens, CacheStoredTokens, TrigonometricTokens

if __name__ == '__main__':
    
    '''
    
    Loading data, representing the wave equation solution and moving the time axis into 
    the first position. 
    
    '''
    
    try:
        print(os.path.dirname( __file__ ))
        data_file = os.path.join(os.path.dirname( __file__ ), 'data/wave_sln_80.csv')
        data = np.loadtxt(data_file, delimiter = ',').T
    except FileNotFoundError:
        data_file = '/home/maslyaev/epde/EPDE_main/examples/data/wave_sln_80.csv'
        data = np.loadtxt(data_file, delimiter = ',').T
    
    '''

    Defining grids and boundary for the domain. The grids can be (and will be) 
    used as a separate family of tokens and is necessary, when we add functions,
    dependent on the coordinates, such as trigonometric functions, into the pool.
    Also, we specify boundary for the domain: the derivatives near the domain 
    boundary tend to have high computational errors.

    '''
    
    t = np.linspace(0, 1, 81); x = np.linspace(0, 1, 81)
    grids = np.meshgrid(t, x, indexing = 'ij')
    dimensionality = data.ndim - 1   
    
    boundary = 10
    '''
    Here, we define the object, dedicated to the equation search: among the 
    initialization arguments, the most important include dimensionality (here we 
    must pass the dimensionality of the input dataset), and number of the equation 
    search iterations. Multiobjective mode flag controls the optimization procedure: if it is False, 
    the algorithm executes a singleobjective optimization, detecting only a single best candidate equation from the 
    point of process representation. Otherwise, in multi-objective optimization mode, a Pareto frontier, 
    containing solutions, "best", according to selected metrics (that are complexity and quality) is detected.
    '''

    multiobjective_mode = False
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                          dimensionality = dimensionality, boundary = 10,
                                          coordinate_tensors = grids)

    '''
    Setting memory usage for cache stored tokens and terms (that are the ones, that 
    are saved after initial calculations during algorithm operations to avoid 
    redundant computations).

    To prepare for the equation search, data can be denoised, and derivatives have to be computed. 
    Here by .set_preprocessor() we set ANN-based data smoothing, and from that ANN values, the derivatives 
    are calculated, using finite differences. By default, Chebyshev polynomials (default_preprocessor_type='poly') 
    are used to represent the data, and their analytical derivativs are used in algorithm.
    
    Next, specifying parameters of the optimization algorithm, such as number of epochs and population size,
    with .set_singleobjective_params(). 
    '''
    
     
    epde_search_obj.set_memory_properties(example_tensor = data, mem_for_cache_frac = 15)

    epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', preprocessor_kwargs={'epochs_max' : 10000})

    popsize = 7
    # if multiobjective_mode:
    #     epde_search_obj.set_moeadd_params(population_size = popsize, 
    #                                       training_epochs=40)
    # else:
    epde_search_obj.set_singleobjective_params(population_size = popsize, 
                                               training_epochs=60)
    '''

    Defining tokens, containing grid, so our discovered equation can have 
    terms like "t * du/dt". Here, we operate on the synthetic data and we do 
    not expect their presence in the desired equation. However, they can be present
    in an equation, describing some real-world data, thus we provide a tool for their
    inclusion.
    
    To increase the pool size and artificially complicate the equation search problem, we
    can include trigonometric tokens.
    
    '''

    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           token_labels = ['t', 'x'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)

    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)

    '''
    Method epde_search.fit() is used to initiate the equation search. 
    '''
    
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    opt_val = 5e-1
    bounds = (1e-8, 1e0) if multiobjective_mode else (opt_val, opt_val)    
    epde_search_obj.fit(data=data, variable_names=['u',], max_deriv_order=(2, 2),
                        equation_terms_max_number=6, data_fun_pow = 1, additional_tokens=[trig_tokens, custom_grid_tokens], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds)

    '''

    Here we have conducted a single-objective optimization, ideally detecting equation, similar to 

    0.0 * du/dx1{power: 1.0} + 0.0 * du/dx2{power: 1.0} * cos{power: 1.0, freq: 4.663145421047243, dim: 1.0} 
    + 0.0 * du/dx2{power: 1.0} + 0.0 * u{power: 1.0} + 0.039843458066873096 * d^2u/dx2^2{power: 1.0} + 
    -0.021396897100174943 = d^2u/dx1^2{power: 1.0}
    {'terms_number': {'optimizable': False, 'value': 6}, 
     'max_factors_in_term': {'optimizable': False, 'value': {'factors_num': [1, 2], 
                                                             'probas': [0.8, 0.2]}}, 
    ('sparsity', 'u'): {'optimizable': False, 'value': 0.5}} ,
    with objective function values of [9.13122923]
    
    '''
    
    epde_search_obj.equation_search_results(only_print=True, num = 1)