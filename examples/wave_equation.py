#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import epde.interface.interface as epde_alg

from epde.interface.prepared_tokens import CustomTokens, CacheStoredTokens

if __name__ == '__main__':
    
    '''
    
    Loading data, representing the wave equation solution, reshaping it to its 
    original shape and moving the time axis into the first position. 
    
    '''
    u = np.loadtxt('examples/data/Wave_101x101x101.csv').reshape((101, 101, 101))
    u = np.moveaxis(u, 2, 0) # moving time axis to be the first one in the ndarray shape
    
    '''
    Defining grids and boundary for the domain. The grids can be (and will be) 
    used as a separate family of tokens and is necessary, when we add functions,
    dependent on the coordinates, such as trigonometric functions, into the pool.
    Also, we specify boundary for the domain: the derivatives near the domain 
    boundary tend to have high computational errors.
    '''
    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 0.2, u.shape[1])
    y = np.linspace(0, 0.2, u.shape[2])
    
    boundary = 20

    dimensionality = u.ndim    
    grids = np.meshgrid(t, x, y, indexing = 'ij')

    '''
    Here, we define the object, dedicated to the equation search: among the 
    initialization arguments, the most important include dimensionality (here we 
    must pass the dimensionality of the input dataset), and number of the equation 
    search iterations. 
    '''

    epde_search_obj = epde_alg.epde_search(multiobjective_mode=False, use_solver=False, 
                                           eq_search_iter = 100, dimensionality=dimensionality)

    '''
    Setting memory usage for cache stored tokens and terms (that are the ones, that 
    are saved after initial calculations during algorithm operations to avoid 
    redundant computations).

    Also, specifying parameters of the multiobjective optimization algorithm with 
    .set_moeadd_params(). That ste is optional     
    '''

    epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)
    epde_search_obj.set_moeadd_params(population_size=3, training_epochs = 5)

    '''
    Defining tokens, containing grid, so our discovered equation can have 
    terms like "t * du/dt". Here, we operate on the synthetic data and we do 
    not expect their presence in the desired equation. However, they can be present
    in an equation, describing some real-world data, thus we provide a tool for their
    inclusion.
    '''

    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           boundary = boundary,
                                           token_labels = ['t', 'x', 'y'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1], 
                                                          'y' : grids[2]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)

    '''
    Method epde_search.fit() is used to initiate the equation search.
    '''
    
    epde_search_obj.fit(data = u, max_deriv_order=(2, 2, 2), boundary = boundary, 
                        equation_terms_max_number = 4, equation_factors_max_number = 1,
                        coordinate_tensors = grids, eq_sparsity_interval = (1e-8, 5.0),
                        deriv_method='poly', deriv_method_kwargs={'smooth': True, 'grid': grids},
                        additional_tokens = [custom_grid_tokens,],
                        memory_for_cache=25, prune_domain = False)

    '''
    The results of the equation search have the following format: if we call method 
    .equation_search_results() with "only_print = True", the Pareto frontiers 
    of equations of varying complexities will be shown, as in the following example:
        
    0-th non-dominated level


    0.04180013539743802 * d^2u/dx2^2{power: 1.0} + 0.0 * du/dx3{power: 1.0} + 0.041800135397438 * d^2u/dx3^2{power: 1.0} + -0.012393007634545834 = d^2u/dx1^2{power: 1.0}
     , with objective function values of [2.96089253e+04 3.00000000e+00] 
    
    0.0 * d^2u/dx2^2{power: 1.0} + 0.0 * d^2u/dx1^2{power: 1.0} + 0.0 * du/dx2{power: 1.0} + 0.0 = du/dx1{power: 1.0}
     , with objective function values of [1.46711383e+05 1.00000000e+00] 
    
    
    
    1-th non-dominated level
    
    
    -0.33704911693096196 * u{power: 1.0} + 0.04130442805964803 * d^2u/dx3^2{power: 1.0} + 0.04130442805964804 * d^2u/dx2^2{power: 1.0} + -0.019017614951822803 = d^2u/dx1^2{power: 1.0}
     , with objective function values of [2.97841965e+04 4.00000000e+00]     
        
    If the method is called with the "only_print = False", the algorithm will return list 
    of Pareto frontiers with the desired equations.
    '''
    
    epde_search_obj.equation_search_results(only_print=True, level_num = 2)