#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:51:40 2022

@author: maslyaev
"""

import numpy as np
import epde.interface.interface as epde_alg

from epde.interface.equation_translator import CoeffLessEquation
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens

import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

# from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator
# TODO^ caching of the pre-calculated derivatives

def translate_eq():
    """
    Translates and evaluates a partial differential equation (PDE) by constructing candidate equations and evaluating their fitness against the provided data.
        
        This method sets up the coordinate grid, defines the left-hand side (LHS) and right-hand side
        (RHS) terms of the PDE, and then uses the EPDE algorithm to search for a solution by
        exploring combinations of terms and coefficients. Finally, it evaluates the identified
        equation and prints the equation's text form and the mean of the evaluated result,
        demonstrating the equation discovery process.
        
        Args:
            None
        
        Returns:
            None. Prints the equation's text form and the mean of the evaluated result. The equation
            represents the discovered relationship between the terms, and the mean provides a measure
            of how well the equation fits the data.
    """
    u = np.loadtxt('/home/maslyaev/epde/EPDE_main/projects/wave/data.csv').reshape((101, 101, 101))
    u = np.moveaxis(u, 2, 0)

    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 0.2, u.shape[1])
    y = np.linspace(0, 0.2, u.shape[2])    
    grids = np.meshgrid(t, x, y, indexing = 'ij')
    
    dimensionality = u.ndim - 1; boundary = 20

    lp_terms = [['d^2u/dx3^2{power: 1}'], ['d^2u/dx2^2{power: 1}']]
    rp_term = ['d^2u/dx1^2{power: 1}',]
    epde_search_obj = epde_alg.epde_search(use_solver = False, dimensionality = dimensionality, boundary = boundary,
                                           coordinate_tensors = grids)    

    epde_search_obj.create_pool(data = u, max_deriv_order=(2, 2, 2), additional_tokens = [], 
                                method = 'poly', method_kwargs={'smooth': False, 'grid': grids})

    test1 = CoeffLessEquation(lp_terms, rp_term, epde_search_obj.pool)
    def map_to_equation(equation, function, function_kwargs = dict()):
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        return function(np.abs(np.dot(features, equation.weights_final[:-1]) + 
                              np.full(target.shape, equation.weights_final[-1]) - 
                              target), **function_kwargs)    
    
    print(test1.equation.text_form)
    print(map_to_equation(test1.equation, np.mean))
    
def run_wave_eq_search(multiobjective_mode):
    """
    Runs the equation search for the wave equation using the EPDE framework.
        
        This method leverages the EPDE (Equation Parameter Discovery and Estimation) search algorithm to automatically identify the mathematical equation that best describes the wave phenomenon, given the provided data and configuration. It defines the search space of possible equation terms, preprocesses the input data to enhance the search, and optimizes the equation parameters to achieve the best fit. This automated approach helps in understanding the underlying physics of wave propagation by discovering the governing equation directly from data.
        
        Args:
            multiobjective_mode (bool): A boolean flag indicating whether to use multiobjective optimization. When True, the search aims to balance multiple objectives, such as accuracy and equation complexity.
        
        Returns:
            list: A list of the best identified equations, ranked according to their fitness. Each equation represents a potential mathematical model for the wave equation.
    """
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver=False, 
                                           dimensionality=dimensionality, boundary=boundary, 
                                           coordinate_tensors = grids, 
                                           verbose_params = {})    
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True
                                     preprocessor_kwargs={'use_smoothing' : False})
    popsize = 7
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=40)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize, 
                                                   training_epochs=40)


    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           # boundary = boundary,
                                           token_labels = ['t', 'x'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)

    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    opt_val = 1e-1
    bounds = (1e-8, 1e0) if multiobjective_mode else (opt_val, opt_val)
    epde_search_obj.fit(data=[u, ], variable_names=['u',], max_deriv_order=(2, 2),
                        equation_terms_max_number=4, data_fun_pow = 2, additional_tokens=[trig_tokens,], #custom_grid_tokens 
                        equation_factors_max_number = factors_max_number, 
                        eq_sparsity_interval = bounds)
    return epde_search_obj.equation_search_results(only_print = False, num = 1)        
    
if __name__ == "__main__":
    
    # multiobjective_mode = True
    
    try:
        u_file = os.path.join(os.path.dirname( __file__ ), 'projects/benchmarking/wave/wave_sln_80.csv')
        u = np.loadtxt(u_file, delimiter=',')
    except (FileNotFoundError, OSError):
        u_file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/wave/wave_sln_90.csv'
        u = np.loadtxt(u_file, delimiter=',')
        
    u = np.moveaxis(u, 1, 0)

    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])
    grids = np.meshgrid(t, x, indexing = 'ij')
    
    dimensionality = u.ndim - 1; boundary = 20

    paretos = []
    exp_num = 2
    for exp_run in range(exp_num):
        paretos.append(run_wave_eq_search(multiobjective_mode = False))