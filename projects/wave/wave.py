#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:51:40 2022

@author: maslyaev
"""

import numpy as np
import epde.interface.interface as epde_alg

from epde.interface.equation_translator import Coeff_less_equation
from epde.interface.prepared_tokens import CustomTokens, CacheStoredTokens
from epde.interface.prepared_tokens import TrigonometricTokens


from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator
# TODO^ caching of the pre-calculated derivatives

def translate_eq():
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
                                           coordinate_tensors = grids, verbose_params = {'show_moeadd_epochs' : True})    

    epde_search_obj.create_pool(data = u, max_deriv_order=(2, 2, 2), additional_tokens = [], 
                                method = 'poly', method_kwargs={'smooth': False, 'grid': grids})

    test1 = Coeff_less_equation(lp_terms, rp_term, epde_search_obj.pool)
    def map_to_equation(equation, function, function_kwargs = dict()):
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        return function(np.abs(np.dot(features, equation.weights_final[:-1]) + 
                              np.full(target.shape, equation.weights_final[-1]) - 
                              target), **function_kwargs)    
    
    print(test1.equation.text_form)
    print(map_to_equation(test1.equation, np.mean))
    
if __name__ == "__main__":

    # def full_search():
        u = np.loadtxt('/home/maslyaev/epde/EPDE_main/projects/wave/data.csv').reshape((101, 101, 101))
        u = np.moveaxis(u, 2, 0)
    
        t = np.linspace(0, 1, u.shape[0])
        x = np.linspace(0, 0.2, u.shape[1])
        y = np.linspace(0, 0.2, u.shape[2])    
        grids = np.meshgrid(t, x, y, indexing = 'ij')
        
        dimensionality = u.ndim - 1; boundary = 30
    
        paretos = []
        exp_num = 1
        for i in range(exp_num):
            epde_search_obj = epde_alg.epde_search(use_solver = False, dimensionality = dimensionality, boundary = boundary,
                                                   coordinate_tensors = grids, verbose_params = {'show_moeadd_epochs' : True})    
            
            popsize = 7
            epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=40)
            epde_search_obj.set_preprocessor(default_preprocessor_type='poly', 
                                             preprocessor_kwargs={'use_smoothing' : False,
                                                                  })
        
            order = 5
            
            def get_polynomial_family(tensor, order, token_type = 'polynomials'):
                '''
                Get family of tokens for polynomials of orders from second up to order argument.
                '''
                assert order > 1
                labels = [f'p^{idx+1}' for idx in range(1, order)]
                tensors = {label : tensor ** (idx + 2) for idx, label in enumerate(labels)}
                return CacheStoredTokens(token_type = token_type,
                                         token_labels = labels,
                                         token_tensors = tensors,
                                         params_ranges = {'power' : (1, 1)},
                                         params_equality_ranges = None, 
                                         meaningful = True)
        
            trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
            factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.95, 0.05]}
            
            epde_search_obj.fit(data=[u, ], variable_names=['u',], max_deriv_order=(2, 2, 2),
                                equation_terms_max_number=5, data_fun_pow = 1, additional_tokens=[trig_tokens,], #custom_grid_tokens 
                                equation_factors_max_number = factors_max_number, 
                                eq_sparsity_interval=(1e-10, 1e-4), coordinate_tensors=grids)
            paretos.append(epde_search_obj.equations(only_print = False, level_num = 1))
    #     return paretos
    
    # translate_eq()