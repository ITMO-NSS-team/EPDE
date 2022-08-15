#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:10:30 2021

@author: maslyaev
"""

import os
import pandas as pd

import numpy as np
import epde.interface.interface as epde_alg
import pickle

from epde.interface.equation_translator import Coeff_less_equation
from epde.interface.prepared_tokens import Custom_tokens, Cache_stored_tokens
from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator

from epde.prep.interpolation_oversampling import BesselInterpolator

def load_data(directory, data_file_format = 'dat', loader_kwargs = {}, 
              temp_initial = {}, voltage_prefix = '06'):
    data = []
    t = None
    x = 1e-4 * np.array(list(temp_initial.keys()))
    
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".dat"):
            temp = pd.read_csv(directory + '/' + filename, **loader_kwargs)
            if t is None:
                t = temp.loc[temp.tc >= 0].ts.to_numpy()
            dist = int(filename.replace(voltage_prefix + 'V_', '').replace('mm_u.dat', ''))
                  
            data.append((dist, temp.loc[temp.tc >= 0].Tr.to_numpy()))
    len_min = min([series.size for _, series in data])
    data = sorted([(dist, series[:len_min]) for dist, series in data], 
                  key = lambda x: x[0])
    
    return t, x, np.stack([series - temp_initial[dist] for dist, series in data])

# def process_row(row):
#     new_row = np.empty_like(row)
    
#     for idx, elem in np.ndenumerate(row):
#         if row[-1] = 

# def remove_stairs(data, axis = 0):
#     data_new = data.copy()
#     data_new = np.moveaxis(data_new, source = axis, destination = 0)
    


def load_velocity_data(filename = '/home/maslyaev/epde/EPDE_main/projects/cylinder/data/velocity.dat', t_interval = [0, -1]):
    data = np.loadtxt(filename, skiprows = 5, delimiter = ',')
    time_idx = (0,)
    temp_index = (1, 5, 9, 13, 17, 21)
    velocity_index = (2, 6, 10, 14, 18, 22)
    return (data[t_interval[0]:t_interval[1], time_idx],
            data[t_interval[0]:t_interval[1], temp_index],
            data[t_interval[0]:t_interval[1], velocity_index])


def velocity_assumptions():
    temp_initial = {7 : 27.1019,
                    12 : 27.2864,
                    17 : 27.3925,
                    22 : 27.4159,
                    27 : 27.6132}

    t_min = 3000
    t_max = 9000
    directory = 'tests/cylinder/data/Convection'
    
    t, x, u = load_data(os.getcwd() + '/' + directory, 
                        loader_kwargs={'skiprows':2, 
                                       'sep' : '\t', 
                                       'error_bad_lines' : False, 
                                       'names' : ["ts","tc","Tr","Tm"]}, 
                        temp_initial=temp_initial, voltage_prefix='13')
    u = u.T[t_min:t_max, :]; t = t[t_min:t_max]

    grids = np.meshgrid(t, x, indexing = 'ij')
    # u = file[:t_max, 1:]

    boundary = [10, 0]

    dimensionality = u.ndim
        
    epde_search_obj = epde_alg.epde_search()
    epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    velo_tensor = np.load('/home/maslyaev/epde/EPDE_stable/tests/cylinder/data/Convection/velocity.npy')
    # velo_left = 
    velo_tensor = velo_tensor[t_min : t_max, :]
    velocity_tokens = Cache_stored_tokens('velocity', boundary, ['v'], {'v':velo_tensor}, 
                                          {'power' : (1, 1)}, {'power' : 1})
    
    inv_fun_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality - 1)}
    
    custom_inv_fun_tokens = Custom_tokens(token_type = 'inverse', # Выбираем название для семейства токенов - обратных функций.
                                       token_labels = ['1/x_[dim]',], # Задаём названия токенов семейства в формате python-list'a.
                                                                     # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                       evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                       params_ranges = inv_fun_params_ranges, # Используем заявленные диапазоны параметров
                                       params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
                                                                      # (равенство при лишь полном совпадении дискретных параметров)
                                                                      # нас устраивает.
                                                                     
    custom_grid_tokens = Cache_stored_tokens(token_type = 'grid', 
                                       boundary = boundary,
                                       token_labels = ['t', 'r'], 
                                       token_tensors={'t' : grids[0], 'r' : grids[1]},
                                       params_ranges = {'power' : (1, 1)},
                                       params_equality_ranges = None)


    epde_search_obj.create_pool(data = u, max_deriv_order=(1, 2), boundary=boundary, coordinate_tensors = grids, 
                        additional_tokens = [custom_grid_tokens, custom_inv_fun_tokens, velocity_tokens], 
                        method='ANN', method_kwargs = {'epochs_max' : 7500}, 
                        memory_for_cache=5, prune_domain = True,
                        division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 4.)))
    
    lp_terms = [['1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], ['d^2u/dx2^2{power: 1}'], 
                ['v{power: 1}', 'du/dx2{power: 1}']]
    rp_term = ['du/dx1{power: 1}',]
    test1 = Coeff_less_equation(lp_terms, rp_term, epde_search_obj.pool)
    
    def map_to_equation(equation, function, function_kwargs = dict()):
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        return function(np.abs(np.dot(features, equation.weights_final[:-1]) + 
                              np.full(target.shape, equation.weights_final[-1]) - 
                              target), **function_kwargs)    
    
    print(test1.equation.text_form)
    print(map_to_equation(test1.equation, np.mean))

    lp_terms = [['1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], ['d^2u/dx2^2{power: 1}']]
    rp_term = ['du/dx1{power: 1}',]
    test2 = Coeff_less_equation(lp_terms, rp_term, epde_search_obj.pool)

    def map_to_equation(equation, function, function_kwargs = dict()):
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        return function(np.abs(np.dot(features, equation.weights_final[:-1]) + 
                              np.full(target.shape, equation.weights_final[-1]) - 
                              target), **function_kwargs)

    print(test2.equation.text_form)
    print(map_to_equation(test2.equation, np.mean))
    return None

    
def velocity_model(t_interval = [0, -1]):
    x = 1e-4 * np.array([7, 12, 17, 22, 27, 32])
    t, u_smol, velocity_smol = load_velocity_data(t_interval = t_interval)
    
    boundary = [10, 5]
    
    oversampling_size = 30
    oversampling_x = np.linspace(x[0], x[-1], oversampling_size)  
    
    def oversampling_approx(x_init, x_new, row, order = 4):
        BI = BesselInterpolator(x_init, row, max_order = order)
        return BI.approximate(x_new)    
    
    print(x, oversampling_x, u_smol.shape)
    u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
                   for idx in range(u_smol.shape[0])])
    
    velocity = np.vstack([oversampling_approx(x, oversampling_x, velocity_smol[idx, :]) 
                   for idx in range(velocity_smol.shape[0])])
    
    grids = np.meshgrid(t, oversampling_x, indexing = 'ij')
    dimensionality = u.ndim
    


    epde_search_obj = epde_alg.epde_search()
    epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    velocity_tokens = Cache_stored_tokens('velocity', boundary, ['v'], {'v':velocity}, 
                                          {'power' : (1, 1)}, {'power' : 1})
    
    
    inv_fun_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality - 1)}

    custom_inv_fun_tokens = Custom_tokens(token_type = 'inverse', # Выбираем название для семейства токенов - обратных функций.
                                       token_labels = ['1/x_[dim]',], # Задаём названия токенов семейства в формате python-list'a.
                                                                     # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                       evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                       params_ranges = inv_fun_params_ranges, # Используем заявленные диапазоны параметров
                                       params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
                                                                      # (равенство при лишь полном совпадении дискретных параметров)
                                                                      # нас устраивает.
                                                                     
    custom_grid_tokens = Cache_stored_tokens(token_type = 'grid', 
                                       boundary = boundary,
                                       token_labels = ['t', 'r'], 
                                       token_tensors={'t' : grids[0], 'r' : grids[1]},
                                       params_ranges = {'power' : (1, 1)},
                                       params_equality_ranges = None)


    epde_search_obj.create_pool(data = u, max_deriv_order=(1, 2), boundary=boundary, coordinate_tensors = grids, 
                        additional_tokens = [custom_grid_tokens, custom_inv_fun_tokens, velocity_tokens], 
                        method='poly', method_kwargs = {'smooth': True, 'grid': grids, 'sigma' : 3}, 
                        memory_for_cache=5, prune_domain = True,
                        division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 4.)))
    
    lp_terms = [['1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], ['d^2u/dx2^2{power: 1}'], 
                ['v{power: 1}', 'du/dx2{power: 1}']]
    # lp_terms = [['1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], ['d^2u/dx2^2{power: 1}']]    
    rp_term = ['du/dx1{power: 1}',]
    test1 = Coeff_less_equation(lp_terms, rp_term, epde_search_obj.pool)
    
    def map_to_equation(equation, function, function_kwargs = dict()):
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        return function(np.abs(np.dot(features, equation.weights_final[:-1]) + 
                              np.full(target.shape, equation.weights_final[-1]) - 
                              target), **function_kwargs)

    print(test1.equation.text_form)
    print(map_to_equation(test1.equation, np.mean))    
    return None

    
def velocity_fair():
    x = 1e-4 * np.array([7, 12, 17, 22, 27, 32])
    t, u_smol, velocity_smol = load_velocity_data()
    
    boundary = [10, 0]
    
    oversampling_size = 30
    oversampling_x = np.linspace(x[0], x[-1], oversampling_size)  
    
    def oversampling_approx(x_init, x_new, row, order = 4):
        BI = BesselInterpolator(x_init, row, max_order = order)
        return BI.approximate(x_new)    
    
    print(x, oversampling_x, u_smol.shape)
    u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
                   for idx in range(u_smol.shape[0])])
    
    velocity = np.vstack([oversampling_approx(x, oversampling_x, velocity_smol[idx, :]) 
                   for idx in range(velocity_smol.shape[0])])
    
    grids = np.meshgrid(t, oversampling_x, indexing = 'ij')
    dimensionality = u.ndim
    


    epde_search_obj = epde_alg.epde_search()
    epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    velocity_tokens = Cache_stored_tokens('velocity', boundary, ['v'], {'v':velocity}, 
                                          {'power' : (1, 1)}, {'power' : 1})
    
    
    inv_fun_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality - 1)}

    custom_inv_fun_tokens = Custom_tokens(token_type = 'inverse', # Выбираем название для семейства токенов - обратных функций.
                                       token_labels = ['1/x_[dim]',], # Задаём названия токенов семейства в формате python-list'a.
                                                                     # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                       evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                       params_ranges = inv_fun_params_ranges, # Используем заявленные диапазоны параметров
                                       params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
                                                                      # (равенство при лишь полном совпадении дискретных параметров)
                                                                      # нас устраивает.
                                                                     
    custom_grid_tokens = Cache_stored_tokens(token_type = 'grid', 
                                       boundary = boundary,
                                       token_labels = ['t', 'r'], 
                                       token_tensors={'t' : grids[0], 'r' : grids[1]},
                                       params_ranges = {'power' : (1, 1)},
                                       params_equality_ranges = None)


    epde_search_obj.set_moeadd_params(population_size=3, training_epochs = 5)
    
    epde_search_obj.fit(data = u, max_deriv_order=(1, 2), boundary=boundary, equation_terms_max_number = 4,
                        equation_factors_max_number = 2, deriv_method='poly', eq_sparsity_interval = (1e-6, 3.0),
                        deriv_method_kwargs = {'smooth': True, 'grid': grids, 'sigma' : 2}, coordinate_tensors = grids,
                        additional_tokens = [custom_grid_tokens, custom_inv_fun_tokens, velocity_tokens],
                        memory_for_cache=25, prune_domain = True,
                        division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 4.)))

    res = epde_search_obj.equation_search_results(level_num = 1)
    return res


if __name__ == '__main__':        
    output = velocity_model(t_interval = [315, -1])
    # output = velocity_fair()
