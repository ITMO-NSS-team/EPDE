#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:03:57 2022

@author: maslyaev
"""
import os
import matplotlib.pyplot as plt
import pickle

import numpy as np
import epde.interface.interface as epde_alg
import pandas as pd

from epde.interface.equation_translator import Coeff_less_equation
from epde.interface.prepared_tokens import Custom_tokens, Cache_stored_tokens
from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator

from epde.prep.interpolation_oversampling import BesselInterpolator

def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1)), xlabel = '', ylabel = '', figsize=(8,6), filename = None, title = ''):
    y, x = np.meshgrid(np.linspace(area[0][0], area[0][1], Matrix.shape[0]), np.linspace(area[1][0], area[1][1], Matrix.shape[1]))
    fig, ax = plt.subplots(figsize = figsize)
    plt.xlabel(xlabel)
    ax.set(ylabel=ylabel)
    ax.xaxis.labelpad = -10
    if interval:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=interval[0], vmax=interval[1])    
    else:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=min(-abs(np.max(Matrix)), -abs(np.min(Matrix))),
                          vmax=max(abs(np.max(Matrix)), abs(np.min(Matrix)))) 
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.title(title)
    if filename is not None: plt.savefig(filename + '.png', format='png')    
    plt.show()

def run_epde_on_noise(filename, x = np.linspace(0.5, 10, 20) * 1e-3, 
                      prune = True, epochs = 3600):
    sep = '/'
    
    t_max = 1900
    t_min = 100 
    file = np.loadtxt(filename, skiprows = 3)
    print(filename, x.size)
    print(file.shape)
    print(file[0])
    # x = 
    t = file[t_min:t_max, 0]
    # grids = np.meshgrid(t, x, indexing = 'ij')
    u_smol = file[t_min:t_max, 1:]
    filename_temp = filename.split(sep)
    u_smol = u_smol[t_min:t_max, :]; t = t[t_min:t_max]
    # file = np.loadtxt('/home/maslyaev/epde/EPDE_stable/tests/cylinder/data/Data_32_points_.dat', 
    #                   delimiter=' ', usecols=range(33))
    
    oversampling_size = 30
    oversampling_x = np.linspace(x[0], x[-1], oversampling_size)  
    
    def oversampling_approx(x_init, x_new, row, order = 4):
        BI = BesselInterpolator(x_init, row, max_order = order)
        return BI.approximate(x_new)
    
    print(x, oversampling_x, u_smol.shape)
    u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
                   for idx in range(u_smol.shape[0])])
    # u = u.T
    
    
    # x = np.linspace(0.5, 16, 32)
    # t = file[:t_max, 0]
    grids = np.meshgrid(t, oversampling_x, indexing = 'ij')
    print(u.shape, grids[0].shape)
    
    Heatmap(u, title=filename_temp[-1], interval = (u.min(), u.max()),
            filename = sep.join(filename_temp[:-1]) + sep + filename_temp[-1].split('.')[0])        

    boundary = 8
    
    dimensionality = u.ndim
    epde_search_obj = epde_alg.epde_search()
    epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

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
                        additional_tokens = [custom_grid_tokens, custom_inv_fun_tokens], 
                        # method='poly', method_kwargs = {'smooth' : True, 'sigma' : 5,
                        #                                 'mp_poolsize' : 4, 'max_order' : 2, 
                        #                                 'polynomial_window' : 5, 'poly_order' : 7}, 
                        method='ANN', method_kwargs = {'epochs_max' : epochs},                        
                        memory_for_cache=5, prune_domain = prune,
                        division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 6.)))
    
    lp_terms = [['1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], ['d^2u/dx2^2{power: 1}']]
    rp_term = ['du/dx1{power: 1}',]
    test = Coeff_less_equation(lp_terms, rp_term, epde_search_obj.pool)
    
    def map_to_equation(equation, function, function_kwargs = dict()):
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        return function(np.dot(features, equation.weights_final[:-1]) + 
                              np.full(target.shape, equation.weights_final[-1]) - 
                              target, **function_kwargs)    
    print('---------------------------------------------')
    print(filename)
    print(test.equation.text_form)
    print(map_to_equation(test.equation, np.mean))
    return test.equation

def iterate_over_files():
    equations = []
    
    x = np.array([0.7, 1.2, 1.7, 2.2, 2.7, 3.2]) * 1e-3
    
    directory = '/home/maslyaev/epde/EPDE_stable/tests/cylinder/data/Noise_3'
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".dat"):
            try:
                file_desc_eq = run_epde_on_noise(directory + '/' + filename, x = x)
            except:
                print("Exception met while discovering equation on pruned domain")
                file_desc_eq =  run_epde_on_noise(directory + '/' + filename, x = x, prune=False)                
            res_temp = (filename, file_desc_eq.text_form)      
            
            equations.append(res_temp)

            fname = '/home/maslyaev/epde/EPDE_stable/tests/cylinder/' + filename + '.pickle'
            file_to_store = open(fname, "wb")
            pickle.dump(res_temp, file_to_store)
            file_to_store.close()
            
def iterate_over_epochs(epochs = np.arange(1200, 5500, 600), repeats = 4,
                        file = '/home/maslyaev/epde/EPDE_stable/tests/cylinder/data/Noise_3/Data_6_random_0.dat'):
    equations = []
    x = np.array([0.7, 1.2, 1.7, 2.2, 2.7, 3.2]) * 1e-3

    for epoch_limit in epochs:
        for repeat in range(repeats):
            try:
                file_desc_eq = run_epde_on_noise(file, 
                                                 x = x, epochs = epoch_limit)
            except:
                print("Exception met while discovering equation on pruned domain")
                file_desc_eq = run_epde_on_noise(file, 
                                                 x = x, prune=False, epochs = epoch_limit)                
            res_temp = (epoch_limit, file_desc_eq.text_form)      
            
            equations.append(res_temp)
    
            fname = file + '_' + str(epoch_limit) + '_' + str(repeat) + '.pickle'
            file_to_store = open(fname, "wb")
            pickle.dump(res_temp, file_to_store)
            file_to_store.close()
            
if __name__ == "__main__":
    iterate_over_epochs()