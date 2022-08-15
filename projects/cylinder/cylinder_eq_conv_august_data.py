#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:11:52 2021

@author: mike_ubuntu
"""
import os
import pandas as pd

import numpy as np
import epde.interface.interface as epde_alg
import pickle

from epde.interface.equation_translator import Coeff_less_equation
from epde.interface.prepared_tokens import CustomTokens, CacheStoredTokens, Velocity_HEQ_tokens, ConstantToken
from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator

from epde.prep.interpolation_oversampling import BesselInterpolator
from epde.parametric.parametric_eq_translator import optimize_parametric_form

def load_data():
    data_raw = np.genfromtxt('/home/maslyaev/epde/EPDE_main/projects/cylinder/data/1_august_data/0_8_W_expD.dat', 
                             delimiter='\t', skip_header = 1, encoding='latin-1', dtype=None)

    replacer = lambda a: a.replace(',', '.')
    repl_v = np.vectorize(replacer)
    idx = tuple(np.arange(0, np.where(data_raw == '')[0][0], 250))

    data = repl_v(data_raw[idx, :]).astype(np.float32)

    t = data[:, 0]; x = np.array([0.3, 1.1, 1.6]) * 1e-3; u = data[:, (2, 5, 8)]
    return t, x, u


def get_fd(u, x, t):
    tt, xx = np.meshgrid(t, x, indexing = 'ij')    
    dudx0 = (u[1:-1, 1] - u[1:-1, 0])/(x[1] - x[0])
    dudx1 = (u[1:-1, 2] - u[1:-1, 0])/(x[2] - x[0])
    dudx2 = (u[1:-1, 2] - u[1:-1, 1])/(x[2] - x[1])
    
    dudx = np.array([dudx0, dudx1, dudx2]).T
    dudx_by_r = np.multiply(dudx, 1./xx[1:-1, ...])

    dudt = (u[2:, ...] - u[:-2, ...]) / (tt[2:, ...] - tt[:-2, ...])
    return dudt, dudx_by_r


def load_interp_data():
    t = np.load('/home/maslyaev/epde/EPDE_for_roman/projects/cylinder/data/1_august_data/rbf_interp_t_dim.npy')
    x = np.load('/home/maslyaev/epde/EPDE_for_roman/projects/cylinder/data/1_august_data/rbf_interp_x_dim.npy')
    data = np.load('/home/maslyaev/epde/EPDE_for_roman/projects/cylinder/data/1_august_data/rbf_interp_u.npy')
    return t, x, data

def run_parametric(t_interval = None, oversample = False):
    # t, x, u_smol = load_data()
    # if t_interval is not None:
    #     mask = (t > t_interval[0]) & (t < t_interval[1])
    #     t = t[mask]
    #     u_smol = u_smol[mask, ...]
    # if oversample:
    #     oversampling_size = 30
    #     oversampling_x = np.linspace(x[0], x[-1], oversampling_size)  
        
    #     def oversampling_approx(x_init, x_new, row, order = 4):
    #         BI = BesselInterpolator(x_init, row, max_order = order)
    #         return BI.approximate(x_new)
        
    #     u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
    #                    for idx in range(u_smol.shape[0])])
    # else:
    #     oversampling_x = x
    #     u = u_smol
    t, x, u = load_interp_data()
    
    
    # grids = np.meshgrid(t, oversampling_x, indexing = 'ij')
    grids = [t, x]

    boundary = [10, 4]

    dimensionality = u.ndim
    
    epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter = 100, dimensionality=dimensionality,
                                           verbose_params={'show_moeadd_epochs' : True}, boundary=boundary, 
                                           memory_for_cache=25, coordinate_tensors=grids)
    # epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)
                                                                     
    # custom_grid_tokens = CacheStoredTokens(token_type = 'grid', 
    #                                          boundary = boundary,
    #                                          token_labels = ['t', 'r'], 
    #                                          token_tensors={'t' : grids[0], 'r' : grids[1]},
    #                                          params_ranges = {'power' : (1, 1)},
    #                                          params_equality_ranges = None)

    velocity_params = [(-0.0390729, 0.0390729), (-2., 2.), (-100., 100.), 
                       (-0.00294, 0.00294), (-0.218, 0.218), (-20., 20.), 
                       (-0.00003102, 0.00003102), (-0.001, 0.001), (-0.01, 0.01),
                       (-3., 3.), (-80., 80.), (-50000., 50000.), 
                       (-0.0006, 0.0006), (-0.04, 0.04), (-5., 5.)] # Insert parameter ranges
    velocity_tokens = Velocity_HEQ_tokens(velocity_params)

    const_tokens = ConstantToken(values_range = (-10, 10))

    # trig_tokens = TrigonometricTokens(freq = (0.95, 1.05))

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, 
                                               eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    inv_fun_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality - 1)}

    custom_inv_fun_tokens = CustomTokens(token_type = 'inverse', # Выбираем название для семейства токенов - обратных функций.
                                          token_labels = ['1/x_[dim]',], # Задаём названия токенов семейства в формате python-list'a.
                                                                      # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                          evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                          params_ranges = inv_fun_params_ranges, # Используем заявленные диапазоны параметров
                                          params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
                                                                      # (равенство при лишь полном совпадении дискретных параметров)
                                                                      # нас устраивает.

    epde_search_obj.create_pool(data = u, max_deriv_order=(1, 2),
                        additional_tokens = [velocity_tokens, const_tokens, custom_inv_fun_tokens],
                        method='poly', method_kwargs = {'smooth' : False, 'sigma' : 3})
    
    velocity_str = 'v{power : 1, p1 : None, p2 : None, p3 : None, p4 : None, p5 : None, p6 : None, p7 : None, p8 : None, p9 : None, p10 : None, p11 : None, p12 : None, p13 : None, p14 : None, p15 : None}'
    equation_form = [['const{power: 1, value : None}', '1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], #'const{power: 1, value : None}', 
                     # ['const{power: 1, value : None}', 'd^2u/dx2^2{power: 1}'], # 'const{power: 1, value : None}', 
                     # [velocity_str, 'du/dx2{power: 1}'], # 'const{power: 1, value : None}', 
                     ['du/dx1{power: 1}']]
    
    random_initial_params = np.array([1e-7, 1e-7] + [np.random.uniform(bounds[0], bounds[1]) for bounds in velocity_params])
    
    return optimize_parametric_form(equation_form, epde_search_obj.pool, 
                                    initial_params = random_initial_params)


def run_LASSO(t_interval = None, oversample = False):
    # t, x, u_smol = load_data()
    # if t_interval is not None:
    #     mask = (t > t_interval[0]) & (t < t_interval[1])
    #     t = t[mask]
    #     u_smol = u_smol[mask, ...]
    # if oversample:
    #     oversampling_size = 30
    #     oversampling_x = np.linspace(x[0], x[-1], oversampling_size)  
        
    #     def oversampling_approx(x_init, x_new, row, order = 4):
    #         BI = BesselInterpolator(x_init, row, max_order = order)
    #         return BI.approximate(x_new)
        
    #     u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
    #                    for idx in range(u_smol.shape[0])])
    # else:
    #     oversampling_x = x
    #     u = u_smol
    t, x, u = load_interp_data()
    
    
    # grids = np.meshgrid(t, oversampling_x, indexing = 'ij')
    grids = [t, x]

    boundary = [10, 4]

    dimensionality = u.ndim
    
    epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter = 100, dimensionality=dimensionality,
                                           verbose_params={'show_moeadd_epochs' : True}, boundary=boundary, 
                                           memory_for_cache=25, coordinate_tensors=grids)
    # epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)
                                                                     
    # custom_grid_tokens = CacheStoredTokens(token_type = 'grid', 
    #                                          boundary = boundary,
    #                                          token_labels = ['t', 'r'], 
    #                                          token_tensors={'t' : grids[0], 'r' : grids[1]},
    #                                          params_ranges = {'power' : (1, 1)},
    #                                          params_equality_ranges = None)

    velocity_params = [(-0.0390729, 0.0390729), (-2., 2.), (-100., 100.), 
                       (-0.00294, 0.00294), (-0.218, 0.218), (-20., 20.), 
                       (-0.00003102, 0.00003102), (-0.001, 0.001), (-0.01, 0.01),
                       (-3., 3.), (-80., 80.), (-50000., 50000.), 
                       (-0.0006, 0.0006), (-0.04, 0.04), (-5., 5.)] # Insert parameter ranges
    velocity_tokens = Velocity_HEQ_tokens(velocity_params)

    const_tokens = ConstantToken(values_range = (-10, 10))

    # trig_tokens = TrigonometricTokens(freq = (0.95, 1.05))

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, 
                                               eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    inv_fun_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality - 1)}

    custom_inv_fun_tokens = CustomTokens(token_type = 'inverse', # Выбираем название для семейства токенов - обратных функций.
                                          token_labels = ['1/x_[dim]',], # Задаём названия токенов семейства в формате python-list'a.
                                                                      # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                          evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                          params_ranges = inv_fun_params_ranges, # Используем заявленные диапазоны параметров
                                          params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
                                                                      # (равенство при лишь полном совпадении дискретных параметров)
                                                                      # нас устраивает.

    epde_search_obj.create_pool(data = u, max_deriv_order=(1, 2),
                        additional_tokens = [velocity_tokens, const_tokens, custom_inv_fun_tokens],
                        method='poly', method_kwargs = {'smooth' : False, 'sigma' : 3})
    
    # velocity_str = 'v{power : 1, p1 : None, p2 : None, p3 : None, p4 : None, p5 : None, p6 : None, p7 : None, p8 : None, p9 : None, p10 : None, p11 : None, p12 : None, p13 : None, p14 : None, p15 : None}'
    # equation_form = [['const{power: 1, value : None}', '1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], #'const{power: 1, value : None}', 
    #                  # ['const{power: 1, value : None}', 'd^2u/dx2^2{power: 1}'], # 'const{power: 1, value : None}', 
    #                  # [velocity_str, 'du/dx2{power: 1}'], # 'const{power: 1, value : None}', 
    #                  ['du/dx1{power: 1}']]
    
    lp_terms = [['1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}']]#, ['d^2u/dx2^2{power: 1}']]
    rp_term = ['du/dx1{power: 1}',]
    return Coeff_less_equation(lp_terms, rp_term, epde_search_obj.pool)
        

if __name__ == "__main__":
    results_new = []
    test_iter_limit = 15
    for test_idx in np.arange(test_iter_limit):
        results_new.append(run_parametric())