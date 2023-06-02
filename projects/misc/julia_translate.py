#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:07:44 2023

@author: maslyaev
"""

import numpy as np
import torch

from epde.interface.prepared_tokens import CustomTokens, PhasedSine1DTokens, ConstantToken
from epde.interface.equation_translator import translate_equation
from epde.evaluators import CustomEvaluator, simple_function_evaluator
from epde.interface.interface import EpdeSearch

from epde.interface.solver_integration import BoundaryConditions, BOPElement

SOLVER_STRATEGY = 'autograd'

if __name__ == "__main__":
    t = np.load('projects/misc/data/t.npy')
    u = np.load('projects/misc/data/u.npy')
    dudt = np.load('projects/misc/data/du.npy')
    
    text_form_old =  "0.6305375487538072 * sine{power : 1.0, freq : 0.16, phase : 0.42} * du/dx1{power: 1.0} + \
                  0.00021205339758654683 * sine{power : 1.0, freq : 0.03, phase : 0.79} * du/dx1{power: 1.0} + \
                  0.00032221 * pow{power : -0.970413802, dim : 0} * du/dx1{power: 1.0} + \
                  0.02098415305452864 * sine{power : 1.0, freq : 0.16, phase : 0.72} * du/dx1{power: 1.0} + \
                  0.00013769133379913995 * sine{power : 1.0, freq : 0.58, phase : 0.9} * du/dx1{power: 1.0} + \
                  0.00021489681489331796 * sine{power : 1.0, freq : 0.14, phase : 0.45} * du/dx1{power: 1.0} + \
                  0.6198270382749153 * sine{power : 1.0, freq : -0.16, phase : 0.18} * u{power: 1.0} + \
                  -0.003322268 * pow{power : 0.81227378, dim : 0} * u{power: 1.0} + \
                  8.514359191989314e-05 * sine{power : 1.0, freq : 1.13, phase : 0.31} * u{power: 1.0} + \
                  7.389674383847592e-05 * sine{power : 1.0, freq : 0.62, phase : 0.41} * u{power: 1.0} + \
                  9.393134067153445e-05 * sine{power : 1.0, freq : 0.79, phase : 0.31} * u{power: 1.0} + \
                  0.002344347 * pow{power : 0.952463034, dim : 0} * u{power: 1.0} + \
                  5.592239039367684e-05 * sine{power : 1.0, freq : -2.24, phase : 0.65} * u{power: 1.0} + 0.0\
                  = const{power : 1, value : 0.001494062} * pow{power : -0.159960063, dim : 0} * u{power: 1.0}"

    text_form_old2 = "0.7002306443567028 * sine{power : 1.0, freq : 0.16, phase : 0.48} * du/dx1{power: 1.0} + \
                 0.006205713520583175 * sine{power : 1.0, freq : 0.16, phase : 0.15} * du/dx1{power: 1.0} + \
                 0.6926349136625439 * sine{power : 1.0, freq : 0.16, phase : 0.22} * u{power: 1.0} + \
                 5.109782570791753e-05 * sine{power : 1.0, freq : 2.29, phase : 0.13} * u{power: 1.0} + \
                 0.00016343939451610628 * sine{power : 1.0, freq : 0.02, phase : 0.9} * u{power: 1.0} + 0.0 \
                 = const{power : 1, value : -1.}"
                 
    
                 
    custom_pow_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], kwargs['power']) 
    custom_pow_fun_evaluator = CustomEvaluator(custom_pow_eval_fun, eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    pow_params_ranges = {'power' : (-2., 2.), 'dim' : (0, 0)}
    
    custom_pow_fun_tokens = CustomTokens(token_type = 'pow', # Выбираем название для семейства токенов - обратных функций.
                                         token_labels = ['pow',], # Задаём названия токенов семейства в формате python-list'a.
                                         evaluator = custom_pow_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                         params_ranges = pow_params_ranges, # Используем заявленные диапазоны параметров
                                         params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 

    phased_sine = PhasedSine1DTokens(freq = (-10, 10))
    consts = ConstantToken()
    epde_search_obj = EpdeSearch(use_solver = False, dimensionality = 0, boundary = 10,
                                 coordinate_tensors = [t,])
    epde_search_obj.create_pool(data = u, derivs = [dudt,], additional_tokens = [phased_sine, custom_pow_fun_tokens, consts])
    eq_translated = translate_equation(text_form, epde_search_obj.pool)
    
    def get_ode_bop(key, var, grid_loc, value):
        bop = BOPElement(axis = 0, key = key, term = [None], power = 1, var = var)
        bop_grd_np = np.array([[grid_loc,]])
        bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
        bop.values = torch.from_numpy(np.array([[value,]])).float()
        return bop
                
            
    bop_x = get_ode_bop('u', 0, t[0], u[0])
    pred_u_v = epde_search_obj.predict(system=eq_translated, boundary_conditions=[bop_x(),], 
                                       grid = [t,], strategy=SOLVER_STRATEGY)
    