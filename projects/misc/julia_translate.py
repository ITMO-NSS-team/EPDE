#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:07:44 2023

@author: maslyaev
"""

import numpy as np

from epde.interface.prepared_tokens import Custom_tokens
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch

if __name__ == "__main__":
    t = np.load('projects/misc/data/t.npy')
    u = np.load('projects/misc/data/u.npy')
    dudt = np.load('projects/misc/data/dudt.npy')
    
    text_form = ("1 + 0.6305375487538072 * sine{power : 1.0, freq : 0.16, phase : 0.42} du/dx1{power: 1.0} + \
                  0.00021205339758654683 * sine{power : 1.0, freq : 0.03, phase : 0.79} du/dx1{power: 1.0} + \
                  0.00032221 * pow{power : -0.970413802, dim : 0} * du/dt{power: 1.0} + \
                  0.02098415305452864 * sine{power : 1.0, freq : 0.16, phase : 0.72} * du/dx1{power: 1.0} + \
                  0.00013769133379913995 * sine{power : 1.0, 0.58t + 0.9pi} * du/dt{power: 1.0} + \
                  0.00021489681489331796 * sine{power : 1.0, 0.14t + 0.45pi} * du/dt{power: 1.0} + \
                  0.6198270382749153 * sine{power : 1.0, -0.16t + 0.18pi} * u{power: 1.0} + \
                  -0.003322268 * pow{power : 0.81227378} * u{power: 1.0} + \
                  8.514359191989314e-05 * sine{power : 1.0, 1.13t + 0.31pi} u{power: 1.0} + \
                  7.389674383847592e-05 * sine{power : 1.0, 0.62t + 0.41pi} u{power: 1.0} + \
                  9.393134067153445e-05 * sine{power : 1.0, 0.79t + 0.31pi} u{power: 1.0} + \
                  0.002344347(t**0.952463034) * u + \
                  5.592239039367684e-05 * sine{power : 1.0, freq : -2.24, phase : 0.65pi} * u + \
                  = 0.001494062(t**-0.159960063) u"
                  
    custom_pow_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], kwargs['power']) 
    custom_pow_fun_evaluator = CustomEvaluator(custom_pow_eval_fun, eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    pow_params_ranges = {'power' : (-2., 2.), 'dim' : (0, 0)}
    
    custom_pow_fun_tokens = Custom_tokens(token_type = 'pow', # Выбираем название для семейства токенов - обратных функций.
                                          token_labels = ['pow',], # Задаём названия токенов семейства в формате python-list'a.
                                          evaluator = custom_pow_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                          params_ranges = pow_params_ranges, # Используем заявленные диапазоны параметров
                                          params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
