#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:47:47 2022

@author: maslyaev
"""

import numpy as np

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens

if __name__ == '__main__':
    '''
    Подгружаем данные, содержащие временные ряды динамики "вида-охотника" и "вида-жертвы"
    '''
    t = np.load('/home/maslyaev/epde/EPDE_rework/projects/hunter-prey/t.npy')
    data = np.load('/home/maslyaev/epde/EPDE_rework/projects/hunter-prey/data.npy')
    x = data[:, 0]; y = data[:, 1]
        
    dimensionality = x.ndim - 1
    
    '''
    Подбираем Парето-множество систем дифф. уравнений.
    '''
    epde_search_obj = epde_alg.epde_search(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                           coordinate_tensors = [t,], verbose_params = {'show_moeadd_epochs' : True})    
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly', 
                                     preprocessor_kwargs={'use_smoothing' : False})
    popsize = 7
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=100)
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    epde_search_obj.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,),
                        equation_terms_max_number=3, data_fun_pow = 1, additional_tokens=[trig_tokens,], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-10, 1e-4), coordinate_tensors=[t, ])
    '''
    Смотрим на найденное Парето-множество, 
    
    "идеальное уравнение" имеет вид:
     / -1.3048888807580532 * u{power: 1.0} * v{power: 1.0} + 0.3922851274813135 * u{power: 1.0} + -0.0003917278536547386 = du/dx1{power: 1.0}
     \ -0.9740492564964498 * v{power: 1.0} + 0.9717873909925121 * u{power: 1.0} * v{power: 1.0} + 0.0003500773115704403 = dv/dx1{power: 1.0}
    {'terms_number': {'optimizable': False, 'value': 3}, 'max_factors_in_term': {'optimizable': False, 'value': {'factors_num': [1, 2], 'probas': [0.8, 0.2]}}, 
     ('sparsity', 'u'): {'optimizable': True, 'value': 0.00027172388370453704}, ('sparsity', 'v'): {'optimizable': True, 'value': 0.00019292375116125682}} , with objective function values of [0.2800438  0.18041074 4.         4.        ]         
    '''
    epde_search_obj.equation_search_results(only_print = True, level_num = 1)
    
    res = epde_search_obj.equation_search_results(only_print = False, level_num = 1)
    sys = res[0][1]
    sys.text_form
    
    '''
    Решаем уравнение (систему уравнений) при помощи метода .predict(...)
    '''
    predictions = epde_search_obj.predict(system = sys, boundary_conditions=None)