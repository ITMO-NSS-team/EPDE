#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:47:47 2022

@author: maslyaev
"""

import numpy as np

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens

import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

def write_pareto(dict_of_exp):
    """
    Writes Pareto front solutions to text files for each experiment.
    
        This function iterates through a dictionary of experimental results, extracting
        Pareto front solutions and writing them to separate text files. The filename
        is derived from the experiment's parameters. Each iteration's Pareto front
        is written with equation strings to the file. This is done to store the discovered
        equation structures that represent the best trade-offs between different objectives
        at each stage of the evolutionary search process, allowing for later analysis and
        selection of the most suitable models.
        
        Args:
            dict_of_exp: Dictionary where keys are tuples representing experiment
                parameters and values are lists of Pareto front solutions for each
                iteration of the experiment.
        
        Returns:
            None. This method writes data to files and does not return any value.
    """
    for key, item in dict_of_exp.items():
        test_key = str(key[0]).replace('.', '_') + '__' + str(key[1]).replace('.', '_')
        with open('/home/maslyaev/epde/EPDE_main/projects/hunter-prey/param_var/'+test_key+'.txt', 'w') as f:
            for iteration in range(len(item)):
                f.write(f'Iteration {iteration}\n\n')
                for ind in [pareto.text_form for pareto in item[iteration][0]]:
                    f.write(ind + '\n\n')

if __name__ == '__main__':
    '''
    Подгружаем данные, содержащие временные ряды динамики "вида-охотника" и "вида-жертвы"
    '''
    try:
        t_file = os.path.join(os.path.dirname( __file__ ), 'data/t.npy')
        t = np.load(t_file)
    except FileNotFoundError:
        t_file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/data/t.npy'
        t = np.load(t_file)
    
    try:
        data_file =  os.path.join(os.path.dirname( __file__ ), 'data/lorenz.npy')
        data = np.load(data_file)
    except FileNotFoundError:
        data_file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/data/lorenz.npy'
        data = np.load(data_file)
        
    end = 10000
    t = t[:end]
    x = data[:end, 0]; y = data[:end, 1]; z = data[:end, 2]
        
    dimensionality = x.ndim - 1
    
    '''
    Подбираем Парето-множество систем дифф. уравнений.
    '''
    epde_search_obj = epde_alg.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                           coordinate_tensors = [t,])    
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True
                                     preprocessor_kwargs={})
    popsize = 20
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=40)
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    epde_search_obj.fit(data=[x, y, z], variable_names=['u', 'v', 'w'], max_deriv_order=(1,),
                        equation_terms_max_number=5, data_fun_pow = 1, additional_tokens=[trig_tokens,], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-10, 1e-4))
    '''
    Смотрим на найденное Парето-множество, 
    
    "идеальное уравнение" имеет вид:
     / -1.3048888807580532 * u{power: 1.0} * v{power: 1.0} + 0.3922851274813135 * u{power: 1.0} + -0.0003917278536547386 = du/dx1{power: 1.0}
     \ -0.9740492564964498 * v{power: 1.0} + 0.9717873909925121 * u{power: 1.0} * v{power: 1.0} + 0.0003500773115704403 = dv/dx1{power: 1.0}
    {'terms_number': {'optimizable': False, 'value': 3}, 'max_factors_in_term': {'optimizable': False, 'value': {'factors_num': [1, 2], 'probas': [0.8, 0.2]}}, 
     ('sparsity', 'u'): {'optimizable': True, 'value': 0.00027172388370453704}, ('sparsity', 'v'): {'optimizable': True, 'value': 0.00019292375116125682}} , with objective function values of [0.2800438  0.18041074 4.         4.        ]         
    '''
    epde_search_obj.equation_search_results(only_print = True, num = 1)
    
    res = epde_search_obj.equation_search_results(only_print = False, num = 1)
    # sys = res[0][1]
    # sys.text_form
    # pareto_frontiers[(err_factor, sigma)].append(res)
'''
Решаем уравнение (систему уравнений) при помощи метода .predict(...)
'''
# predictions = epde_search_obj.predict(system = sys, boundary_conditions=None)