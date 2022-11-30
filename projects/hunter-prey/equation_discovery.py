#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:47:47 2022

@author: maslyaev
"""

import numpy as np
import matplotlib.pyplot as plt

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import CustomTokens, CacheStoredTokens
from epde.interface.prepared_tokens import TrigonometricTokens

def plot_pareto_exp():
    points = np.array([[3.11424976 + 4.61797389, 5],
                       [0.18041074 + 5.07648402, 6],
                       [2.05781832 + 0.2800438, 7], 
                       [0.18041074 + 0.2800438, 8],
                       [0.17038125 + 0.2714551, 10]])
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(points[:, 1], points[:, 0], 'k')
    ax.set_xlabel('Complexity, active tokens')
    ax.set_ylabel('Modelling error')
    plt.savefig(fname = 'projects/hunter-prey/pareto.png', dpi = 200)

if __name__ == '__main__':
    t = np.load('/home/maslyaev/epde/EPDE_rework/projects/hunter-prey/t.npy')
    data = np.load('/home/maslyaev/epde/EPDE_rework/projects/hunter-prey/data.npy')
    x = data[:, 0]; y = data[:, 1]
        
    dimensionality = x.ndim - 1
    # for i in range(10000):
    epde_search_obj = epde_alg.epde_search(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                           coordinate_tensors = [t,], verbose_params = {'show_moeadd_epochs' : True})    
    
    popsize = 7
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=50)
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    epde_search_obj.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,),
                        equation_terms_max_number=3, data_fun_pow = 1, additional_tokens=[trig_tokens,], 
                        equation_factors_max_number=factors_max_number, deriv_method='poly', 
                        eq_sparsity_interval=(1e-10, 1e-2),
                        deriv_method_kwargs={'smooth': False, 'grid': [t, ]}, coordinate_tensors=[t, ])    
    epde_search_obj.equation_search_results(only_print = True, level_num = 1)