#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:47:47 2022

@author: maslyaev
"""

import numpy as np
import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import CustomTokens, CacheStoredTokens
from epde.interface.prepared_tokens import TrigonometricTokens

if __name__ == '__main__':
    t = np.load('projects/hunter-prey/t.npy')
    data = np.load('projects/hunter-prey/data.npy')
    x = data[:, 0]; y = data[:, 1]
    
    dimensionality = x.ndim - 1
    epde_search_obj = epde_alg.epde_search(use_solver = False, dimensionality = dimensionality, coordinate_tensors = [t,],
                                           verbose_params = {'show_moeadd_epochs' : True})    
    
    popsize = 7
    epde_search_obj.set_moeadd_params(population_size = popsize, weight_num = popsize, training_epochs=25)
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    epde_search_obj.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,), boundary=(10,), 
                        equation_terms_max_number=4, data_fun_pow = 1, additional_tokens=[trig_tokens,], 
                        equation_factors_max_number=2, deriv_method='poly', eq_sparsity_interval=(1e-4, 1e1),
                        deriv_method_kwargs={'smooth': False, 'grid': [t, ]}, coordinate_tensors=[t, ])    
    epde_search_obj.equation_search_results(only_print = True, level_num = 1)