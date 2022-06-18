#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 14:04:54 2022

@author: maslyaev
"""

import pandas as pd
import numpy as np
import matplotlib
import epde.interface.interface as epde_alg
import os
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens, ExternalDerivativesTokens

import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    data_table = pd.read_excel('projects/epde_ts_pred/prepared_data/data_E1.xlsx', skiprows = 3)
    data = data_table.to_numpy()
    
    t = data[:, 0]
    T0 = data[:, 1]; T1 = data[:, 2]; phi = data[:, 3]
    
    dimensionality = t.ndim # - 1
    boundary = 10
    
    
    epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter = 200, dimensionality=dimensionality) #verbose_params={'show_moeadd_epochs' : True}
    epde_search_obj.set_moeadd_params(population_size=20)
    
    T1_derivs = ExternalDerivativesTokens('T1 derivs', boundary = boundary, time_axis = 0, base_token_label = 'T1',
                                          token_tensor = T1, max_orders = 2, deriv_method='poly', 
                                          deriv_method_kwargs = {'smooth': False, 'grid': [t, ]}, 
                                          params_ranges = {'power' : (1, 1)},
                                          params_equality_ranges = None, meaningful=True)
    
    power_tokens = CacheStoredTokens(token_type = 'power',
                                     boundary = boundary,
                                     token_labels = ['P'],
                                     token_tensors={'P' : phi},
                                     params_ranges = {'power' : (1, 1)},
                                     params_equality_ranges = None, meaningful=True)
    
    epde_search_obj.fit(data=T0, max_deriv_order=(3,), boundary=boundary, equation_terms_max_number=5, 
                        data_fun_pow = 2, variable_names = ['T0'], equation_factors_max_number=2, 
                        deriv_method='poly', eq_sparsity_interval=(1e-7, 1000), 
                        additional_tokens=[T1_derivs, power_tokens],
                        deriv_method_kwargs={'smooth': False, 'grid': [t, ]}, coordinate_tensors=[t, ])    
    res = epde_search_obj.equation_search_results(level_num = 1)    