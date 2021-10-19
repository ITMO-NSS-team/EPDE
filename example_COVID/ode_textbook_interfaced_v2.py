#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:45:14 2021

@author: mike_ubuntu
"""
#v2 - visualisation elements for testing purposes

import numpy as np
import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import Trigonometric_tokens

import matplotlib.pyplot as plt

if __name__ == '__main__':


    #u = np.load('Test_data/fill366.npy') # loading data with the solution of ODE
    #np.savetxt('Test_data/input.txt', u)

    u = np.loadtxt('Test_data\\input.txt')  # loading data with the solution of ODE

    N = len(u)
    print(N)

    t = np.linspace(0, 4 * np.pi, N)  # setting time axis, corresonding to the solution of ODE
    #t = np.linspace(0, 1000, N)  # setting time axis, corresonding to the solution of ODE

    plt.plot(t,u)
    plt.show()

    # Trying to create population for mulit-objective optimization with only 
    # derivatives as allowed tokens. Spoiler: only one equation structure will be 
    # discovered, thus MOO algorithm will not be launched.

    dimensionality = u.ndim # - 1

    epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter = 100, dimensionality=dimensionality,
                                           verbose_params={'show_moeadd_epochs' : True})
    
    add_tokens = Trigonometric_tokens(freq = (0.95, 1.05))

    epde_search_obj.set_moeadd_params(population_size=1)
    epde_search_obj.fit(data = u, max_deriv_order=(1,), boundary=(10,), equation_terms_max_number = 2,
                        equation_factors_max_number = 2, deriv_method='poly', eq_sparsity_interval = (1e-7, 10),
                        deriv_method_kwargs = {'smooth' : False, 'grid' : [t,]}, coordinate_tensors = [t,])    
    
    epde_search_obj.equation_search_results(only_print = True, level_num = 1) # showing the Pareto-optimal set of discovered equations 
