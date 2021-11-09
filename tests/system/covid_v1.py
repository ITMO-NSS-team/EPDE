#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:50:09 2021

@author: maslyaev
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:45:14 2021
@author: mike_ubuntu
"""
#v2 - visualisation elements for testing purposes

import numpy as np
import matplotlib
import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import Trigonometric_tokens

import matplotlib.pyplot as plt

if __name__ == '__main__':


    #u = np.load('Test_data/fill366.npy') # loading data with the solution of ODE
    #np.savetxt('Test_data/input.txt', u)

    #Flu data

    # exp_name = "flu"
    # u = np.loadtxt('input\\prev_weekly.txt')  # loading data with the solution of ODE
    # tick_title = "Week"

    exp_name = "flu"
    u = np.loadtxt('/home/maslyaev/epde/EPDE_stable/tests/system/Test_data/prev_daily.txt')
    tick_title = "Day"

    N = len(u)
    print(N)

    t = np.arange(0,N)

    matplotlib.rcParams.update({'font.size': 20})
    plt.rcParams["figure.figsize"] = [14, 7]

    plt.plot(t,u, 'bo')
    plt.xlabel(tick_title)
    plt.ylabel("Number of registered cases")
    # plt.savefig(exp_name+"_"+tick_title+"_input.png", dpi=150, bbox_inches='tight')
    # plt.savefig(exp_name + "_" + tick_title + "_input.pdf", dpi=150, bbox_inches='tight')
    # plt.show()

    # Trying to create population for mulit-objective optimization with only 
    # derivatives as allowed tokens. Spoiler: only one equation structure will be 
    # discovered, thus MOO algorithm will not be launched.

    dimensionality = u.ndim # - 1

    epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter = 100, dimensionality=dimensionality) #verbose_params={'show_moeadd_epochs' : True}
    
    add_tokens = Trigonometric_tokens(freq = (0.95, 1.05))

    epde_search_obj.set_moeadd_params(population_size=1)

    # Вариант запуска по дефолту: equation_terms_max_number=2 - суммарно два слагаемых в левой и правой части (без учёта констант)
    # data_fun_pow = 1 - максимальная степень одного множителя (переменной) в слагаемом
    # equation_factors_max_number =2 - максимальное количество множителей в слагаемом

    # epde_search_obj.fit(data = u, max_deriv_order=(1,), boundary=(10,), equation_terms_max_number = 2,
    #                     equation_factors_max_number = 2, deriv_method='poly', eq_sparsity_interval = (1e-7, 10),
    #                     deriv_method_kwargs = {'smooth' : False, 'grid' : [t,]}, coordinate_tensors = [t,])

    # Вариант 2: equation_terms_max_number=3
    # data_fun_pow = 2
    # equation_factors_max_number = 2

    epde_search_obj.fit(data=u, max_deriv_order=(1,), boundary=(10,), equation_terms_max_number=3, data_fun_pow = 2,
                        equation_factors_max_number=2, deriv_method='poly', eq_sparsity_interval=(1e-7, 10),
                        deriv_method_kwargs={'smooth': False, 'grid': [t, ]}, coordinate_tensors=[t, ])

    res = epde_search_obj.equation_search_results(only_print = False, level_num = 1) # showing the Pareto-optimal set of discovered equations 

    solver_forms = []    
    grids = []
    bconds = []

    for level in res:
        for eq in level:
            solver_form, grid, bcond = eq.solver_params()
            solver_forms.append(solver_form)
            grids.append(grid)
            bconds.append(bcond)            
    epde_search_obj.equation_search_results(only_print = True, level_num = 1)            