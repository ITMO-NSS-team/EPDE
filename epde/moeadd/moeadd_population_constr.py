#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:57:44 2022

@author: maslyaev
"""

import numpy as np
from typing import Callable

from epde.structure.main_structures import SoEq


def set_argument(var, fun_kwargs, base_value):
    try:
        res = fun_kwargs[var]
    except KeyError:
        res = base_value
    return res

class SystemsPopulationConstructor(object):
    def __init__(self, pool, terms_number : int = 8, max_factors_in_term : int = 2, 
                 obj_functions : Callable = None, sparsity_interval : tuple = (0, 1)):
        self.pool = pool; self.terms_number = terms_number
        self.max_factors_in_term = max_factors_in_term 
        self.vars_demand_equation = [family.ftype for family in self.pool.families_demand_equation]
        self.sparsity_interval = sparsity_interval

    def create(self, **kwargs): # Дописать
        sparsity = set_argument('sparsity', kwargs, np.power(np.e, np.random.uniform(low = np.log(self.sparsity_interval[0]),
                                                                      high = np.log(self.sparsity_interval[1]),
                                                                      size = len(self.vars_demand_equation))))
        terms_number = set_argument('terms_number', kwargs, self.terms_number)
        max_factors_in_term = set_argument('max_factors_in_term', kwargs, self.max_factors_in_term)
        
        print(f'Creating new equation, sparsity value {sparsity}')
        metaparameters = {'terms_number'        : {'optimizable' : False, 'value' : terms_number},
                          'max_factors_in_term' : {'optimizable' : False, 'value' : max_factors_in_term}}
        for idx, variable in enumerate(self.vars_demand_equation):
            metaparameters[('sparsity', variable)] = {'optimizable' : True, 'value' : sparsity[idx]}

        created_solution = SoEq(pool = self.pool, metaparameters = metaparameters)

        try:
            created_solution.set_objective_functions(kwargs['obj_funs'])
        except KeyError:
            created_solution.use_default_objective_function()

        created_solution.create_equations()
        # print(f'Initial solution for MOEADD: {created_solution.text_form}')

        return created_solution