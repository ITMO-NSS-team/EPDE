#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:57:44 2022

@author: maslyaev
"""

import numpy as np
from typing import Callable

from epde.structure.main_structures import SoEq

class SystemsPopulationConstructor(object):
    """
    Class for creating population according to the current optimization method/

    Attributes:
        pool (``):
        terms_number (`int`):
        max_factors_in_term (`int`):
        vars_demand_equation (``):
        sparsity_internal (`tuple`): 
    """
    def __init__(self, pool, terms_number : int = 8, max_factors_in_term : int = 2, 
                 obj_functions : Callable = None, sparsity_interval : tuple = (0, 1)):
        self.pool = pool; self.terms_number = terms_number
        self.max_factors_in_term = max_factors_in_term 
        self.vars_demand_equation = [family.ftype for family in self.pool.families_demand_equation]
        self.sparsity_interval = sparsity_interval

    def create(self, **kwargs):
        sparsity = kwargs.get('sparsity', np.exp(np.random.uniform(low = np.log(self.sparsity_interval[0]),
                                                                      high = np.log(self.sparsity_interval[1]),
                                                                      size = len(self.vars_demand_equation))))
        terms_number = kwargs.get('terms_number', self.terms_number)
        max_factors_in_term = kwargs.get('max_factors_in_term', self.max_factors_in_term)
        
        print(f'Creating new equation, sparsity value {sparsity}')
        metaparameters = {'terms_number'        : {'optimizable' : False, 'value' : terms_number},
                          'max_factors_in_term' : {'optimizable' : False, 'value' : max_factors_in_term}}
        for idx, variable in enumerate(self.vars_demand_equation):
            metaparameters[('sparsity', variable)] = {'optimizable' : True, 'value' : sparsity[idx]}

        created_solution = SoEq(pool = self.pool, metaparameters = metaparameters)

        try:
            created_solution.set_objective_functions(kwargs['obj_funs'])
        except KeyError:
            created_solution.use_default_multiobjective_function()

        created_solution.create_equations()

        return created_solution
