#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:34:49 2022

@author: maslyaev
"""


import numpy as np
import sys
import getopt
from collections import OrderedDict

global opt, args
opt, args = getopt.getopt(sys.argv[2:], '', ['path='])

sys.path.append(opt[0][1]) # $1 "--path=" pwd  + '/epde'

from epde.interface.equation_translator import translate_equation
import epde.globals as global_var

from epde.interface.prepared_tokens import TrigonometricTokens
from epde.evaluators import simple_function_evaluator, trigonometric_evaluator
from epde.interface.token_family import TokenFamily, TF_Pool
from epde.cache.cache import prepare_var_tensor, upload_simple_tokens, upload_grids
from epde.supplementary import Define_Derivatives
from epde.preprocessing.derivatives import Preprocess_derivatives

# from epde.interface.equation_translator import translate_equation

def get_basic_var_family(var_name, deriv_names, deriv_orders):
    entry_token_family = TokenFamily(var_name, family_of_derivs = True)
    entry_token_family.set_status(demands_equation=True, unique_specific_token=False, 
                                  unique_token_type=False, s_and_d_merged = False, 
                                  meaningful = True)     
    entry_token_family.set_params(deriv_names, OrderedDict([('power', (1, 1))]),
                                  {'power' : 0}, deriv_orders)
    entry_token_family.set_evaluator(simple_function_evaluator, [])    
    return entry_token_family

def prepare_basic_inputs():
    grids = [np.linspace(0, 4*np.pi, 1000),]
    var_name = 'u'
    u = np.sin(x) + 1.3 * np.cos(x)#np.load('/home/maslyaev/epde/EPDE_main/tests/system/Test_data/fill366.npy')

    global_var.init_caches(set_grids = True)
    global_var.set_time_axis(0)
    global_var.grid_cache.memory_usage_properties(u, 3, None)
    global_var.tensor_cache.memory_usage_properties(u, 3, None)

    deriv_names, deriv_orders = Define_Derivatives(var_name, dimensionality=u.ndim, max_order = 1)

    method = 'poly'; method_kwargs = {'grid' : grids, 'smooth' : False}
    data_tensor, derivatives = Preprocess_derivatives(u, method=method, method_kwargs=method_kwargs)
    derivs_stacked = prepare_var_tensor(u, derivatives, time_axis = global_var.time_axis)

    upload_grids(grids, global_var.grid_cache)
    upload_simple_tokens(deriv_names, global_var.tensor_cache, derivs_stacked)
    global_var.tensor_cache.use_structural()

    var_family = get_basic_var_family(var_name, deriv_names, deriv_orders)
    trig_tokens = TrigonometricTokens(dimensionality = 0, freq = (0.95, 1.05))
    # trig_tokens.token_family
    pool = TF_Pool([var_family, trig_tokens.token_family])

    return grids, pool

def mock_equation():
    grids, mock_pool = prepare_basic_inputs()
    print('Mock pool families:', [family.tokens for family in mock_pool.families])
    text_form = ('1.0 * u{power: 1} * sin{freq: 1, power: 1, dim: 0} + 1. = '
                 'du/dx1{power: 1} * cos{freq: 1, power: 1, dim: 0}')
    return grids, translate_equation(text_form, mock_pool)

def test_adapter_form_only():
    grids, equation = mock_equation()
    solver_form_adapter = SolverFormAdapter()
    equation_form_no_grids = solver_form_adapter.form()
    equation_form_base_grids = solver_form_adapter.form(grids)
    
    reference_basic_solver_form = []    
    equation_form_no_grids == equation_form_base_grids == reference_solver_form
    
    
    
def test_adapter_full_solution():
        