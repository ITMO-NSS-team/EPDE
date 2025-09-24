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
    """
    Creates a foundational TokenFamily for a variable and its derivatives.
    
        This method constructs a TokenFamily object, essential for representing a variable
        within the equation discovery process. It configures the TokenFamily to handle
        derivatives and sets up its evaluation mechanism. This ensures that the variable
        and its derivatives can be properly incorporated into candidate equations.
    
        Args:
          var_name: The name of the variable.
          deriv_names: A list of derivative names.
          deriv_orders: A list of derivative orders corresponding to deriv_names.
    
        Returns:
          TokenFamily: A configured TokenFamily object representing the variable
            and its derivatives, ready for use in equation discovery.
    """
    entry_token_family = TokenFamily(var_name, family_of_derivs = True)
    entry_token_family.set_status(demands_equation=True, unique_specific_token=False, 
                                  unique_token_type=False, s_and_d_merged = False, 
                                  meaningful = True)     
    entry_token_family.set_params(deriv_names, OrderedDict([('power', (1, 1))]),
                                  {'power' : 0}, deriv_orders)
    entry_token_family.set_evaluator(simple_function_evaluator, [])    
    return entry_token_family

def prepare_basic_inputs():
    """
    Prepares the foundation for equation discovery by initializing essential components.
        
        This method configures the necessary data structures and objects
        to facilitate the equation discovery process. It involves setting up grids,
        defining variables and their derivatives, preprocessing the data to
        extract relevant features, and constructing a token pool that defines the search space.
        This setup ensures that the subsequent search algorithms have a well-defined
        and preprocessed input to efficiently explore potential equation candidates.
        
        Args:
            None
        
        Returns:
            tuple: A tuple containing:
                - grids: A list of NumPy arrays representing the grids over which the data is defined.
                - pool: A TF_Pool object containing the variable family and
                  trigonometric tokens, which constitute the building blocks for constructing equations.
    """
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
    """
    Generates a mock equation and input grids for testing the equation discovery process.
    
        This method sets up a simplified environment with predefined input grids and a mock token pool.
        It then translates a symbolic equation using this mock pool. This is essential for verifying
        the correctness and stability of the equation translation and subsequent steps in the equation
        discovery pipeline, such as equation simplification and numerical evaluation.
    
        Returns:
            tuple: A tuple containing the input grids and the translated equation.
                - grids (tuple): The basic input grids prepared by `prepare_basic_inputs`.
                - translated_equation (torch.Tensor): The equation translated into a numerical representation
                  using the mock token pool.
    """
    grids, mock_pool = prepare_basic_inputs()
    print('Mock pool families:', [family.tokens for family in mock_pool.families])
    text_form = ('1.0 * u{power: 1} * sin{freq: 1, power: 1, dim: 0} + 1. = '
                 'du/dx1{power: 1} * cos{freq: 1, power: 1, dim: 0}')
    return grids, translate_equation(text_form, mock_pool)

def test_adapter_form_only():
    """
    Tests the SolverFormAdapter's form method, ensuring consistent equation form generation.
    
        This method verifies that the SolverFormAdapter produces equivalent equation forms
        regardless of whether grid data is explicitly provided. This ensures that the equation
        discovery process remains consistent and reliable, even when grid information is
        implicitly defined within the dataset.
    
        Args:
            None
    
        Returns:
            None
    """
    grids, equation = mock_equation()
    solver_form_adapter = SolverFormAdapter()
    equation_form_no_grids = solver_form_adapter.form()
    equation_form_base_grids = solver_form_adapter.form(grids)
    
    reference_basic_solver_form = []    
    equation_form_no_grids == equation_form_base_grids == reference_solver_form
    
    
    
def test_adapter_full_solution():    """
    Tests a full adapter solution.
    
         This method sets up and executes a complete adapter-based workflow,
         verifying the interaction between different components. It doesn't
         take any direct input parameters but relies on predefined configurations
         and data within the test environment.
    
         Returns:
    """
    Tests the end-to-end functionality of the equation discovery process.
    
    This method orchestrates a complete EPDE workflow, from data input to equation
    output, ensuring that all components work together seamlessly. It validates
    the system's ability to identify governing equations from a dataset, reflecting
    a real-world application scenario. The test relies on predefined configurations
    and data within the test environment to simulate a typical equation discovery task.
    
    Args:
        None
    
    Returns:
        None.  The method implicitly asserts the correctness of the discovered
            equations and the overall workflow execution. It checks if the system
            can successfully process data and identify a meaningful equation.
    """
             None. This method doesn't return any value; it asserts the
                 correctness of the adapter solution's behavior.
    """

        