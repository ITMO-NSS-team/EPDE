#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:21:57 2021

@author: maslyaev
"""

import numpy as np
from collections import OrderedDict
import os
import sys
import getopt

global opt, args
#opt, args = getopt.getopt(sys.argv[2:], '', ['path='])

#sys.path.append(opt[0][1])

import epde.globals as global_var

from epde.moeadd.moeadd import *
from epde.moeadd.moeadd_supplementary import *

from epde.prep.DomainPruning import Domain_Pruner

import epde.operators.sys_search_operators as operators
#from epde.src.evo_optimizer import Operator_director
from epde.evaluators import simple_function_evaluator, trigonometric_evaluator, inverse_function_evaluator
from epde.supplementary import Define_Derivatives
from epde.cache.cache import upload_simple_tokens, upload_grids, prepare_var_tensor, np_ndarray_section
from epde.prep.derivatives import Preprocess_derivatives
from epde.interface.token_family import TF_Pool, TokenFamily

from epde.eq_search_strategy import Strategy_director, Strategy_director_solver
from epde.operators.ea_stop_criteria import Iteration_limit

if __name__ == "__main__":
    x = np.linspace(0.5, 5, 10)
    file = np.loadtxt('/home/maslyaev/epde/EPDE/tests/cylinder/data/Data_32_points_.dat', 
                      delimiter=' ', usecols=range(33))
    t_max = 1000
    t = file[:t_max, 0]
    T_vals = file[:t_max, 1:11] 
    grids = np.meshgrid(t, x, indexing = 'ij')
    
    # ff_filename = '/media/mike_ubuntu/DATA/EPDE_publication/tests/cylinder/data/grids_saved.npy'
    # output_file_name = '/media/mike_ubuntu/DATA/EPDE_publication/tests/cylinder/data/derivs.npy'

    max_order = 3

    poly_kwargs = {'grid' : grids, 'smooth' : False, 'max_order' : max_order, 
                   'polynomial_window' : 7, 'poly_order' : 6}
    pruner = Domain_Pruner(domain_selector_kwargs={'threshold' : 1e-5})
    
    derivs = Preprocess_derivatives(T_vals, method = 'poly', method_kwargs = poly_kwargs)
    derivs_new = Preprocess_derivatives(T_vals, method = 'ANN', method_kwargs = {'grid' : grids, 'max_order' : max_order})
    # derivatives_new = derivs_new[1].reshape((T_vals.shape[0], T_vals.shape[1], -1))
    # derivatives = derivs[1].reshape((T_vals.shape[0], T_vals.shape[1], -1))