#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:20:59 2021

@author: mike_ubuntu
"""

import numpy as np
import torch
import time
from typing import Union
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import cm

from epde.structure.main_structures import SoEq, Equation
from epde.operators.template import CompoundOperator
import epde.globals as global_var
# from TEDEouS.solver import point_sort_shift_solver


class L2Fitness(CompoundOperator):
    """
    The operator, which calculates fitness function to the individual (equation) as the L2 norm 
    of the vector of disrepancy between left part of the equation and the right part, evaluated
    on the grid nodes.
    
    Notable attributes:
    -------------------
        
    params : dict
        Inhereted from the ``CompoundOperator`` class. 
        Parameters of the operator; main parameters: 
            
            penalty_coeff - penalty coefficient, to that the fitness function value of equation with no non-zero coefficients, is multiplied;
            
    suboperators : dict
        
        
    Methods:
    -----------
    apply(equation)
        calculate the fitness function of the equation, that will be stored in the equation.fitness_value.    
        
    """
    def apply(self, objective : Equation, arguments : dict):
        """
        Calculate the fitness function values. The result is not returned, but stored in the equation.fitness_value attribute.
        
        Parameters:
        ------------
        equation : Equation object
            the equation object, to that the fitness function is obtained.
            
        Returns:
        ------------
        
        None
        """        
        # print(f'CALCULATING FITNESS FOR {objective.text_form}')
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])
        
        _, target, features = objective.evaluate(normalize = False, return_val = False)
        try:
            discr = (np.dot(features, objective.weights_final[:-1]) + 
                     np.full(target.shape, objective.weights_final[-1]) - target)
            self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)
            discr = np.multiply(discr, self.g_fun_vals)
            rl_error = np.linalg.norm(discr, ord = 2)
        except ValueError:
            raise ValueError('An error in getting weights ')

        if not (self.params['penalty_coeff'] > 0. and self.params['penalty_coeff'] < 1.):
            raise ValueError('Incorrect penalty coefficient set, value shall be in (0, 1).')
            
        fitness_value = rl_error
        if np.sum(objective.weights_final) == 0:
            fitness_value /= self.params['penalty_coeff']
            
        objective.fitness_calculated = True
        objective.fitness_value = fitness_value

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'gene level', 'contains suboperators', 'inplace'}


# class SolverBasedFitness(CompoundOperator):
#     def __init__(self, param_keys : list, model_architecture = None):
#         super().__init__(param_keys)
#         if model_architecture is None:
#             try:
#                 dim = global_var.tensor_cache.get(label = None).ndim
#             except ValueError:
#                 dim = global_var.dimensionality
#             self.model_architecture = torch.nn.Sequential(
#                                                           torch.nn.Linear(dim, 256),
#                                                           torch.nn.Tanh(),
#                                                           torch.nn.Linear(256, 64),
#                                                           torch.nn.Tanh(),       
#                                                           torch.nn.Linear(64, 64),
#                                                           torch.nn.Tanh(),
#                                                           torch.nn.Linear(64, 1024),
#                                                           torch.nn.Tanh(),
#                                                           torch.nn.Linear(1024, 1)
#                                                           )
#         else:
#             self.model_architecture = model_architecture
#         self.training_grid_set = False
        
#     def apply(self, equation, weights_internal = None, weights_final = None):
#         raise NotImplementedError('Solver evaluation approach is a subject to be done.')
#         if not self.training_grid_set:
#             self.set_training_grid()
            
#         if weights_internal is None:
#             self.suboperators['sparsity'].apply(equation)
#         else:
#             equation.weights_internal = weights_internal
#             equation.weights_internal_evald = True
#         if weights_final is None:
#             self.suboperators['coeff_calc'].apply(equation)        
#         else:
#             equation.weights_final = weights_final
#             equation.weights_final_evald = True
        
#         architecture = deepcopy(self.model_architecture)

#         eq_bc = equation.boundary_conditions()
#         eq_form = equation.solver_form()   

#         equation.model = point_sort_shift_solver(self.training_grid, architecture, eq_form, eq_bc,
#                        lambda_bound = self.params['lambda_bound'], verbose = False, 
#                        learning_rate = self.params['learning_rate'], eps = self.params['eps'], 
#                        tmin = self.params['tmin'], tmax = self.params['tmax'], use_cache=True, 
#                        print_every=None,
#                        cache_dir='../cache/')
        
#         main_var_key = ('u', (1.0,))        
#         u_modeled = equation.model(self.training_grid).detach().numpy().reshape(global_var.tensor_cache.get(main_var_key).shape)
#         rl_error = np.linalg.norm(u_modeled - global_var.tensor_cache.get(main_var_key), ord = 2)
        
#         if self.params['verbose']:
#             self.plot_data_vs_solution(global_var.tensor_cache.get(main_var_key), u_modeled)
            
#         if rl_error == 0:
#             fitness_value = np.inf
#             print('infinite fitness!', equation.text_form)
#         else:
#             fitness_value = 1 / (rl_error)

#         equation.fitness_calculated = True
#         equation.fitness_value = fitness_value      
#         equation.clear_after_solver()
        
#     def plot_data_vs_solution(self, data, solution):

#         if self.training_grid.shape[1]==2:
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.plot_trisurf(self.training_grid[:,0].reshape(-1), self.training_grid[:,1].reshape(-1),
#                             solution.reshape(-1), cmap=cm.jet, linewidth=0.2)
#             ax.set_xlabel("x1")
#             ax.set_ylabel("x2")
#             plt.show()
#         if self.training_grid.shape[1]==1:
#             fig = plt.figure()
#             plt.scatter(self.training_grid.reshape(-1), solution.reshape(-1), color = 'r')
#             plt.scatter(self.training_grid.reshape(-1), data.reshape(-1), color = 'k')            
#             plt.show()
