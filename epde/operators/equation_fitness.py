#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:20:59 2021

@author: mike_ubuntu
"""

import numpy as np
import torch
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import cm

from epde.operators.template import Compound_Operator
import epde.globals as global_var
from TEDEouS.solver import point_sort_shift_solver

class L2_fitness(Compound_Operator):
    """
    The operator, which calculates fitness function to the individual (equation) as the L2 norm 
    of the vector of disrepancy between left part of the equation and the right part, evaluated
    on the grid nodes.
    
    Notable attributes:
    -------------------
        
    params : dict
        Inhereted from the ``Compound_Operator`` class. 
        Parameters of the operator; main parameters: 
            
            penalty_coeff - penalty coefficient, to that the fitness function value of equation with no non-zero coefficients, is multiplied;
            
    suboperators : dict
        
        
    Methods:
    -----------
    apply(equation)
        calculate the fitness function of the equation, that will be stored in the equation.fitness_value.    
        
    """
    def apply(self, equation):
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

        self.suboperators['sparsity'].apply(equation)
        self.suboperators['coeff_calc'].apply(equation)
        
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        try:
            rl_error = np.linalg.norm(np.dot(features, equation.weights_final[:-1]) + 
                                  np.full(target.shape, equation.weights_final[-1]) - target, ord = 2)

        except ValueError:
            raise ValueError('An error in getting weights ')
        if rl_error == 0:
            fitness_value = np.inf
            print('infinite fitness!', equation.text_form)
        else:
            fitness_value = 1 / (rl_error)
        if np.sum(equation.weights_final) == 0:
            fitness_value = fitness_value * self.params['penalty_coeff']

        equation.fitness_calculated = True
        equation.fitness_value = fitness_value
        # print(equation.fitness_value)

    @property
    def operator_tags(self):
        return {'fitness evaluation', 'equation level', 'contains suboperators'}  


class Solver_based_fitness(Compound_Operator):
    def __init__(self, param_keys : list, model_architecture = None, dimensionality = 1):
        super().__init__(param_keys)
        if model_architecture is None:
            self.model_architecture = torch.nn.Sequential(
                torch.nn.Linear(dimensionality, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 1)#,
#                torch.nn.Tanh()
                )
        else:
            self.model_architecture = model_architecture
        self.training_grid_set = False
        
    def apply(self, equation):
        if not self.training_grid_set:
            self.set_training_grid()
        self.suboperators['sparsity'].apply(equation)
        self.suboperators['coeff_calc'].apply(equation)        
        
        equation.model = deepcopy(self.model_architecture)

        eq_bc = equation.boundary_conditions()
        eq_form = equation.solver_form()   
        print(equation.text_form)
        equation.model = point_sort_shift_solver(self.training_grid, equation.model, eq_form, eq_bc,
                       lambda_bound = self.params['lambda_bound'], verbose = True, 
                       learning_rate = self.params['learning_rate'], eps = self.params['eps'], 
                       tmin = self.params['tmin'], tmax = self.params['tmax'], use_cache=True, cache_dir='/home/maslyaev/epde/EPDE/solver/cache/')
        
        main_var_key = ('u', (1.0,))        
        u_modeled = equation.model(self.training_grid).detach().numpy().reshape(global_var.tensor_cache.get(main_var_key).shape)
        rl_error = np.linalg.norm(u_modeled - global_var.tensor_cache.get(main_var_key), ord = 2)
        
        if self.params['verbose']:
            self.plot_data_vs_solution(global_var.tensor_cache.get(main_var_key), u_modeled)
            
        if rl_error == 0:
            fitness_value = np.inf
            print('infinite fitness!', equation.text_form)
        else:
            fitness_value = 1 / (rl_error)

        equation.fitness_calculated = True
        equation.fitness_value = fitness_value      
        
    def plot_data_vs_solution(self, data, solution):

        if self.training_grid.shape[1]==2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(self.training_grid[:,0].reshape(-1), self.training_grid[:,1].reshape(-1),
                            solution.reshape(-1), cmap=cm.jet, linewidth=0.2)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            plt.show()
        if self.training_grid.shape[1]==1:
            fig = plt.figure()
            plt.scatter(self.training_grid.reshape(-1), solution.reshape(-1), color = 'r')
            plt.scatter(self.training_grid.reshape(-1), data.reshape(-1), color = 'k')            
            plt.show()        
            
    def set_training_grid(self):
        keys, training_grid = global_var.grid_cache.get_all()
        assert len(keys) == training_grid[0].ndim, 'Mismatching dimensionalities'
        training_grid = np.array(training_grid).reshape((len(training_grid), -1))
        self.training_grid = torch.from_numpy(training_grid).T.type(torch.FloatTensor)
#        print('grid after creation:', self.training_grid[:10], '...', self.training_grid[-10:])
        self.device = torch.device('cpu')
        self.training_grid.to(self.device)     
        self.training_grid_set = True
        # Возможная проблема, когда подаётся тензор со значениями коэфф-тов перед производными
        
        
