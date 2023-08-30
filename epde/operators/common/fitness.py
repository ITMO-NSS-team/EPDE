#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:20:59 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import cm

from epde.interface.solver_integration import SolverAdapter
from epde.structure.main_structures import SoEq, Equation
from epde.operators.utils.template import CompoundOperator
import epde.globals as global_var

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
    
    key = 'DiscrepancyBasedFitness'

    def apply(self, objective: Equation, arguments: dict):
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
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])
        
        _, target, features = objective.evaluate(normalize = False, return_val = False)
        try:
            if features is None:
                discr_feats = 0
                # print('having equation with no features')
            else:
                discr_feats = np.dot(features, objective.weights_final[:-1][objective.weights_final[:-1] != 0])
                # print(features.shape, objective.weights_final[:-1][objective.weights_final[:-1] != 0].shape)

            discr = (discr_feats + np.full(target.shape, objective.weights_final[-1]) - target)
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


class SolverBasedFitness(CompoundOperator):
    key = 'SolverBasedFitness'
    
    def __init__(self, param_keys: list, solver_kwargs: dict = {'model' : None, 'use_cache' : True}):
        super().__init__(param_keys)
        solver_kwargs['dim'] = len(global_var.grid_cache.get_all()[1])
        
        self.adapter = None # SolverAdapter(var_number = len(system.vars_to_describe))

    def set_adapter(self, var_number):
        if self.adapter is not None:
            self.adapter = SolverAdapter(var_number = var_number)

    def apply(self, objective : SoEq, arguments : dict):
        self.set_adapter(len(objective.vars_to_describe))
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])

        # _, target, features = objective.evaluate(normalize = False, return_val = False)

        grid = global_var.grid_cache.get_all()[1]
        solution_model = self.adapter.solve_epde_system(system = objective, grids = grid, 
                                                        boundary_conditions = None)

        self.g_fun_vals = global_var.grid_cache.g_func #.reshape(-1)
        
        solution = solution_model(self.adapter.convert_grid(grid)).detach().numpy()
        for eq_idx, eq in enumerate(objective.structure):
            referential_data = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))

            discr = (solution[eq_idx, ...] - referential_data.reshape(solution[eq_idx, ...].shape))

            discr = np.multiply(discr, self.g_fun_vals.reshape(discr.shape))
            rl_error = np.linalg.norm(discr, ord = 2)
            
            fitness_value = rl_error
            if np.sum(eq.weights_final) == 0: 
                fitness_value /= self.params['penalty_coeff']

            eq.fitness_calculated = True
            eq.fitness_value = fitness_value

            if global_var.verbose.plot_DE_solutions:
                plot_data_vs_solution(self.adapter.convert_grid(grid),
                                      data = referential_data.reshape(solution[eq_idx, ...].shape), 
                                      solution = solution[eq_idx, ...])

    
def plot_data_vs_solution(grid, data, solution):
    if grid.shape[1]==2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1),
                        solution.reshape(-1), cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()
    if grid.shape[1]==1:
        fig = plt.figure()
        plt.scatter(grid.reshape(-1), solution.reshape(-1), color = 'r')
        plt.scatter(grid.reshape(-1), data.reshape(-1), color = 'k')            
        plt.show()
    else:
        raise Exception('Infeasible dimensionality of the input dataset.')
