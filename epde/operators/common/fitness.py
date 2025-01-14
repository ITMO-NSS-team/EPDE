#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:20:59 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
import torch

import matplotlib.pyplot as plt
from matplotlib import cm

from epde.integrate import SolverAdapter
from epde.structure.main_structures import SoEq, Equation
from epde.operators.utils.template import CompoundOperator
import epde.globals as global_var
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

LOSS_NAN_VAL = 1e7

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
            else:
                discr_feats = np.dot(features, objective.weights_final[:-1][objective.weights_internal != 0])

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
    # To be modified to include physics-informed information criterion (PIC)

    key = 'SolverBasedFitness'
    
    def __init__(self, param_keys: list):
        super().__init__(param_keys)
        self.adapter = None

    def set_adapter(self, net = None):

        if self.adapter is None or net is not None:
            compiling_params = {'mode': 'autograd', 'tol':0.01, 'lambda_bound': 100} #  'h': 1e-1
            optimizer_params = {}
            training_params = {'epochs': 4e3, 'info_string_every' : 1e3}
            early_stopping_params = {'patience': 4, 'no_improvement_patience' : 250}

            explicit_cpu = True
            device = 'cuda' if (torch.cuda.is_available and not explicit_cpu) else 'cpu'

            self.adapter = SolverAdapter(net = net, use_cache = False, device=device)

            self.adapter.set_compiling_params(**compiling_params)            
            self.adapter.set_optimizer_params(**optimizer_params)
            self.adapter.set_early_stopping_params(**early_stopping_params)
            self.adapter.set_training_params(**training_params)

    def apply(self, objective : SoEq, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        try:
            net = deepcopy(global_var.solution_guess_nn)
        except NameError:
            net = None

        self.set_adapter(net=net)

        self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])

        print('solving equation:')
        print(objective.text_form)

        loss_add, solution_nn = self.adapter.solve_epde_system(system = objective, grids = None, 
                                                               boundary_conditions = None, use_fourier=True)
        _, grids = global_var.grid_cache.get_all(mode = 'torch')
        
        grids = torch.stack([grid.reshape(-1) for grid in grids], dim = 1).float()
        solution = solution_nn(grids).detach().cpu().numpy()
        self.g_fun_vals = global_var.grid_cache.g_func
        
        for eq_idx, eq in enumerate(objective.vals):
            if torch.isnan(loss_add):
                fitness_value = 2*LOSS_NAN_VAL
            else:
                referential_data = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))

                print(f'solution shape {solution.shape}')
                print(f'solution[..., eq_idx] {solution[..., eq_idx].shape}, eq_idx {eq_idx}')
                discr = (solution[..., eq_idx] - referential_data.reshape(solution[..., eq_idx].shape))
                discr = np.multiply(discr, self.g_fun_vals.reshape(discr.shape))
                rl_error = np.linalg.norm(discr, ord = 2) 
                
                print(f'fitness error is {rl_error}, while loss addition is {float(loss_add)}')            
                fitness_value = rl_error + self.params['pinn_loss_mult'] * float(loss_add) # TODO: make pinn_loss_mult case dependent
                if np.sum(eq.weights_final) == 0: 
                    fitness_value /= self.params['penalty_coeff']

            eq.fitness_calculated = True
            eq.fitness_value = fitness_value

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'chromosome level', 'contains suboperators', 'inplace'}


class PIC(CompoundOperator):

    key = 'PIC'

    def __init__(self, param_keys: list):
        super().__init__(param_keys)
        self.adapter = None
        self.window_size = 10   # Idk which one we need

    def set_adapter(self, net=None):

        if self.adapter is None or net is not None:
            compiling_params = {'mode': 'autograd', 'tol': 0.01, 'lambda_bound': 100}  # 'h': 1e-1
            optimizer_params = {}
            training_params = {'epochs': 4e3, 'info_string_every': 1e3}
            early_stopping_params = {'patience': 4, 'no_improvement_patience': 250}

            explicit_cpu = True
            device = 'cuda' if (torch.cuda.is_available and not explicit_cpu) else 'cpu'

            self.adapter = SolverAdapter(net=net, use_cache=False, device=device)

            self.adapter.set_compiling_params(**compiling_params)
            self.adapter.set_optimizer_params(**optimizer_params)
            self.adapter.set_early_stopping_params(**early_stopping_params)
            self.adapter.set_training_params(**training_params)

    def apply(self, objective: SoEq, arguments: dict):
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)

        try:
            net = deepcopy(global_var.solution_guess_nn)
        except NameError:
            net = None

        self.set_adapter(net=net)

        self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])

        print('solving equation:')
        print(objective.text_form)

        loss_add, solution_nn = self.adapter.solve_epde_system(system=objective, grids=None,
                                                               boundary_conditions=None, use_fourier=True)

        _, grids = global_var.grid_cache.get_all(mode='torch')

        grids = torch.stack([grid.reshape(-1) for grid in grids], dim=1).float()
        solution = solution_nn(grids).detach().cpu().numpy()
        self.g_fun_vals = global_var.grid_cache.g_func

        for eq_idx, eq in enumerate(objective.vals):
            # Calculate r-loss
            target = eq.structure[eq.target_idx]
            target_vals = target.evaluate(False)
            features_vals = []
            nonzero_features_indexes = []

            # Collect non-zero feature values and their indices
            for i in range(len(eq.structure)):
                if i == eq.target_idx:
                    continue
                idx = i if i < eq.target_idx else i - 1
                if eq.weights_internal[idx] != 0:
                    features_vals.append(eq.structure[i].evaluate(False))
                    nonzero_features_indexes.append(idx)

            # Stack features and add constant term for bias
            features = features_vals[0]
            if len(features_vals) > 1:
                for i in range(1, len(features_vals)):
                    features = np.vstack([features, features_vals[i]])
            features = np.vstack([features, np.ones(features_vals[0].shape)])  # Add constant feature
            features = np.transpose(features)

            self.window_size = len(target_vals) // 2
            num_horizons = len(target_vals) - self.window_size + 1
            eq_window_weights = []

            # Perform optimization over sliding windows
            for start_idx in range(num_horizons):
                end_idx = start_idx + self.window_size

                target_window = target_vals[start_idx:end_idx]
                feature_window = features[start_idx:end_idx, :]

                # Linear Regression
                estimator = LinearRegression(fit_intercept=False)
                if feature_window.ndim == 1:
                    feature_window = feature_window.reshape(-1, 1)
                try:
                    self.g_fun_vals_window = self.g_fun_vals.reshape(-1)[start_idx:end_idx]
                except AttributeError:
                    self.g_fun_vals_window = None
                estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals_window)
                valuable_weights = estimator.coef_

                window_weights = np.zeros(len(eq.structure))
                # for weight_idx in range(len(window_weights) - 1):
                for weight_idx in range(len(window_weights)):
                    if weight_idx in nonzero_features_indexes:
                        window_weights[weight_idx] = valuable_weights[nonzero_features_indexes.index(weight_idx)]
                # window_weights[-1] = valuable_weights[-1]
                eq_window_weights.append(window_weights)

            # eq_cv = [np.std(_) / np.mean(_) for _ in zip(*eq_window_weights)]  # Default std
            eq_cv = [np.abs(np.std(_) / (np.mean(_))) for _ in zip(*eq_window_weights)]  # As in papers' repo
            # eq_cv = [np.mean((_ - np.mean(_)) / np.mean(_)) for _ in zip(*eq_window_weights)]  # As in paper formula (BUG)
            eq_cv_valuable = [x for x in eq_cv if not np.isnan(x)]
            print('eq_cv: ', eq_cv)
            print('eq_cv_valuable: ', eq_cv_valuable)
            lr = np.mean(eq_cv_valuable)

            # Calculate p-loss
            if torch.isnan(loss_add):
                lp = 2 * LOSS_NAN_VAL
            else:
                referential_data = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))

                print(f'solution shape {solution.shape}')
                print(f'solution[..., eq_idx] {solution[..., eq_idx].shape}, eq_idx {eq_idx}')
                sol_pinn = solution[..., eq_idx]
                sol_ann = referential_data.reshape(solution[..., eq_idx].shape)
                sol_pinn_normalized = (sol_pinn - min(sol_pinn)) / (max(sol_pinn) - min(sol_pinn))
                sol_ann_normalized = (sol_ann - min(sol_ann)) / (max(sol_ann) - min(sol_ann))
                discr = sol_pinn_normalized - sol_ann_normalized

                # discr = (solution[..., eq_idx] - referential_data.reshape(solution[..., eq_idx].shape))  # Default
                discr = np.multiply(discr, self.g_fun_vals.reshape(discr.shape))
                rl_error = np.linalg.norm(discr, ord=2)

                print(f'fitness error is {rl_error}, while loss addition is {float(loss_add)}')
                lp = rl_error + self.params['pinn_loss_mult'] * float(
                    loss_add)  # TODO: make pinn_loss_mult case dependent
                if np.sum(eq.weights_final) == 0:
                    lp /= self.params['penalty_coeff']

                # lp = np.sqrt((discr ** 2).mean())

            # Fit
            eq.fitness_calculated = True
            print('lr: ', lr, '\t lp: ', lp, '\t PIC: ', lr * lp)
            eq.fitness_value = lr * lp

    
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
