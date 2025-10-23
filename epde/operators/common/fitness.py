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

    def apply(self, objective: Equation, arguments: dict, force_out_of_place: bool = False):
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

        if force_out_of_place:
            self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])
        
        _, target, features = objective.evaluate(normalize = False, return_val = False)
        if features is None:
            discr_feats = 0
        else:
            discr_feats = np.dot(features, objective.weights_final[:-1][objective.weights_internal != 0])

        discr = (discr_feats + np.full(target.shape, objective.weights_final[-1]) - target)
        self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)
        discr = np.multiply(discr, self.g_fun_vals)
        rl_error = np.linalg.norm(discr, ord = 2)

        if not (self.params['penalty_coeff'] > 0. and self.params['penalty_coeff'] < 1.):
            raise ValueError('Incorrect penalty coefficient set, value shall be in (0, 1).')
            
        fitness_value = rl_error
        if np.sum(objective.weights_final) == 0:
            fitness_value /= self.params['penalty_coeff']
        
        if force_out_of_place:
            return fitness_value
        else:
            objective.fitness_calculated = True
            objective.fitness_value = fitness_value

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'gene level', 'contains suboperators', 'inplace'}


class L2LRFitness(CompoundOperator):
    key = 'DiscrepancyBasedFitnessWithCV'

    def apply(self, objective: Equation, arguments: dict, force_out_of_place: bool = False):
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
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)
        if force_out_of_place:
            self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])

        _, target, features = objective.evaluate(normalize=False, return_val=False)

        self.get_g_fun_vals()
        data_shape = global_var.grid_cache.g_func.shape

        if features is None:
            discr = target - objective.weights_final[-1]
        else:
            discr_feats = np.dot(features, objective.weights_final[:-1][objective.weights_internal != 0])
            discr_feats = discr_feats + objective.weights_final[-1]
            discr = discr_feats - target

        discr = np.multiply(discr, self.g_fun_vals) / np.std(target)
        rl_error = np.sqrt(np.mean(discr ** 2))

        if not (self.params['penalty_coeff'] > 0. and self.params['penalty_coeff'] < 1.):
            raise ValueError('Incorrect penalty coefficient set, value shall be in (0, 1).')

        fitness_value = rl_error

        if force_out_of_place:
            return fitness_value

        objective.aic = None
        objective.aic_calculated = True

        # assert objective.simplified, 'Trying to evaluate not simplified equation.'

        # Calculate r-loss
        target_vals = target.reshape(*data_shape)
        features_vals = []

        if target_vals.ndim == 1:
            horizons_default = 30
            window_size = len(target_vals) // 2
            num_horizons = len(target_vals) - window_size + 1
            if num_horizons < horizons_default:
                step_size = 1
            else:
                step_size = num_horizons // horizons_default
            eq_window_weights = []
            # Compute coefficients and collect statistics over horizons
            if features is None:
                for start_idx in range(0, num_horizons, step_size):
                    end_idx = start_idx + window_size
                    target_window = target_vals[start_idx:end_idx]
                    if np.isclose(np.mean(target_window), 0):
                        window_stability = np.abs(np.std(target_window))
                    else:
                        window_stability = np.abs(np.std(target_window) / np.mean(target_window))
                    eq_window_weights.append(window_stability)
                lr = np.mean(eq_window_weights)
            else:
                for start_idx in range(0, num_horizons, step_size):
                    end_idx = start_idx + window_size
                    target_window = target_vals[start_idx:end_idx]
                    feature_window = features.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1, features.shape[-1])
                    estimator = LinearRegression(fit_intercept=True)
                    estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals[start_idx:end_idx])
                    valuable_weights = estimator.coef_
                    eq_window_weights.append(valuable_weights)
                eq_cv = np.array([
                    np.abs(np.std(_)) if np.isclose(np.mean(_), 0)
                    # else np.abs(np.std(_) / np.sqrt(np.mean(np.pow(_, 2))))
                    else np.abs(np.std(_) / np.mean(_))
                    for _ in zip(*eq_window_weights)
                ])
                lr = eq_cv.sum()

        elif target_vals.ndim == 2:
            lr = 0
            for dim in range(target_vals.ndim):
                horizons_default = 30
                eq_window_weights = []
                window_size = target_vals.shape[dim] // 2
                num_horizons = target_vals.shape[dim] - window_size + 1
                if num_horizons < horizons_default:
                    step_size = 1
                else:
                    step_size = num_horizons // horizons_default
                # Compute coefficients and collect statistics over horizons
                if features is None:
                    for start_idx in range(0, num_horizons, step_size):
                        end_idx = start_idx + window_size
                        if dim == 0:
                            target_window = target_vals[start_idx:end_idx, :].reshape(-1)
                        else:
                            target_window = target_vals[:, start_idx:end_idx].reshape(-1)
                        if np.isclose(np.mean(target_window), 0):
                            window_stability = np.abs(np.std(target_window))
                        else:
                            window_stability = np.abs(np.std(target_window) / np.mean(target_window))
                        eq_window_weights.append(window_stability)
                    lr += np.mean(eq_window_weights)
                else:
                    for start_idx in range(0, num_horizons, step_size):
                        end_idx = start_idx + window_size
                        estimator = LinearRegression(fit_intercept=True)
                        if dim == 0:
                            target_window = target_vals[start_idx:end_idx, :].reshape(-1)
                            feature_window = features.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1, features.shape[-1])
                            estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1))
                        else:
                            target_window = target_vals[:, start_idx:end_idx].reshape(-1)
                            feature_window = features.reshape(*data_shape, -1)[:, start_idx:end_idx].reshape(-1, features.shape[-1])
                            estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, start_idx:end_idx].reshape(-1))
                        valuable_weights = estimator.coef_
                        eq_window_weights.append(valuable_weights)
                    eq_cv = np.array([
                        np.abs(np.std(_)) if np.isclose(np.mean(_), 0)
                        else np.abs(np.std(_) / np.mean(_))
                        for _ in zip(*eq_window_weights)
                    ])
                    lr += eq_cv.sum()

        elif target_vals.ndim == 3:
            lr = 0
            for dim in range(target_vals.ndim):
                eq_window_weights = []
                window_size = target_vals.shape[dim] // 2
                num_horizons = target_vals.shape[dim] - window_size + 1
                # Compute coefficients and collect statistics over horizons
                if features is None:
                    for start_idx in range(num_horizons):
                        end_idx = start_idx + window_size
                        if dim == 0:
                            target_window = target_vals[start_idx:end_idx, :, :].reshape(-1)
                        elif dim == 1:
                            target_window = target_vals[:, start_idx:end_idx, :].reshape(-1)
                        else:
                            target_window = target_vals[:, :, start_idx:end_idx].reshape(-1)
                        if np.isclose(np.mean(target_window), 0):
                            window_stability = np.abs(np.std(target_window))
                        else:
                            window_stability = np.abs(np.std(target_window) / np.mean(target_window))
                        eq_window_weights.append(window_stability)
                    lr += np.mean(eq_window_weights)
                else:
                    for start_idx in range(num_horizons):
                        end_idx = start_idx + window_size
                        estimator = LinearRegression(fit_intercept=True)
                        if dim == 0:
                            target_window = target_vals[start_idx:end_idx, :, :].reshape(-1)
                            feature_window = features.reshape(*data_shape, -1)[start_idx:end_idx, :, :].reshape(-1, features.shape[-1])
                            estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[start_idx:end_idx, :, :].reshape(-1))
                        elif dim == 1:
                            target_window = target_vals[:, start_idx:end_idx, :].reshape(-1)
                            feature_window = features.reshape(*data_shape, -1)[:, start_idx:end_idx, :].reshape(-1, features.shape[-1])
                            estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, start_idx:end_idx, :].reshape(-1))
                        elif dim == 2:
                            target_window = target_vals[:, :, start_idx:end_idx].reshape(-1)
                            feature_window = features.reshape(*data_shape, -1)[:, :, start_idx:end_idx].reshape(-1, features.shape[-1])
                            estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, :, start_idx:end_idx].reshape(-1))
                        valuable_weights = estimator.coef_
                        eq_window_weights.append(valuable_weights)
                    eq_cv = np.array([
                        np.abs(np.std(_)) if np.isclose(np.mean(_), 0)
                        else np.abs(np.std(_) / np.mean(_))
                        for _ in zip(*eq_window_weights)
                    ])
                    lr += eq_cv.sum()

        objective.fitness_calculated = True
        objective.fitness_value = fitness_value
        objective.stability_calculated = True
        objective.coefficients_stability = lr

    def get_g_fun_vals(self):
        try:
            self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)
        except AttributeError:
            self.g_fun_vals = None

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

            explicit_cpu = False
            device = 'cuda' if (torch.cuda.is_available and not explicit_cpu) else 'cpu'

            self.adapter = SolverAdapter(net = net, use_cache = False, device=device)

            self.adapter.set_compiling_params(**compiling_params)            
            self.adapter.set_optimizer_params(**optimizer_params)
            self.adapter.set_early_stopping_params(**early_stopping_params)
            self.adapter.set_training_params(**training_params)

    def apply(self, objective : SoEq, arguments : dict, force_out_of_place: bool = False):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        try:
            net = deepcopy(global_var.solution_guess_nn)
        except NameError:
            net = None

        self.set_adapter(net=net)
        if force_out_of_place:
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
        
        if force_out_of_place:
            sum_err = 0

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

                if force_out_of_place:
                    sum_err += fitness_value
                else:
                    eq.fitness_calculated = True
                    eq.fitness_value = fitness_value

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'chromosome level', 'contains suboperators', 'inplace'}


class PIC(CompoundOperator):

    key = 'PIC'

    def __init__(self, param_keys: list):
        super().__init__(param_keys)
        self.adapter = None

    def set_adapter(self, net=None):

        if self.adapter is None or net is not None:
            compiling_params = {'mode': 'autograd', 'tol': 0.01, 'lambda_bound': 100}  # 'h': 1e-1
            optimizer_params = {}
            training_params = {'epochs': 4e3, 'info_string_every': 1e3}
            early_stopping_params = {'patience': 4, 'no_improvement_patience': 250}

            explicit_cpu = False
            device = 'cuda' if (torch.cuda.is_available and not explicit_cpu) else 'cpu'

            self.adapter = SolverAdapter(net=net, use_cache=False, device=device)

            self.adapter.set_compiling_params(**compiling_params)
            self.adapter.set_optimizer_params(**optimizer_params)
            self.adapter.set_early_stopping_params(**early_stopping_params)
            self.adapter.set_training_params(**training_params)

    def apply(self, objective: SoEq, arguments: dict, force_out_of_place: bool = False):
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)

        try:
            net = deepcopy(global_var.solution_guess_nn)
        except NameError:
            net = None

        self.set_adapter(net=net)

        if force_out_of_place:
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

        if force_out_of_place:
            sum_err = 0

        for eq_idx, eq in enumerate(objective.vals):
            # Calculate p-loss
            if torch.isnan(loss_add):
                lp = 2 * LOSS_NAN_VAL
            else:
                print(f'solution shape {solution.shape}')
                print(f'solution[..., eq_idx] {solution[..., eq_idx].shape}, eq_idx {eq_idx}')
                referential_data = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))
                discr = solution[..., eq_idx] - referential_data.reshape(solution[..., eq_idx].shape)
                discr = np.multiply(discr, self.g_fun_vals.reshape(discr.shape)) / np.std(discr)
                rl_error = np.sqrt(np.mean(discr ** 2))

                print(f'fitness error is {rl_error}, while loss addition is {float(loss_add)}')
                lp = rl_error + self.params['pinn_loss_mult'] * float(
                    loss_add)  # TODO: make pinn_loss_mult case dependent

            if force_out_of_place:
                sum_err += lp
                continue

            eq.aic_calculated = True

            # Calculate r-loss
            _, target, features = eq.evaluate(normalize=False, return_val=False)
            data_shape = global_var.grid_cache.g_func.shape
            target_vals = target.reshape(*data_shape)

            if target_vals.ndim == 1:
                window_size = len(target_vals) // 2
                num_horizons = len(target_vals) - window_size + 1
                if num_horizons < 30:
                    step_size = 1
                else:
                    step_size = num_horizons // 30
                eq_window_weights = []
                # Compute coefficients and collect statistics over horizons
                if features is None:
                    for start_idx in range(0, num_horizons, step_size):
                        end_idx = start_idx + window_size
                        target_window = target_vals[start_idx:end_idx]
                        if np.isclose(np.mean(target_window), 0):
                            window_stability = np.abs(np.std(target_window))
                        else:
                            window_stability = np.abs(np.std(target_window) / np.mean(target_window))
                        eq_window_weights.append(window_stability)
                    lr = np.mean(eq_window_weights)
                else:
                    for start_idx in range(0, num_horizons, step_size):
                        end_idx = start_idx + window_size
                        target_window = target_vals[start_idx:end_idx]
                        feature_window = features.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1, features.shape[-1])
                        estimator = LinearRegression(fit_intercept=True)
                        estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals[start_idx:end_idx])
                        valuable_weights = estimator.coef_
                        eq_window_weights.append(valuable_weights)
                    eq_cv = np.array([
                        np.abs(np.std(_)) if np.isclose(np.mean(_), 0)
                        else np.abs(np.std(_) / np.mean(_))
                        for _ in zip(*eq_window_weights)
                    ])
                    lr = eq_cv.sum()

            elif target_vals.ndim == 2:
                lr = 0
                for dim in range(target_vals.ndim):
                    eq_window_weights = []
                    window_size = target_vals.shape[dim] // 2
                    num_horizons = target_vals.shape[dim] - window_size + 1
                    if num_horizons < 30:
                        step_size = 1
                    else:
                        step_size = num_horizons // 30
                    # Compute coefficients and collect statistics over horizons
                    if features is None:
                        for start_idx in range(0, num_horizons, step_size):
                            end_idx = start_idx + window_size
                            if dim == 0:
                                target_window = target_vals[start_idx:end_idx, :].reshape(-1)
                            else:
                                target_window = target_vals[:, start_idx:end_idx].reshape(-1)
                            if np.isclose(np.mean(target_window), 0):
                                window_stability = np.abs(np.std(target_window))
                            else:
                                window_stability = np.abs(np.std(target_window) / np.mean(target_window))
                            eq_window_weights.append(window_stability)
                        lr += np.mean(eq_window_weights)
                    else:
                        for start_idx in range(0, num_horizons, step_size):
                            end_idx = start_idx + window_size
                            estimator = LinearRegression(fit_intercept=True)
                            if dim == 0:
                                target_window = target_vals[start_idx:end_idx, :].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1, features.shape[-1])
                                estimator.fit(feature_window, target_window,
                                              sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1))
                            else:
                                target_window = target_vals[:, start_idx:end_idx].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[:, start_idx:end_idx].reshape(-1, features.shape[-1])
                                estimator.fit(feature_window, target_window,
                                              sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, start_idx:end_idx].reshape(-1))
                            valuable_weights = estimator.coef_
                            eq_window_weights.append(valuable_weights)
                        eq_cv = np.array([
                            np.abs(np.std(_)) if np.isclose(np.mean(_), 0)
                            else np.abs(np.std(_) / np.mean(_))
                            for _ in zip(*eq_window_weights)
                        ])
                        lr += eq_cv.sum()

            elif target_vals.ndim == 3:
                lr = 0
                for dim in range(target_vals.ndim):
                    eq_window_weights = []
                    window_size = target_vals.shape[dim] // 2
                    num_horizons = target_vals.shape[dim] - window_size + 1
                    # Compute coefficients and collect statistics over horizons
                    if features is None:
                        for start_idx in range(num_horizons):
                            end_idx = start_idx + window_size
                            if dim == 0:
                                target_window = target_vals[start_idx:end_idx, :, :].reshape(-1)
                            elif dim == 1:
                                target_window = target_vals[:, start_idx:end_idx, :].reshape(-1)
                            else:
                                target_window = target_vals[:, :, start_idx:end_idx].reshape(-1)
                            if np.isclose(np.mean(target_window), 0):
                                window_stability = np.abs(np.std(target_window))
                            else:
                                window_stability = np.abs(np.std(target_window) / np.mean(target_window))
                            eq_window_weights.append(window_stability)
                        lr += np.mean(eq_window_weights)
                    else:
                        for start_idx in range(num_horizons):
                            end_idx = start_idx + window_size
                            estimator = LinearRegression(fit_intercept=True)
                            if dim == 0:
                                target_window = target_vals[start_idx:end_idx, :, :].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[start_idx:end_idx, :, :].reshape(-1, features.shape[-1])
                                estimator.fit(feature_window, target_window,
                                              sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[start_idx:end_idx, :, :].reshape(-1))
                            elif dim == 1:
                                target_window = target_vals[:, start_idx:end_idx, :].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[:, start_idx:end_idx, :].reshape(-1, features.shape[-1])
                                estimator.fit(feature_window, target_window,
                                              sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, start_idx:end_idx, :].reshape(-1))
                            elif dim == 2:
                                target_window = target_vals[:, :, start_idx:end_idx].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[:, :, start_idx:end_idx].reshape(-1, features.shape[-1])
                                estimator.fit(feature_window, target_window,
                                              sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, :, start_idx:end_idx].reshape(-1))
                            valuable_weights = estimator.coef_
                            eq_window_weights.append(valuable_weights)
                        eq_cv = np.array([
                            np.abs(np.std(_)) if np.isclose(np.mean(_), 0)
                            else np.abs(np.std(_) / np.mean(_))
                            for _ in zip(*eq_window_weights)
                        ])
                        lr += eq_cv.sum()

            eq.fitness_calculated = True
            eq.fitness_value = lp
            eq.stability_calculated = True
            eq.coefficients_stability = lr

            print('Lr: ', lr, '\t Lp: ', lp)

    def feature_reshape(self, features_vals):
        features = features_vals[0]
        if len(features_vals) > 1:
            for i in range(1, len(features_vals)):
                features = np.vstack([features, features_vals[i]])
        features = np.vstack([features, np.ones(features_vals[0].shape)])  # Add constant feature
        features = np.transpose(features)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        return features

    def get_g_fun_vals(self):
        try:
            self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)
        except AttributeError:
            self.g_fun_vals = None
                
    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'chromosome level', 'contains suboperators', 'inplace'}


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

