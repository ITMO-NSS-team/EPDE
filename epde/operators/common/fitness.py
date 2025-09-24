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
        Calculates the fitness of an equation by evaluating its ability to fit the target data, considering sparsity and coefficient values. The fitness value, reflecting the equation's accuracy, is then stored within the equation object.
        
                Args:
                    objective (Equation): The equation object to evaluate. Its `fitness_value` attribute will be updated with the calculated fitness.
                    arguments (dict): A dictionary containing arguments needed for sub-operators like sparsity and coefficient calculation.
                    force_out_of_place (bool, optional): If True, the fitness value is returned directly instead of being stored in the equation object. Defaults to False.
        
                Returns:
                    float, None: If `force_out_of_place` is True, returns the calculated fitness value. Otherwise, returns None. The fitness value is also stored in `objective.fitness_value`.
        
                Why:
                    This method quantifies how well a given equation represents the underlying relationships in the data. By calculating a fitness score, the evolutionary algorithm can effectively search for equations that accurately capture the system's dynamics.
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
        
        if force_out_of_place:
            return fitness_value
        else:
            objective.fitness_calculated = True
            objective.fitness_value = fitness_value

    def use_default_tags(self):
        """
        Sets the operator's tags to a predefined default. This ensures a consistent and informative categorization of the operator, facilitating its proper use and interpretation within the evolutionary process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        
                Class Fields Initialized:
                    _tags (set): A set containing default tags: 'fitness evaluation', 'gene level', 'contains suboperators', and 'inplace'. These tags provide essential metadata about the operator's functionality and behavior.
        """
        self._tags = {'fitness evaluation', 'gene level', 'contains suboperators', 'inplace'}


class L2LRFitness(CompoundOperator):
    """
    Represents the L2-LR fitness evaluation method.
    
        Calculates fitness based on the L2 loss with L2 regularization.
    
        Class Attributes:
        - lambd
        - use_g_fun
        - g_fun_key
    
        Class Methods:
        - apply:
    """

    key = 'DiscrepancyBasedFitnessWithCV'

    def apply(self, objective: Equation, arguments: dict, force_out_of_place: bool = False):
        """
        Calculates the fitness of an equation by evaluating its ability to capture the underlying dynamics of the data. The fitness value, representing the equation's accuracy and stability, is stored directly within the equation object.
        
                Args:
                    objective (Equation): The equation to evaluate. Its `fitness_value` attribute will be updated with the calculated fitness.
                    arguments (dict): A dictionary containing arguments needed for sub-operators.
                    force_out_of_place (bool, optional): If True, the fitness value is returned instead of being stored in the equation object. Defaults to False.
        
                Returns:
                    float, optional: The fitness value, only returned if `force_out_of_place` is True.
        
                Why:
                This method quantifies how well a candidate equation represents the relationships present in the data. By calculating a fitness score, the evolutionary algorithm can effectively search for equations that accurately model the observed system behavior.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)

        self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])

        _, target, features = objective.evaluate(normalize=False, return_val=False)

        self.get_g_fun_vals()

        try:
            if features is None:
                maximum = target.max(axis=0)
                minimum = target.min(axis=0)
                discr = (target - target.mean(axis=0) - minimum) / (maximum - minimum)
            else:
                discr_feats = np.dot(features, objective.weights_final[:-1][objective.weights_internal != 0])
                discr_feats = discr_feats + np.full(target.shape, objective.weights_final[-1])
                maximum = np.max([discr_feats.max(axis=0), target.max(axis=0)])
                minimum = np.min([discr_feats.min(axis=0), target.min(axis=0)])
                discr = ((discr_feats - minimum) - (target - minimum)) / (maximum - minimum)
            discr = np.multiply(discr, self.g_fun_vals)
            rl_error = np.linalg.norm(discr, ord=2)
        except ValueError:
            raise ValueError('An error in getting weights ')

        if not (self.params['penalty_coeff'] > 0. and self.params['penalty_coeff'] < 1.):
            raise ValueError('Incorrect penalty coefficient set, value shall be in (0, 1).')

        fitness_value = rl_error
        # if np.sum(objective.weights_final) == 0:
        #     fitness_value /= self.params['penalty_coeff']

        if force_out_of_place:
            return fitness_value

        # discr = np.mean(discr ** 2)
        # ll = np.log(discr)
        # aic = 2 * len(objective.weights_final) - 2 * ll
        # ssr = np.sum(discr ** 2)
        # n = len(target)
        # llf = - n / 2 * np.log(2 * np.pi) - n / 2 * np.log(ssr / n) - n / 2
        # aic = 2 * len([_ for _ in objective.weights_final if _ != 0]) - 2 * llf
        # aic = np.log(n) * len([_ for _ in objective.weights_final if _ != 0]) - 2 * llf
        # objective.aic = 1/(1 + np.exp(- 1e-4 * ll))

        # if force_out_of_place:
        #     return 1 / (np.exp(-aic / 3e5))

        # objective.aic = 1 / (np.exp(-aic / 3e5))
        objective.aic = None
        objective.aic_calculated = True
        # print(aic)
        # print(len([_ for _ in objective.weights_final if _ !=0]))
        # print(objective.aic)
        assert objective.simplified, 'Trying to evaluate not simplified equation.'

        # Calculate r-loss
        data_shape = global_var.grid_cache.g_func.shape
        target = objective.structure[objective.target_idx]
        target_vals = target.evaluate(False).reshape(*data_shape)
        features_vals = []
        nonzero_features_indexes = []

        for i in range(len(objective.structure)):
            if i == objective.target_idx:
                continue
            idx = i if i < objective.target_idx else i - 1
            if objective.weights_internal[idx] != 0:
                features_vals.append(objective.structure[i].evaluate(False))
                nonzero_features_indexes.append(idx)

        if target_vals.ndim == 1:
            window_size = len(target_vals) // 2
            num_horizons = len(target_vals) - window_size + 1
            if window_size < 15:
                step_size = 1
            else:
                step_size = num_horizons // 30
            eq_window_weights = []
            # Compute coefficients and collect statistics over horizons
            if len(features_vals) == 0:
                for start_idx in range(0, num_horizons, step_size):
                    end_idx = start_idx + window_size
                    target_window = target_vals[start_idx:end_idx]
                    if np.isclose(np.sqrt(np.mean(np.power(target_window, 2))), 0, atol=1e-10):
                        window_stability = np.abs(np.std(target_window))
                    else:
                        window_stability = np.abs(np.std(target_window) / np.sqrt(np.mean(np.power(target_window, 2))))
                    eq_window_weights.append(window_stability)
                lr = np.mean(eq_window_weights)
            else:
                features = self.feature_reshape(features_vals)
                for start_idx in range(0, num_horizons, step_size):
                    end_idx = start_idx + window_size
                    target_window = target_vals[start_idx:end_idx]
                    feature_window = features[start_idx:end_idx, :]
                    estimator = LinearRegression(fit_intercept=False)
                    estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals[start_idx:end_idx])
                    valuable_weights = estimator.coef_[:-1]
                    eq_window_weights.append(valuable_weights)
                eq_cv = np.array([
                    np.abs(np.std(_)) if np.isclose(np.sqrt(np.mean(np.power(_, 2))), 0, atol=1e-10)
                    else np.abs(np.std(_) / np.sqrt(np.mean(np.power(_, 2))))
                    for _ in zip(*eq_window_weights)
                ])
                lr = eq_cv.mean()

        elif target_vals.ndim == 2:
            lr = 0
            for dim in range(target_vals.ndim):
                eq_window_weights = []
                window_size = target_vals.shape[dim] // 2
                num_horizons = target_vals.shape[dim] - window_size + 1
                if window_size < 15:
                    step_size = 1
                else:
                    step_size = num_horizons // 30
                # Compute coefficients and collect statistics over horizons
                if len(features_vals) == 0:
                    for start_idx in range(0, num_horizons, step_size):
                        end_idx = start_idx + window_size
                        if dim == 0:
                            target_window = target_vals[start_idx:end_idx, :].reshape(-1)
                        else:
                            target_window = target_vals[:, start_idx:end_idx].reshape(-1)
                        if np.isclose(np.sqrt(np.mean(np.power(target_window, 2))), 0, atol=1e-10):
                            window_stability = np.abs(np.std(target_window))
                        else:
                            window_stability = np.abs(np.std(target_window) / np.sqrt(np.mean(np.power(target_window, 2))))
                        eq_window_weights.append(window_stability)
                    lr += np.mean(eq_window_weights)
                else:
                    features = self.feature_reshape(features_vals)
                    for start_idx in range(0, num_horizons, step_size):
                        end_idx = start_idx + window_size
                        estimator = LinearRegression(fit_intercept=False)
                        if dim == 0:
                            target_window = target_vals[start_idx:end_idx, :].reshape(-1)
                            feature_window = features.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1, features.shape[-1])
                            estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1))
                        else:
                            target_window = target_vals[:, start_idx:end_idx].reshape(-1)
                            feature_window = features.reshape(*data_shape, -1)[:, start_idx:end_idx].reshape(-1, features.shape[-1])
                            estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, start_idx:end_idx].reshape(-1))
                        valuable_weights = estimator.coef_[:-1]
                        eq_window_weights.append(valuable_weights)
                    eq_cv = np.array([
                        np.abs(np.std(_)) if np.isclose(np.sqrt(np.mean(np.power(_, 2))), 0, atol=1e-10)
                        else np.abs(np.std(_) / np.sqrt(np.mean(np.power(_, 2))))
                        for _ in zip(*eq_window_weights)
                    ])
                    lr += eq_cv.mean()

        elif target_vals.ndim == 3:
            lr = 0
            for dim in range(target_vals.ndim):
                eq_window_weights = []
                window_size = target_vals.shape[dim] // 2
                num_horizons = target_vals.shape[dim] - window_size + 1
                # Compute coefficients and collect statistics over horizons
                if len(features_vals) == 0:
                    for start_idx in range(num_horizons):
                        end_idx = start_idx + window_size
                        if dim == 0:
                            target_window = target_vals[start_idx:end_idx, :, :].reshape(-1)
                        elif dim == 1:
                            target_window = target_vals[:, start_idx:end_idx, :].reshape(-1)
                        else:
                            target_window = target_vals[:, :, start_idx:end_idx].reshape(-1)
                        if np.isclose(np.sqrt(np.mean(np.power(target_window, 2))), 0, atol=1e-10):
                            window_stability = np.abs(np.std(target_window))
                        else:
                            window_stability = np.abs(np.std(target_window) / np.sqrt(np.mean(np.power(target_window, 2))))
                        eq_window_weights.append(window_stability)
                    lr += np.mean(eq_window_weights)
                else:
                    features = self.feature_reshape(features_vals)
                    for start_idx in range(num_horizons):
                        end_idx = start_idx + window_size
                        estimator = LinearRegression(fit_intercept=False)
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
                        valuable_weights = estimator.coef_[:-1]
                        eq_window_weights.append(valuable_weights)
                    eq_cv = np.array([
                        np.abs(np.std(_)) if np.isclose(np.sqrt(np.mean(np.power(_, 2))), 0, atol=1e-10)
                        else np.abs(np.std(_) / np.sqrt(np.mean(np.power(_, 2))))
                        for _ in zip(*eq_window_weights)
                    ])

                    lr += eq_cv.mean()

        objective.fitness_calculated = True
        objective.fitness_value = fitness_value
        objective.stability_calculated = True
        objective.coefficients_stability = lr

    def feature_reshape(self, features_vals):
        """
        Reshapes and prepares feature values for equation discovery.
        
                This method consolidates feature value arrays, incorporates a constant
                feature, and transposes the result to create a standardized input
                format suitable for the equation learning process. The constant feature
                allows the model to account for bias or offset terms in the discovered equations.
        
                Args:
                    features_vals: A list of NumPy arrays, where each array
                        represents a set of feature values.
        
                Returns:
                    A NumPy array containing the reshaped feature values. The array
                    is transposed, has a constant feature added, and is guaranteed
                    to be a 2D array.
        """
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
        """
        Retrieves and prepares the g-function values for fitness evaluation.
        
        The g-function, representing a component of the discovered differential equation, 
        is retrieved from the global grid cache. If found, it's reshaped into a 1D array 
        and stored in `self.g_fun_vals` for efficient fitness calculation. This is done to 
        speed up fitness evaluation by pre-calculating and storing this component. If the 
        g-function is not available, `self.g_fun_vals` is set to None, indicating its absence 
        in the current equation candidate.
        
        Args:
            self: The instance of the L2LRFitness class.
        
        Returns:
            np.ndarray: The reshaped g-function values if available in the grid cache; otherwise, None.
        
        Attributes:
            g_fun_vals (np.ndarray): Stores the reshaped g-function values, or None if the g-function is not available.
        """
        try:
            self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)
        except AttributeError:
            self.g_fun_vals = None

    def use_default_tags(self):
        """
        Applies a predefined set of tags to categorize the fitness evaluation operator.
        
        This tagging helps in organizing and filtering operators based on their characteristics,
        allowing for more efficient management and selection of appropriate operators
        during the equation discovery process.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Initializes:
            _tags (set): A set containing default tags: 'fitness evaluation', 'gene level', 'contains suboperators', and 'inplace'.
        """
        self._tags = {'fitness evaluation', 'gene level', 'contains suboperators', 'inplace'}

class SolverBasedFitness(CompoundOperator):
    """
    A base class for fitness evaluation based on solving a system of equations.
    
        This class provides a framework for evaluating the fitness of a solution by
        solving a system of equations and comparing the solution to reference data.
    
        Class Methods:
        - __init__
        - set_adapter
        - apply
        - use_default_tags
    """

    # To be modified to include physics-informed information criterion (PIC)

    key = 'SolverBasedFitness'
    
    def __init__(self, param_keys: list):
        """
        Initializes the object with parameter keys and sets the adapter to None.
        
        This initialization prepares the fitness evaluation component by storing the parameter keys 
        associated with the equation's structure. The adapter, responsible for interfacing with a 
        numerical solver, is initially set to None, as it will be configured later based on the 
        specific equation and solver chosen during the evolutionary process.
        
        Args:
            param_keys: A list of parameter keys, defining the variables and constants in the equation.
        
        Returns:
            None.
        
        Class Fields:
            adapter (Any): An adapter object for numerical solvers, initialized to None.
        """
        super().__init__(param_keys)
        self.adapter = None

    def set_adapter(self, net = None):
        """
        Sets up the solver adapter, which is crucial for training the neural network to approximate the solution of the discovered differential equation.
        
                This method initializes or updates the `SolverAdapter` with a given neural network. If no adapter exists or a new network is provided, a new `SolverAdapter` is created and configured with default parameters for compilation, optimization, training, and early stopping. This ensures that the neural network is properly set up to learn the solution of the differential equation.
        
                Args:
                    net: The neural network to be used by the adapter. If None, the
                        existing network is used, or a new one is created if no adapter
                        exists.
        
                Returns:
                    None
        
                Class Fields Initialized:
                    adapter (SolverAdapter): An instance of the SolverAdapter class,
                        responsible for managing the optimization and training of the
                        neural network. It is initialized with the provided network (or a
                        new one if none is provided), caching disabled, and the specified
                        device (CUDA if available, otherwise CPU).
        """

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
        """
        Applies the operator to the given objective function.
        
                This method orchestrates the application of sub-operators to refine equation structure and coefficient values,
                solves the EPDE system using a neural network-based solver, and calculates the fitness value based on the solution's accuracy
                compared to referential data. This process evaluates how well a candidate equation represents the underlying dynamics.
        
                Args:
                    objective: The objective function (SoEq) representing the EPDE system to be solved.
                    arguments: A dictionary containing arguments for the sub-operators, such as sparsity and coefficient calculation.
                    force_out_of_place: A boolean indicating whether to force out-of-place updates to the fitness values.
        
                Returns:
                    None. The method updates the fitness values of the equations within the objective,
                    reflecting the equation's ability to accurately model the system.
        """
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
        """
        Sets the operator's tags to a predefined default.
        
        This ensures the operator is configured with a standard set of properties
        related to its behavior within the evolutionary process. This includes
        characteristics like its role in fitness evaluation, the level of genetic
        material it operates on, its composition of sub-operators, and whether
        it modifies data directly.
        
        Args:
            self: The operator instance.
        
        Returns:
            None.
        
        Class Fields:
            _tags (set): A set containing default tags for the operator, initialized to {'fitness evaluation', 'chromosome level', 'contains suboperators', 'inplace'}.
        """
        self._tags = {'fitness evaluation', 'chromosome level', 'contains suboperators', 'inplace'}


class PIC(CompoundOperator):
    """
    Represents a Physics-Informed Controller (PIC) for solving EPDE systems.
    
        The PIC class orchestrates the application of suboperators, solves the equation
        using a neural network adapter, and calculates fitness and stability metrics.
    
        Class Methods:
        - __init__
        - set_adapter
        - apply
        - feature_reshape
        - get_g_fun_vals
        - use_default_tags
    """


    key = 'PIC'

    def __init__(self, param_keys: list):
        """
        Initializes the PIC object with parameter keys and sets the adapter to None.
        
        This initialization prepares the object to work with specific parameters 
        required for equation discovery. The adapter, initially set to None, 
        will later be used to facilitate the interaction with different data 
        sources or numerical solvers during the equation search process.
        
        Args:
            param_keys: A list of parameter keys that define the variables 
                and constants to be used in the equation discovery process.
        
        Returns:
            None.
        
        Class Fields:
            adapter (Any): An adapter object, initialized to None. It will be 
                used for data handling and interaction with numerical solvers.
        """
        super().__init__(param_keys)
        self.adapter = None

    def set_adapter(self, net=None):
        """
        Sets up the solver adapter with default configurations for equation discovery.
        
                This method initializes or resets the solver adapter with default parameters
                for compiling, optimization, early stopping, and training. It configures
                the device (CUDA if available, otherwise CPU) to ensure efficient computation
                during the equation discovery process. This setup is crucial for exploring
                the search space of possible differential equations and identifying the best
                candidates that fit the observed data.
        
                Args:
                  net: An optional neural network to use with the adapter. If None, the
                    existing network in the adapter is used (if one exists).
        
                Returns:
                  None
        
                Class Fields Initialized:
                  adapter (SolverAdapter): An instance of the SolverAdapter class, configured
                    with the specified network and default parameters. This adapter is used
                    for training the neural network.
        """

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
        """
        Applies suboperators to refine equation structure and calculate coefficients, then solves the EPDE system.
        
                This method orchestrates the application of suboperators to identify the most relevant terms and their coefficients within the equation. It then leverages a neural network adapter to solve the equation and evaluates the fitness and stability of the resulting solution. This process aims to find a balance between model complexity and accuracy in representing the underlying dynamics of the system.
        
                Args:
                    objective: The objective function (SoEq) representing the EPDE system to be solved.
                    arguments: A dictionary containing arguments for the suboperators.
                    force_out_of_place: A boolean indicating whether to force out-of-place
                        operations.
        
                Returns:
                    float: The sum of errors if `force_out_of_place` is True, otherwise None.
        """
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

        if force_out_of_place:
            sum_err = 0

        data_shape = global_var.grid_cache.g_func.shape

        for eq_idx, eq in enumerate(objective.vals):
            # Calculate p-loss
            if torch.isnan(loss_add):
                lp = 2 * LOSS_NAN_VAL
            else:
                print(f'solution shape {solution.shape}')
                print(f'solution[..., eq_idx] {solution[..., eq_idx].shape}, eq_idx {eq_idx}')
                referential_data = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))
                maximum = np.max([referential_data.max(axis=0), solution[..., eq_idx].max(axis=0)])
                minimum = np.min([referential_data.min(axis=0), solution[..., eq_idx].min(axis=0)])
                discr = ((solution[..., eq_idx] - minimum) - (referential_data - minimum)) / (maximum - minimum) # Normalized
                # discr = (solution[..., eq_idx] - referential_data.reshape(solution[..., eq_idx].shape))  # Default
                discr = np.multiply(discr, self.g_fun_vals.reshape(discr.shape))
                rl_error = np.linalg.norm(discr, ord=2)

                print(f'fitness error is {rl_error}, while loss addition is {float(loss_add)}')
                lp = rl_error + self.params['pinn_loss_mult'] * float(
                    loss_add) * 0  # TODO: make pinn_loss_mult case dependent
                # if np.sum(eq.weights_final) == 0:
                #     lp /= self.params['penalty_coeff']

                ssr = np.sum(discr ** 2)
                n = len(discr)
                llf = - n / 2 * np.log(2 * np.pi) - n / 2 * np.log(ssr / n) - n / 2
                # aic = 2 * len([_ for _ in objective.weights_final if _ != 0]) - 2 * llf
                aic = np.log(n) * len([_ for _ in eq.weights_final if _ != 0]) - 2 * llf
                # objective.aic = 1/(1 + np.exp(- 1e-4 * ll))

            if force_out_of_place:
                sum_err += lp
                continue

            eq.aic = 1 / (np.exp(-aic / 3e5))
            # objective.aic = aic
            eq.aic_calculated = True

            # Calculate r-loss
            data_shape = global_var.grid_cache.g_func.shape
            target = eq.structure[eq.target_idx]
            target_vals = target.evaluate(False).reshape(*data_shape)
            features_vals = []
            nonzero_features_indexes = []

            for i in range(len(eq.structure)):
                if i == eq.target_idx:
                    continue
                idx = i if i < eq.target_idx else i - 1
                if eq.weights_internal[idx] != 0:
                    features_vals.append(eq.structure[i].evaluate(False))
                    nonzero_features_indexes.append(idx)

            if target_vals.ndim == 1:
                window_size = len(target_vals) // 2
                num_horizons = len(target_vals) - window_size + 1
                eq_window_weights = []
                # Compute coefficients and collect statistics over horizons
                if len(features_vals) == 0:
                    for start_idx in range(num_horizons):
                        end_idx = start_idx + window_size
                        target_window = target_vals[start_idx:end_idx]
                        eq_window_weights.append(np.abs(np.std(target_window) / np.mean(target_window)))
                    lr = np.mean(eq_window_weights)
                else:
                    features = self.feature_reshape(features_vals)
                    for start_idx in range(num_horizons):
                        end_idx = start_idx + window_size
                        target_window = target_vals[start_idx:end_idx]
                        feature_window = features[start_idx:end_idx, :]
                        estimator = LinearRegression(fit_intercept=False)
                        estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals[start_idx:end_idx])
                        valuable_weights = estimator.coef_[:-1]
                        eq_window_weights.append(valuable_weights)
                    eq_cv = np.array([np.abs(np.std(_) / np.mean(_)) for _ in zip(*eq_window_weights)])
                    lr = eq_cv.mean()

            elif target_vals.ndim == 2:
                lr = 0
                for dim in range(target_vals.ndim):
                    eq_window_weights = []
                    window_size = target_vals.shape[dim] // 2
                    num_horizons = target_vals.shape[dim] - window_size + 1
                    # Compute coefficients and collect statistics over horizons
                    if len(features_vals) == 0:
                        for start_idx in range(num_horizons):
                            end_idx = start_idx + window_size
                            if dim == 0:
                                target_window = target_vals[start_idx:end_idx, :].reshape(-1)
                            else:
                                target_window = target_vals[:, start_idx:end_idx].reshape(-1)
                            eq_window_weights.append(np.abs(np.std(target_window) / np.mean(target_window)))
                        lr += np.mean(eq_window_weights)
                    else:
                        features = self.feature_reshape(features_vals)
                        for start_idx in range(num_horizons):
                            end_idx = start_idx + window_size
                            estimator = LinearRegression(fit_intercept=False)
                            if dim == 0:
                                target_window = target_vals[start_idx:end_idx, :].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1, features.shape[-1])
                                estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[start_idx:end_idx, :].reshape(-1))
                            else:
                                target_window = target_vals[:, start_idx:end_idx].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[:, start_idx:end_idx].reshape(-1, features.shape[-1])
                                estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, start_idx:end_idx].reshape(-1))
                            valuable_weights = estimator.coef_[:-1]
                            eq_window_weights.append(valuable_weights)
                        eq_cv = np.array([np.abs(np.std(_) / np.mean(_)) for _ in zip(*eq_window_weights)])
                        lr += eq_cv.mean()

            elif target_vals.ndim == 3:
                lr = 0
                for dim in range(target_vals.ndim):
                    eq_window_weights = []
                    window_size = target_vals.shape[dim] // 2
                    num_horizons = target_vals.shape[dim] - window_size + 1
                    # Compute coefficients and collect statistics over horizons
                    if len(features_vals) == 0:
                        for start_idx in range(num_horizons):
                            end_idx = start_idx + window_size
                            if dim == 0:
                                target_window = target_vals[start_idx:end_idx, :, :].reshape(-1)
                            elif dim == 1:
                                target_window = target_vals[:, start_idx:end_idx, :].reshape(-1)
                            else:
                                target_window = target_vals[:, :, start_idx:end_idx].reshape(-1)
                            eq_window_weights.append(np.abs(np.std(target_window) / np.mean(target_window)))
                        lr += np.mean(eq_window_weights)
                    else:
                        features = self.feature_reshape(features_vals)
                        for start_idx in range(num_horizons):
                            end_idx = start_idx + window_size
                            estimator = LinearRegression(fit_intercept=False)
                            if dim == 0:
                                target_window = target_vals[start_idx:end_idx, :, :].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[start_idx:end_idx, :, :].reshape(-1,  features.shape[-1])
                                estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[start_idx:end_idx, :, :].reshape(-1))
                            elif dim == 1:
                                target_window = target_vals[:, start_idx:end_idx, :].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[:, start_idx:end_idx, :].reshape(-1, features.shape[-1])
                                estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, start_idx:end_idx, :].reshape(-1))
                            elif dim == 2:
                                target_window = target_vals[:, :, start_idx:end_idx].reshape(-1)
                                feature_window = features.reshape(*data_shape, -1)[:, :, start_idx:end_idx].reshape(-1, features.shape[-1])
                                estimator.fit(feature_window, target_window, sample_weight=self.g_fun_vals.reshape(*data_shape, -1)[:, :, start_idx:end_idx].reshape(-1))
                            valuable_weights = estimator.coef_[:-1]
                            eq_window_weights.append(valuable_weights)
                        eq_cv = np.array([np.abs(np.std(_) / np.mean(_)) for _ in zip(*eq_window_weights)])
                        lr += eq_cv.mean()

            eq.fitness_calculated = True
            eq.fitness_value = lp

            eq.stability_calculated = True
            eq.coefficients_stability = lr

            print('Lr: ', lr, '\t Lp: ', lp)

    def feature_reshape(self, features_vals):
        """
        Reshapes and prepares feature values for equation discovery.
        
                This method consolidates feature value arrays, introduces a constant
                feature to account for potential bias or offset in the equation,
                transposes the data for compatibility with the equation fitting process,
                and ensures the output is a 2D array suitable for subsequent
                mathematical operations.
        
                Args:
                    features_vals: A list of NumPy arrays, each representing a set of
                        feature values.
        
                Returns:
                    A NumPy array containing the reshaped feature values. The array
                    is transposed, has a constant feature added, and is guaranteed
                    to be 2D.
        """
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
        """
        Retrieves the g-function values from the global grid cache to enable efficient computation of particle interactions.
        
                The method attempts to retrieve the g-function from the global grid cache. If found, it reshapes the data for use in subsequent calculations and stores it in the `g_fun_vals` attribute. If the g-function is not available, `g_fun_vals` is set to None, indicating that this component of the calculation cannot proceed. This caching mechanism avoids redundant computations and speeds up the overall process of modeling the system.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None. The method updates the `g_fun_vals` attribute of the object.
        
                Class Fields Initialized:
                    g_fun_vals (numpy.ndarray or None): The reshaped g-function values
                        from the global grid cache, or None if the g-function is not
                        available.
        """
        try:
            self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)
        except AttributeError:
            self.g_fun_vals = None
                
    def use_default_tags(self):
        """
        Resets the operator's tags to the default set.
        
        This ensures the operator is correctly categorized for equation discovery, 
        allowing it to be properly utilized within the evolutionary process. 
        The default tags reflect common characteristics of operators used in this framework.
        
        Args:
            self: The operator instance.
        
        Returns:
            None.
        
        Class Fields:
            _tags (set): A set containing the tags associated with the operator.
                Initialized to {'fitness evaluation', 'chromosome level', 'contains suboperators', 'inplace'}.
        """
        self._tags = {'fitness evaluation', 'chromosome level', 'contains suboperators', 'inplace'}


def plot_data_vs_solution(grid, data, solution):
    """
    Plots the data and solution against the grid to validate discovered equations.
        
        This method visualizes the data alongside the solution obtained from the
        identified differential equation on the provided grid. It supports 1D and 2D grids,
        using scatter plots for 1D and surface plots for 2D, allowing for a visual
        comparison between the data and the solution. This comparison helps assess
        how well the discovered equation fits the observed data.
        
        Args:
            grid (numpy.ndarray): The grid points where the data and solution are defined.
            data (numpy.ndarray): The data values at the grid points.
            solution (numpy.ndarray): The solution values at the grid points.
        
        Returns:
            None: Displays the plot.
    """
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

