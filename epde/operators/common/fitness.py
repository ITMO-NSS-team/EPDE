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
# DeepXDEAdapter is imported lazily inside DeepXDEBasedFitness.apply() to
# avoid triggering deepxde's import-time backend banner when no DeepXDE
# solver is used (e.g. legacy L2/L2LR fitness paths).
from epde.structure.main_structures import SoEq, Equation
from epde.operators.utils.template import CompoundOperator
import epde.globals as global_var
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import minimize
from epde.supplementary import minmax_normalize
from epde.supplementary import calculate_weights

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

        # Run sparsity when the caller explicitly asks for an
        # out-of-place fitness OR when the equation lacks a valid
        # ``weights_internal`` state. The latter can happen when an
        # upstream ``EqRightPartSelector`` exhausted its
        # ``inf_fitness_regen`` outer loop without committing a target
        # (every candidate target had all-zero LASSO survivors) -- the
        # equation reaches us with ``weights_internal_evald=False``,
        # and without this fallback ``coeff_calc`` would assert.
        need_sparsity = (force_out_of_place
                         or not getattr(objective, 'weights_internal_evald', False))
        if need_sparsity:
            self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
            # Reject degenerate candidates ONLY when this is RPS's
            # term-sweep (``force_out_of_place=True``) -- there the
            # caller compares per-target fitness values and a ``None``
            # signals "skip this target". For the final per-equation
            # fitness eval downstream (``force_out_of_place=False``)
            # we must always return a finite value so
            # ``equation.fitness_calculated`` stays True; otherwise the
            # MOEA/D objective-aggregation asserts. All-zero-weight
            # candidates fall through to the residual computation
            # below, yielding ``rl_error = ||target||`` (just the bare
            # target norm) which is a finite but poor fitness.
            if force_out_of_place and all(objective.weights_internal == 0):
                return None
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])

        _, target, features = objective.evaluate(normalize = False, return_val = False)
        if features is None:
            discr_feats = 0
        else:
            n_cols = features.shape[1] if features.ndim > 1 else 1
            mask = objective.weights_internal != 0
            if n_cols == len(mask):
                discr_feats = np.dot(features, objective.weights_internal)
            elif n_cols == int(mask.sum()):
                discr_feats = np.dot(features, objective.weights_final[:-1])
            else:
                discr_feats = np.zeros(features.shape[0])

        discr = (discr_feats + np.full(target.shape, objective.weights_final[-1]) - target)
        try:
            self.g_fun_vals = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask].reshape(-1)
        except AttributeError:
            self.g_fun_vals = None
        if self.g_fun_vals is not None and self.g_fun_vals.shape == discr.shape:
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

        # When ``use_pic=True`` is paired with ``fitness_cls=L2Fitness``,
        # ``equation_terms_stability`` is registered as an MOEA/D objective
        # but the L2 fitness path never sets ``stability_calculated``.
        # Mirror ``L2LRFitness``'s CV side-effect so the objective's
        # assertion holds without touching the L2 fitness value above.
        try:
            data_shape = global_var.grid_cache.inner_shape
            _, sw_target, sw_features = objective.evaluate(normalize=True, return_val=False)
            if sw_features is None:
                total_lr = 1.0
            else:
                if hasattr(objective, '_cached_sw_weights') and objective._cached_sw_weights is not None:
                    sw_weights = objective._cached_sw_weights
                else:
                    sw_weights = calculate_weights(
                        sw_features, sw_target, self.g_fun_vals, data_shape,
                        objective.weights_final[-1] != 0,
                    )
                sw_arr = np.array(sw_weights)
                std = sw_arr.std(axis=0, ddof=1)
                mu = sw_arr.mean(axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    cv = (std ** 2) / (mu ** 2)
                total_lr = sum(cv) / len(data_shape)
        except Exception:
            total_lr = 1.0
        objective.stability_calculated = True
        objective.coefficients_stability = total_lr

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
            if all(objective.weights_internal == 0):
                return None
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])

        if force_out_of_place:
            _, target, features = objective.evaluate(normalize=False, return_val=False)
        else:
            _, target, features = objective.evaluate(normalize=True, return_val=False)

        # self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        # _, target, features = objective.evaluate(normalize=False, return_val=False)

        self.get_g_fun_vals()

        if features is None:
            discr = target - target.mean()
        else:
            # ``features`` width depends on the ``normalize`` flag passed to
            # ``evaluate`` above: ``normalize=True`` returns all N-1
            # non-target columns; ``normalize=False`` filters to only the
            # nonzero-weight columns. ``weights_final[:-1]`` matches the
            # latter shape (nonzero count); ``weights_internal`` matches the
            # former (full N-1, with zeros). Pick whichever lines up with
            # the actual feature matrix -- same pattern as L2Fitness.apply.
            n_cols = features.shape[1] if features.ndim > 1 else 1
            mask = objective.weights_internal != 0
            if n_cols == len(mask):
                discr_feats = np.dot(features, objective.weights_internal)
            elif n_cols == int(mask.sum()):
                discr_feats = np.dot(features, objective.weights_final[:-1])
            else:
                discr_feats = np.zeros(features.shape[0])
            discr_feats = discr_feats + objective.weights_final[-1]
            discr = target - discr_feats

        rl_error = np.sum(np.abs(discr)) / np.sum(np.abs(target))

        if not (self.params['penalty_coeff'] > 0. and self.params['penalty_coeff'] < 1.):
            raise ValueError('Incorrect penalty coefficient set, value shall be in (0, 1).')

        fitness_value = rl_error

        if force_out_of_place:
            return fitness_value

        objective.aic = None
        objective.aic_calculated = True

        data_shape = global_var.grid_cache.inner_shape
        if features is None:
            # Degenerate candidate (all features pruned by sparsity).
            # Nothing to fit sliding-window weights on -- skip the CV
            # calculation and report unit stability so downstream callers
            # still get a finite value.
            total_lr = 1.0
        else:
            if hasattr(objective, '_cached_sw_weights') and objective._cached_sw_weights is not None:
                weights = objective._cached_sw_weights
            else:
                weights = calculate_weights(features, target, self.g_fun_vals, data_shape, objective.weights_final[-1] != 0)
            weights_arr = np.array(weights)
            std = weights_arr.std(axis=0, ddof=1)
            mu = weights_arr.mean(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = (std ** 2) / (mu ** 2)
            total_lr = sum(cv) / len(data_shape)

        # if force_out_of_place:
        #     return fitness_value * total_lr

        objective.fitness_calculated = True
        objective.fitness_value = fitness_value
        objective.stability_calculated = True
        objective.coefficients_stability = total_lr

    def get_g_fun_vals(self):
        try:
            self.g_fun_vals = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask].reshape(-1)
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
            training_params = {'epochs': 1e3, 'info_string_every' : 1e3}
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
            training_params = {'epochs': 1e3, 'info_string_every': 1e3}
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
        g_mask = global_var.grid_cache.g_func_mask
        grids = [grid[g_mask] for grid in grids]
        grids = torch.stack([grid.reshape(-1) for grid in grids], dim=1).float()
        solution = solution_nn(grids).detach().cpu().numpy()
        self.g_fun_vals = global_var.grid_cache.g_func[g_mask]

        if force_out_of_place:
            sum_err = 0

        for eq_idx, eq in enumerate(objective.vals):
            # Calculate p-loss
            if torch.isnan(loss_add):
                lp = 2 * LOSS_NAN_VAL
            else:
                referential_data = global_var.tensor_cache.get((eq.main_var_to_explain, (1.0,)))
                discr = solution[..., eq_idx] - referential_data.reshape(solution[..., eq_idx].shape)
                discr = np.multiply(discr, self.g_fun_vals.reshape(discr.shape))
                # rl_error = np.sqrt(np.mean(discr ** 2))
                # rl_error = np.sum(np.abs(discr)) / np.sum(np.abs(referential_data.reshape(solution[..., eq_idx].shape))) * 100
                rl_error = np.mean(discr ** 2)

                print(f'fitness error is {rl_error}, while loss addition is {float(loss_add)}')
                lp = rl_error + self.params['pinn_loss_mult'] * float(
                    loss_add)  # TODO: make pinn_loss_mult case dependent

            if force_out_of_place:
                sum_err += lp
                continue

            eq.aic_calculated = True

            # Calculate r-loss
            data_shape = global_var.grid_cache.inner_shape
            _, target, features = eq.evaluate(normalize=True, return_val=False)
            if hasattr(eq, '_cached_sw_weights') and eq._cached_sw_weights is not None:
                weights = eq._cached_sw_weights
            else:
                weights = calculate_weights(features, target, self.g_fun_vals, data_shape)
            weights_arr = np.array(weights)
            std = weights_arr.std(axis=0, ddof=1)
            mu = weights_arr.mean(axis=0)

            # Safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = (std ** 2) / (mu ** 2)

            total_lr = sum(cv) / len(data_shape)

            eq.fitness_calculated = True
            eq.fitness_value = lp
            eq.stability_calculated = True
            eq.coefficients_stability = total_lr

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
            self.g_fun_vals = global_var.grid_cache.g_func_flat
        except AttributeError:
            self.g_fun_vals = None

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'chromosome level', 'contains suboperators', 'inplace'}

class DeepXDEBasedFitness(CompoundOperator):
    key = 'DeepXDEBasedFitness'

    def __init__(self, param_keys: list):
        super().__init__(param_keys)
        self.adapter = None

    def set_adapter(self, config: dict = None, pretrained_net=None):
        if self.adapter is None:
            from epde.integrate.deepxde_integration import DeepXDEAdapter
            cfg = self.params.get('deepxde_config', {}) if config is None else config
            self.adapter = DeepXDEAdapter(pretrained_net=pretrained_net, **cfg)

    def apply(self, objective, arguments: dict, force_out_of_place: bool = False):
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)

        if force_out_of_place:
            self.suboperators['sparsity'].apply(objective, subop_args.get('sparsity', {}))
        self.suboperators['coeff_calc'].apply(objective, subop_args.get('coeff_calc', {}))

        try:
            pretrained_net = deepcopy(global_var.solution_guess_nn)
        except:
            pretrained_net = None
        self.set_adapter(pretrained_net=pretrained_net)

        keys, grids = global_var.grid_cache.get_all(mode='numpy')

        if isinstance(objective, SoEq):
            data_list = []
            for var_name in objective.vars_to_describe:
                eq = objective.vals[var_name]
                _, target, _ = eq.evaluate(normalize=False, return_val=False)
                data_list.append(target.reshape(-1))
        else:
            _, target, _ = objective.evaluate(normalize=False, return_val=False)
            data_list = [target.reshape(-1)]

        try:
            solution_list, loss = self.adapter.solve(equation_or_system=objective,
                                                     grids=grids,
                                                     data=data_list)
            if np.isnan(loss):
                raise ValueError("NaN loss")

            if isinstance(objective, SoEq):
                for idx, (var_name, eq) in enumerate({val: objective.vals[val] for val in objective.vars_to_describe}.items()):
                    err = self._compute_error(solution_list[idx], data_list[idx], eq)
                    if force_out_of_place:
                        pass
                    else:
                        eq.fitness_value = err
                        eq.fitness_calculated = True
                        self._compute_stability_for_equation(eq)
            else:
                solution = solution_list[0]
                data = data_list[0]
                err = self._compute_error(solution, data, objective)
                if force_out_of_place:
                    return err
                else:
                    objective.fitness_value = err
                    objective.fitness_calculated = True
                    self._compute_stability_for_equation(objective)
        except Exception as e:
            print(f'[DeepXDEBasedFitness] DeepXDE solve failed: {e}')
            fitness_value = 1e7
            if force_out_of_place:
                return fitness_value
            else:
                objective.fitness_value = fitness_value
                objective.fitness_calculated = True
                return

        if force_out_of_place and isinstance(objective, SoEq):
            total_err = np.mean([eq.fitness_value for eq in objective.vals.values()])
            return total_err

    def _compute_error(self, solution, data, eq):
        mask = global_var.grid_cache.g_func_mask
        mask_flat = mask.flatten()
        masked_solution = solution[mask_flat]
        masked_data = data
        metric = self.params.get('error_metric', 'rmse')
        if metric == 'rmse':
            err = np.sqrt(np.mean((masked_solution - masked_data) ** 2))
        elif metric == 'l2':
            err = np.linalg.norm(masked_solution - masked_data, ord=2)
        elif metric == 'mae':
            err = np.mean(np.abs(masked_solution - masked_data))
        else:
            err = np.sqrt(np.mean((masked_solution - masked_data) ** 2))
        if np.sum(eq.weights_final) == 0:
            err /= self.params.get('penalty_coeff', 0.2)
        return err

    def _compute_stability_for_equation(self, eq: Equation):
        # Повторно вычисляется evaluate
        _, target, features = eq.evaluate(normalize=False, return_val=False)
        data_shape = global_var.grid_cache.inner_shape
        self.get_g_fun_vals()
        weights = calculate_weights(features, target, self.g_fun_vals, data_shape)
        weights_arr = np.array(weights)
        std = weights_arr.std(axis=0, ddof=1)
        mu = weights_arr.mean(axis=0)
        cv = (std ** 2) / (mu ** 2)
        total_lr = np.sum(cv) / len(data_shape)
        eq.coefficients_stability = total_lr
        eq.stability_calculated = True

    def get_g_fun_vals(self):
        try:
            self.g_fun_vals = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask].reshape(-1)
        except:
            self.g_fun_vals = None

    def use_default_tags(self):
        self._tags = {'fitness evaluation', 'gene level', 'contains suboperators', 'inplace'}

def plot_data_vs_solution(grid, data, solution):
    if grid.shape[1]==2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1),
                        solution.reshape(-1), cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()
        plt.close(fig)
    if grid.shape[1]==1:
        fig = plt.figure()
        plt.scatter(grid.reshape(-1), solution.reshape(-1), color = 'r')
        plt.scatter(grid.reshape(-1), data.reshape(-1), color = 'k')
        plt.show()
        plt.close(fig)
    else:
        raise Exception('Infeasible dimensionality of the input dataset.')

