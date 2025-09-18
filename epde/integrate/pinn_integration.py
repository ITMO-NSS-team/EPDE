'''

'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

from typing import Callable, Union, Dict, List
from functools import singledispatchmethod, singledispatch

from torch.nn import Sequential

from epde.structure.main_structures import Equation, SoEq
import epde.globals as global_var
from epde.evaluators import CustomEvaluator, simple_function_evaluator

from epde.integrate.interface import SystemSolverInterface
from epde.integrate.bop import BOPElement, PregenBOperator

from epde.supplementary import create_solution_net
from epde.solver.data import Domain, Conditions
from epde.solver.data import Equation as SolverEquation
from epde.solver.model import Model
from epde.solver.callbacks import cache, early_stopping, plot, adaptive_lambda
from epde.solver.optimizers.optimizer import Optimizer
from epde.solver.device import solver_device, check_device, device_type
from epde.solver.models import mat_model

'''
Specification of baseline equation solver parameters. Can be separated into its own json file for 
better handling, i.e. manual setup.
'''

BASE_COMPILING_PARAMS = {
                         'mode'                 : 'NN',
                         'lambda_operator'      : 1e1,
                         'lambda_bound'         : 1e4,
                         'normalized_loss_stop' : False,
                         'h'                    : 0.001,
                         'inner_order'          : '1',
                         'boundary_order'       : '2',
                         'weak_form'            : 'None',
                         'tol'                  : 0.005,
                         'derivative_points'    : 3
                         }

ADAM_OPTIMIZER_PARAMS = {
                         'lr'  : 1e-5,
                         'eps' : 1e-6
                         }

SGD_OPTIMIZER_PARAMS = {
                        }

LBFGS_OPTIMIZER_PARAMS = {
                          'lr'       : 1e-2,
                          'max_iter' : 10
                          }

PSO_OPTIMIZER_PARAMS = {
                        'pop_size'   : 30,
                        'b'          : 0.9,
                        'c1'         : 8e-2,
                        'c2'         : 5e-1,
                        'lr'         : 1e-3,
                        'betas'      : (0.99, 0.999),
                        'c_decrease' : False,
                        'variance'   : 1,
                        'epsilon'    : 1e-8,
                        'n_iter'     : 2000
                        }

BASE_OPTIMIZER_PARAMS = {
                         'optimizer'   : 'Adam', # Alternatively, switch to PSO, if it proves to be effective.
                         'gamma'       : 'None',
                         'decay_every' : 'None'
                         }

OPTIMIZERS_MATCHED = {
                      'Adam'  : ADAM_OPTIMIZER_PARAMS,
                      'LBFGS' : LBFGS_OPTIMIZER_PARAMS,
                      'PSO'   : PSO_OPTIMIZER_PARAMS,
                      'SGD'   : SGD_OPTIMIZER_PARAMS
                      }

BASE_CACHE_PARAMS = {
                     'cache_verbose'             : True,
                     'cache_model'               : 'None',
                     'model_randomize_parameter' : 0,
                     'clear_cache'               : False
                     }

BASE_EARLY_STOPPING_PARAMS = {
                              'eps'                     : 1e-7,
                              'loss_window'             : 100,
                              'no_improvement_patience' : 1000,
                              'patience'                : 7,
                              'abs_loss'                : 1e-5,
                              'normalized_loss'         : False,
                              'randomize_parameter'     : 1e-5,
                              'info_string_every'       : 'None',
                              'verbose'                 : False
                              }

try:
    plot_saving_directory = os.path.realpath(__file__)
except NameError:
    plot_saving_directory = 'None'
    
BASE_PLOTTER_PARAMS = {
                       'save_every'  : 1000, 
                       'print_every' : 500,
                       'title'       : 'None',
                       'img_dir'     : plot_saving_directory
                       }


BASE_TRAINING_PARAMS = {
                        'epochs'            : 1e2,  # 1e5
                        'info_string_every' : 'None', #1e4,
                        'mixed_precision'   : False,
                        'save_model'        : False,
                        'model_name'        : 'None'
                        }

def solver_formed_grid(training_grid=None, grid_var_keys=None, device = 'cpu'):
    if training_grid is None:
        keys, training_grid = global_var.grid_cache.get_all(mode = 'torch')
    elif grid_var_keys is None:
        keys, _ = global_var.grid_cache.get_all(mode = 'torch')

    assert len(keys) == training_grid[0].ndim, 'Mismatching dimensionalities'

    training_grid = np.array(training_grid).reshape((len(training_grid), -1))
    return torch.from_numpy(training_grid).T.to(device).float()


class SolverAdapter(object):
    def __init__(self, net=None, use_cache: bool = True, device: str = 'cpu'):
        self._device = device
        self.set_net(net)
        
        self._compiling_params = dict()
        self.set_compiling_params(**BASE_COMPILING_PARAMS)
        
        self._optimizer_params = dict()
        self.set_optimizer_params(**BASE_OPTIMIZER_PARAMS)
        
        self._cache_params = dict()
        self.set_cache_params(**BASE_CACHE_PARAMS)
        
        self._early_stopping_params = dict()
        self.set_early_stopping_params(**BASE_EARLY_STOPPING_PARAMS)
        
        self._ploter_params = dict()
        self.set_plotting_params(**BASE_PLOTTER_PARAMS)
        
        self._training_params = dict()
        self.set_training_params(**BASE_TRAINING_PARAMS)
        
        self.use_cache = use_cache
    
    # def set_net(self, net, get_net_kwargs: dict):
    #     self.net = net if net is None else self.get_net(**get_net_kwargs)

    @property
    def mode(self):
        return self._compiling_params['mode']
    
    def set_net(self, net: torch.nn.Sequential):
        # if self.net is not None and 
        self.net = net

    @staticmethod
    def get_net(equations, mode: str, domain: Domain, use_fourier = True, 
                fft_params: dict = {'L' : [4,], 'M' : [3,]}, device: str = 'cpu'):
        if mode == 'mat':
            return mat_model(domain, equations)
        elif mode in ['autograd', 'NN']:
            return create_solution_net(equations_num=equations.num, domain_dim=domain.dim,
                                       use_fourier=use_fourier, fourier_params=fft_params, device=device)
            

    def set_compiling_params(self, mode: str = None, lambda_operator: float = None, 
                             lambda_bound : float = None, normalized_loss_stop: bool = None,
                             h: float = None, inner_order: str = None, boundary_order: str = None,
                             weak_form: List[Callable] = None, tol: float = None, derivative_points: int = None):
        compiling_params = {'mode' : mode, 'lambda_operator' : lambda_operator, 'lambda_bound' : lambda_bound,
                            'normalized_loss_stop' : normalized_loss_stop, 'h' : h, 'inner_order' : inner_order,
                            'boundary_order' : boundary_order, 'weak_form' : weak_form, 'tol' : tol,
                            'derivative_points' : derivative_points}
        
        for param_key, param_vals in compiling_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._compiling_params[param_key] = None
                    else:
                        self._compiling_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')

    def set_optimizer_params(self, optimizer: str = None, params: Dict[str, float] = None,
                             gamma: float = None, decay_every: int = None):
        optim_params = {'optimizer' : optimizer, 'params' : params, 'gamma' : gamma, 
                        'decay_every' : decay_every}
        
        for param_key, param_vals in optim_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._optimizer_params[param_key] = None
                    else:
                        self._optimizer_params[param_key] = param_vals
                        if param_key == 'optimizer':
                            if param_vals not in ['Adam', 'SGD', 'PSO', 'LBFGS']:
                                raise ValueError(f'Unimplemented optimizer has been selected. Please, use {OPTIMIZERS_MATCHED.keys()}')
                            self._optimizer_params['params'] = OPTIMIZERS_MATCHED[param_vals]
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
    
    def set_cache_params(self, cache_verbose: bool = None, cache_model: Sequential = None, 
                         model_randomize_parameter: Union[int, float] = None, clear_cache: bool = None): # use_cache: bool = None, 

        cache_params = { 'cache_verbose' : cache_verbose, 'cache_model' : cache_model, # 'use_cache' : use_cache,
                        'model_randomize_parameter' : model_randomize_parameter, 'clear_cache' : clear_cache}

        for param_key, param_vals in cache_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._cache_params[param_key] = None
                    else:
                        self._cache_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
                    
    def set_early_stopping_params(self, eps: float = None, loss_window: int = None, no_improvement_patience: int = None,
                                  patience: int = None, abs_loss: float = None, normalized_loss: bool = None,
                                  randomize_parameter: float = None, info_string_every: int = None, verbose: bool = None):
        early_stopping_params = {'eps' : eps, 'loss_window' : loss_window, 'no_improvement_patience' : no_improvement_patience,
                                 'patience' : patience, 'abs_loss' : abs_loss, 'normalized_loss' : normalized_loss, 
                                 'randomize_parameter' : randomize_parameter, 'info_string_every' : info_string_every,
                                 'verbose' : verbose}
    
        for param_key, param_vals in early_stopping_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._early_stopping_params[param_key] = None
                    else:
                        self._early_stopping_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
                    
    def set_plotting_params(self, save_every: int = None, print_every: int = None, title: str = None,
                            img_dir: str = None):
        plotting_params = {'save_every' : save_every, 'print_every' : print_every, 
                           'print_every' : print_every, 'img_dir' : img_dir}
        for param_key, param_vals in plotting_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._ploter_params[param_key] = None
                    else:
                        self._ploter_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
    
    def set_training_params(self, epochs: int = None, info_string_every: int = None, mixed_precision: bool = None,
                            save_model: bool = None, model_name: str = None):
        training_params = {'epochs' : epochs, 'info_string_every' : info_string_every, 
                           'mixed_precision' : mixed_precision, 'save_model' : save_model, 'model_name' : model_name}

        for param_key, param_vals in training_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._training_params[param_key] = None
                    else:
                        self._training_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
                    
    def change_parameter(self, parameter: str, value, param_dict_key: str = None):
        setters = {'compiling_params'      : (BASE_COMPILING_PARAMS, self.set_compiling_params), 
                   'optimizer_params'      : (BASE_OPTIMIZER_PARAMS, self.set_optimizer_params),
                   'cache_params'          : (BASE_CACHE_PARAMS, self.set_cache_params),
                   'early_stopping_params' : (BASE_EARLY_STOPPING_PARAMS, self.set_early_stopping_params),
                   'plotting_params'       : (BASE_PLOTTER_PARAMS, self.set_plotting_params),
                   'training_params'       : (BASE_TRAINING_PARAMS, self.set_training_params)}

        if value is None:
            value = 'None'
            
        if param_dict_key is not None: # TODO: Add regular expressions
            param_labeled = {parameter : value}
            setters[param_dict_key][1](**param_labeled)
        else:
            for key, param_elem in setters.items():
                if parameter in param_elem[0].keys():
                    param_labeled = {parameter : value}
                    param_elem[1](**param_labeled)
                
    @staticmethod
    def create_domain(variables: List[str], grids : List[Union[np.ndarray, torch.Tensor]],
                      device: str = 'cpu') -> Domain:
        assert len(variables) == len(grids), f'Number of passed variables {len(variables)} does not \
            match number of grids {len(grids)}.'
        if isinstance(grids[0], np.ndarray):
            assert len(variables) == grids[0].ndim, 'Grids have to be set as a N-dimensional np.ndarrays with dim \
                matching the domain dimensionality'
        domain = Domain('uniform')

        for idx, var_name in enumerate(variables):
            var_grid = grids[idx].to(device) if isinstance(grids[idx], torch.Tensor) else torch.tensor(grids[idx]).to(device)
            var_grid = var_grid.unique().reshape(-1)
            domain.variable(variable_name = var_name, variable_set = var_grid, 
                            n_points = None)
            
        return domain

    def solve_epde_system(self, system: Union[SoEq, dict], grids: list=None, boundary_conditions=None,
                          mode='NN', data=None, use_cache: bool = False, use_fourier: bool = False,
                          fourier_params: dict = None, use_adaptive_lambdas: bool = False,
                          to_numpy: bool = False, grid_var_keys = None, *args, **kwargs):
        solver_device(device = self._device)

        if isinstance(system, SoEq):
            system_interface = SystemSolverInterface(system_to_adapt=system)
            system_solver_forms = system_interface.form(grids = grids, mode = mode)
        elif isinstance(system, dict):
            system_solver_forms = list(system.values())
        elif isinstance(system, list):
            system_solver_forms = system
        else:
            raise TypeError(f'Incorrect type of the equations passed into solver. Expected dict or SoEq, got {type(system)}.')
        
        if boundary_conditions is None:
            raise NotImplementedError('TBD')
            op_gen = PregenBOperator(system=system,
                                     system_of_equation_solver_form=[sf_labeled[1] for sf_labeled
                                                                     in system.values()])
            op_gen.generate_default_bc(vals = data, grids = grids)
            boundary_conditions = op_gen.conditions
            
        bconds_combined = Conditions()
        for cond in boundary_conditions:
            bconds_combined.operator(bnd = cond['bnd_loc'], operator = cond['bnd_op'], 
                                     value = cond['bnd_val'])

        if grids is None:
            grid_var_keys, grids = global_var.grid_cache.get_all(mode = 'torch')
        elif grid_var_keys is None:
            grid_var_keys, _ = global_var.grid_cache.get_all(mode = 'torch')

        domain = self.create_domain(grid_var_keys, grids, self._device)

        return self.solve(equations=system_solver_forms, domain = domain,
                          boundary_conditions = bconds_combined, mode = mode, use_cache = use_cache,
                          use_fourier = use_fourier, fourier_params = fourier_params, 
                          use_adaptive_lambdas = use_adaptive_lambdas, to_numpy = to_numpy)

    def solve(self, equations: Union[List, SoEq, SolverEquation], domain: Domain,
              boundary_conditions = None, mode = 'NN', use_cache: bool = False, 
              use_fourier: bool = False, fourier_params: dict = None, #  epochs = 1e3, 
              use_adaptive_lambdas: bool = False, to_numpy = False, *args, **kwargs):
    
        if isinstance(equations, SolverEquation):
            equations_prepared = equations
        else:
            equations_prepared = SolverEquation()
            for form in equations:
                equations_prepared.add(form)
        if self.net is None:
            self.net = self.get_net(equations_prepared, mode, domain, use_fourier,
                                    fourier_params, device=self._device)
        
        
        cb_early_stops = early_stopping.EarlyStopping(**self._early_stopping_params)
        callbacks = [cb_early_stops,]
        if use_cache:
            callbacks.append(cache.Cache(**self._cache_params))

        if use_adaptive_lambdas:
            callbacks.append(adaptive_lambda.AdaptiveLambda())
        
        optimizer = Optimizer(**self._optimizer_params)

        self.net.to(device = self._device)        

        model = Model(net = self.net, domain = domain, equation = equations_prepared, 
                      conditions = boundary_conditions)
        
        model.compile(**self._compiling_params)
        loss = model.train(optimizer, callbacks=callbacks, **self._training_params)
        
        grid = domain.build(mode = self.mode)
 
        grid = check_device(grid)
        
        if mode in ['NN', 'autograd'] and to_numpy:
            solution = self.net(grid).detach().cpu().numpy()
        elif mode in ['NN', 'autograd'] and not to_numpy:
            solution = self.net
        elif mode == 'mat' and to_numpy:
            solution = self.net.detach().cpu().numpy()
        elif mode == 'mat' and not to_numpy:
            solution = self.net
        else:
            raise ValueError('Incorrect mode.')
        return loss, solution
