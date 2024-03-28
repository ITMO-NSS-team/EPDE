'''

'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

from typing import Callable, Union, Dict, List
from types import FunctionType
from functools import singledispatchmethod, singledispatch

from torch.nn import Sequential

from epde.structure.main_structures import Equation, SoEq
import epde.globals as global_var

from epde.solver.data import Domain, Conditions
from epde.solver.data import Equation as SolverEquation
from epde.solver.model import Model
from epde.solver.callbacks import cache, early_stopping, plot
from epde.solver.optimizers.optimizer import Optimizer
from epde.solver.device import solver_device, check_device, device_type
from epde.solver.models import Fourier_embedding

# from epde.solver.models import FourierNN
# from epde.solver.solver import grid_format_prepare
# from epde.solver.model import Model

# from epde.solver.input_preprocessing import Equation as SolverEquation
# import epde.solver.solver as solver
# from epde.solver.models import mat_model

VAL_TYPES = Union[FunctionType, int, float, torch.Tensor, np.ndarray]
# BASE_SOLVER_PARAMS = {'lambda_bound' : 100, 'verbose' : True, 
#                       'gamma' : 0.9, 'lr_decay' : 400, 'derivative_points' : 3, 
#                       'learning_rate' : 1e-3, 'eps' : 1e-6, 'tmin' : 5000,
#                       'tmax' : 2*1e4, 'use_cache' : False, 'cache_verbose' : True, 
#                       'patience' : 10, 'loss_oscillation_window' : 100,
#                       'no_improvement_patience' : 100, 'save_always' : False, 
#                       'print_every' : 1000, 'optimizer_mode' : 'Adam', 
#                       'model_randomize_parameter' : 1e-5, 'step_plot_print' : False, 
#                       'step_plot_save' : True, 'image_save_dir' : '/home/maslyaev/epde/EPDE_main/ann_imgs/', 'tol' : 0.01 }

'''
Specification of baseline equation solver parameters. Can be separated into its own json file for 
better handling, i.e. manual setup.
'''

BASE_COMPILING_PARAMS = {
                         'mode'                 : 'NN',
                         'lambda_operator'      : 1,
                         'lambda_bound'         : 1e2,
                         'normalized_loss_stop' : False,
                         'h'                    : 0.001,
                         'inner_order'          : '1',
                         'boundary_order'       : '2',
                         'weak_form'            : 'None',
                         'tol'                  : 0
                         }

BASE_OPTIMIZER_PARAMS = {
                         'optimizer'   : 'Adam', # Alternatively, switch to PSO, if it proves to be effective.
                         'params'      : {'lr'  : 1e-3,
                                          'eps' : 1e-6},
                         'gamma'       : 'None',
                         'decay_every' : 'None'
                         }

BASE_CACHE_PARAMS = {
                     'use_cache'                 : False,
                     'cache_verbose'             : True,
                     'cache_model'               : 'None',
                     'model_randomize_parameter' : 0,
                     'clear_cache'               : False
                     }

BASE_EARLY_STOPPING_PARAMS = {
                              'eps'                     : 1e-7,
                              'loss_window'             : 100,
                              'no_improvement_patience' : 1000,
                              'patience'                : 5,
                              'abs_loss'                : 1e-5,
                              'normalized_loss'         : False,
                              'randomize_parameter'     : 1e-5,
                              'info_string_every'       : 'None',
                              'verbose'                 : True
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




class BOPElement(object):
    def __init__(self, axis: int, key: str, coeff: float = 1., term: list = [None],
                 power: Union[List[int], int] = 1, var: Union[List[int], int] = 1, rel_location: float = 0.):
        self.axis = axis
        self.key = key
        self.coefficient = coeff
        self.term = term
        self.power = power
        self.variables = var
        self.location = rel_location
        self.grid = None
        
        self.status = {'boundary_location_set': False,
                       'boundary_values_set': False}

    def set_grid(self, grid: torch.Tensor):
        self.grid = grid
        self.status['boundary_location_set'] = True

    @property
    def operator_form(self):
        form = {
            'coeff': self.coefficient,
            self.key: self.term,
            'pow': self.power,
            'var': self.variables
        }
        return self.key, form

    @property
    def values(self):
        if isinstance(self._values, FunctionType):
            assert self.grid_set, 'Tring to evaluate variable coefficent without a proper grid.'
            res = self._values(self.grids)
            assert res.shape == self.grids[0].shape
            return torch.from_numpy(res)
        else:
            return self._values

    @values.setter
    def values(self, vals):
        if isinstance(vals, (FunctionType, int, float, torch.Tensor)):
            self._values = vals
            self.vals_set = True
        elif isinstance(vals, np.ndarray):
            self._values = torch.from_numpy(vals)
            self.vals_set = True
        else:
            raise TypeError(
                f'Incorrect type of coefficients. Must be a type from list {VAL_TYPES}.')

    def __call__(self, values: VAL_TYPES = None):
        if not self.vals_set and values is not None:
            self.values = values
            self.status['boundary_values_set'] = True
        elif not self.vals_set and values is None:
            raise ValueError('No location passed into the BOP.')
        if self.grid is not None:
            boundary = self.grid
        elif self.grid is None and self.location is not None:
            _, all_grids = global_var.grid_cache.get_all()

            abs_loc = self.location * all_grids[0].shape[self.axis]
            if all_grids[0].ndim > 1:
                boundary = np.array(all_grids[:self.axis] + all_grids[self.axis+1:])
                if isinstance(values, FunctionType):
                    raise NotImplementedError  # TODO: evaluation of BCs passed as functions or lambdas
                boundary = torch.from_numpy(np.expand_dims(boundary, axis=self.axis)).float()  # .reshape(bnd_shape))

                boundary = torch.cartesian_prod(boundary,
                                                torch.from_numpy(np.array([abs_loc,], dtype=np.float64))).float()
                boundary = torch.moveaxis(boundary, source=0, destination=self.axis).resize()
            else:
                boundary = torch.from_numpy(np.array([[abs_loc,],])).float() # TODO: work from here
            print('boundary.shape', boundary.shape, boundary.ndim)
            
        elif boundary is None and self.location is None:
            raise ValueError('No location passed into the BOP.')
            
        form = self.operator_form
        boundary_operator = {form[0]: form[1]}
        
        boundary_value = self.values
        
        return {'bnd_loc' : boundary, 'bnd_op' : boundary_operator, 'bnd_val' : boundary_value, 
                'variables' : self.variables, 'type' : 'operator'}

class PregenBOperator(object):
    def __init__(self, system: SoEq, system_of_equation_solver_form: list):
        self.system = system
        self.equation_sf = [eq for eq in system_of_equation_solver_form]
        self.variables = list(system.vars_to_describe)

    def demonstrate_required_ords(self):
        linked_ords = list(zip([eq.main_var_to_explain for eq in self.system],
                               self.max_deriv_orders))
        print(
            f'Orders, required by an equation, are as follows: {linked_ords}')

    @property
    def conditions(self):
        return self._bconds

    @conditions.setter
    def conditions(self, conds: List[BOPElement]):
        self._bconds = []
        if len(conds) != int(sum([value.sum() for value in self.max_deriv_orders.values()])):
            raise ValueError(
                'Number of passed boundry conditions does not match requirements of the system.')
        for condition in conds:
            if isinstance(condition, BOPElement):
                self._bconds.append(condition())
            else:
                print('condition is ', type(condition), condition)
                raise NotImplementedError(
                    'In-place initialization of boundary operator has not been implemented yet.')

    @property
    def max_deriv_orders(self):
        return self.get_max_deriv_orders(self.equation_sf, self.variables)

    @staticmethod
    def get_max_deriv_orders(system_sf: List[Dict[str, Dict]], variables: List[str] = ['u',]) -> np.ndarray:
        def count_factor_order(factor_code, deriv_ax):
            if factor_code is None:
                return 0
            else:
                if isinstance(factor_code, list):
                    return factor_code.count(deriv_ax)
                elif isinstance(factor_code, int):
                    return 1 if factor_code == deriv_ax else 0
                else:
                    raise TypeError('Incorrect type of the input.')

        @singledispatch
        def get_equation_requirements(equation_sf, variables=['u',]):
            raise NotImplementedError(
                'Single-dispatch called in generalized form')

        @get_equation_requirements.register
        def _(equation_sf: dict, variables=['u',]) -> dict:  # dict = {u: 0}):
            dim = global_var.grid_cache.get('0').ndim
            if len(variables) == 1:
                print('processing a single variable')
                var_max_orders = np.zeros(dim)
                for term in equation_sf.values():
                    if isinstance(term['pow'], list):
                        for deriv_factor in term['term']:
                            orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                               in np.arange(dim)])
                            var_max_orders = np.maximum(var_max_orders, orders)
                    else:
                        orders = np.array([count_factor_order(term['term'], ax) for ax
                                           in np.arange(dim)])
                var_max_orders = {variables[0]: np.maximum(var_max_orders, orders)}
                return var_max_orders
            else:
                var_max_orders = {var_key: np.zeros(dim) for var_key in variables}
                for term_key, symb_form in equation_sf.items():
                    if isinstance(symb_form['var'], list):
                        assert len(symb_form['term']) == len(symb_form['var'])
                        for factor_idx, deriv_factor in enumerate(symb_form['term']):
                            var_orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                                   in np.arange(dim)])
                            var_key = symb_form['var'][factor_idx] - 1
                            var_max_orders[variables[var_key]] = np.maximum(var_max_orders[variables[var_key]],
                                                                            var_orders)
                    elif isinstance(symb_form['var'], int):
                        raise NotImplementedError()
                        assert len(symb_form['term']) == 1
                        for factor_idx, factor in enumerate([count_factor_order(symb_form['term'], ax) for ax
                                                            in np.arange(dim)]):
                            var_orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                                   in np.arange(dim)])
                            var_key = symb_form['var'][factor_idx]
                            var_max_orders[var_key] = np.maximum(var_max_orders[var_key], var_orders)
                return var_max_orders

        @get_equation_requirements.register
        def _(equation_sf: list, variables=['u',]):
            raise NotImplementedError(
                'TODO: add equation list form processing')  # TODO

        eq_forms = []
        for equation_form in system_sf:
            eq_forms.append(get_equation_requirements(equation_form, variables))

        max_orders = {var: np.maximum.accumulate([eq_list[var] for eq_list in eq_forms])[-1]
                      for var in variables}  # TODO
        return max_orders

    def generate_default_bc(self, vals: Union[np.ndarray, dict] = None, grids: List[np.ndarray] = None,
                            allow_high_ords: bool = False):
        # Implement allow_high_ords - selection of derivatives from
        required_bc_ord = self.max_deriv_orders
        assert set(self.variables) == set(required_bc_ord.keys()), 'Some conditions miss required orders.'

        grid_cache = global_var.initial_data_cache
        tensor_cache = global_var.initial_data_cache

        if vals is None:
            val_keys = {key: (key, (1.0,)) for key in self.variables}

        if grids is None:
            _, grids = grid_cache.get_all()

        relative_bc_location = {0: (), 1: (0,), 2: (0, 1),
                                3: (0., 0.5, 1.), 4: (0., 1/3., 2/3., 1.)}

        bconds = []
        tensor_shape = grids[0].shape

        def get_boundary_ind(tensor_shape, axis, rel_loc):
            return tuple(np.meshgrid(*[np.arange(shape) if dim_idx != axis else min(int(rel_loc * shape), shape-1)
                                       for dim_idx, shape in enumerate(tensor_shape)], indexing='ij'))

        for var_idx, variable in enumerate(self.variables):
            for ax_idx, ax_ord in enumerate(required_bc_ord[variable]):
                for loc in relative_bc_location[ax_ord]:
                    indexes = get_boundary_ind(tensor_shape, ax_idx, rel_loc=loc)

                    coords = np.array([grids[idx][indexes] for idx in np.arange(len(tensor_shape))]).T
                    if coords.ndim > 2:
                        coords = coords.squeeze()

                    if vals is None:
                        bc_values = tensor_cache.get(val_keys[variable])[indexes]
                    else:
                        bc_values = vals[indexes]

                    bc_values = np.expand_dims(bc_values, axis=0).T
                    coords = torch.from_numpy(coords).float()

                    bc_values = torch.from_numpy(bc_values).float()
                    operator = BOPElement(axis=ax_idx, key=variable, coeff=1, term=[None],
                                          power=1, var=var_idx, rel_location=loc)
                    operator.set_grid(grid=coords)
                    operator.values = bc_values
                    bconds.append(operator)
        self.conditions = bconds
        print('cond[0]', [cond[0].shape for cond in self.conditions])
        print('cond[2]', [cond[2].shape for cond in self.conditions])


# class BoundaryConditions(object):
#     def __init__(self, grids=None, partial_operators: dict = []):
#         self.grids_set = (grids is not None)
#         if grids is not None:
#             self.grids = grids
#         self.operators = partial_operators

#     def form_operator(self):
#         return [list(bcond()) for bcond in self.operators.values()]


def solver_formed_grid(training_grid=None):
    if training_grid is None:
        keys, training_grid = global_var.grid_cache.get_all()
    else:
        keys, _ = global_var.grid_cache.get_all()

    assert len(keys) == training_grid[0].ndim, 'Mismatching dimensionalities'

    training_grid = np.array(training_grid).reshape((len(training_grid), -1))
    #return torch.from_numpy(training_grid).T.type(torch.FloatTensor)
    return torch.from_numpy(training_grid).T.float()


class SystemSolverInterface(object):
    def __init__(self, system_to_adapt: SoEq, coeff_tol: float = 1.e-9):
        self.variables = list(system_to_adapt.vars_to_describe)
        self.adaptee = system_to_adapt
        self.grids = None
        self.coeff_tol = coeff_tol

    @staticmethod
    def _term_solver_form(term, grids, default_domain, variables: List[str] = ['u',]):
        deriv_orders = []
        deriv_powers = []
        deriv_vars = []
        derivs_detected = False

        try:
            coeff_tensor = np.ones_like(grids[0])
            
        except KeyError:
            raise NotImplementedError('No cache implemented')
        for factor in term.structure:
            if factor.is_deriv:
                for param_idx, param_descr in factor.params_description.items():
                    if param_descr['name'] == 'power':
                        power_param_idx = param_idx
                deriv_orders.append(factor.deriv_code)
                deriv_powers.append(factor.params[power_param_idx])
                try:
                    cur_deriv_var = variables.index(factor.variable)
                except ValueError:
                    raise ValueError(
                        f'Variable family of passed derivative {variables}, other than {cur_deriv_var}')
                derivs_detected = True

                deriv_vars.append(cur_deriv_var)
            else:
                grid_arg = None if default_domain else grids
                coeff_tensor = coeff_tensor * factor.evaluate(grids=grid_arg)
        if not derivs_detected:
            deriv_powers = [0]
            deriv_orders = [[None,],]
        if len(deriv_powers) == 1:
            deriv_powers = deriv_powers[0]
            deriv_orders = deriv_orders[0]

        coeff_tensor = torch.from_numpy(coeff_tensor)

        if deriv_vars == []:
            if deriv_powers != 0:
                raise Exception('Something went wrong with parsing an equation for solver')
            else:
                deriv_vars = [0]
        res = {'coeff': coeff_tensor,
               'term': deriv_orders,
               'pow': deriv_powers,
               'var': deriv_vars}

        return res

    @singledispatchmethod
    def set_boundary_operator(self, operator_info):
        raise NotImplementedError()

    def _equation_solver_form(self, equation, variables, grids=None, mode = 'NN'):
        assert mode in ['NN', 'autograd', 'mat'], 'Incorrect mode passed. Form available only \
                                                   for "NN", "autograd "and "mat" methods'
        
        def adjust_shape(tensor, mode = 'NN'):
            if mode in ['NN', 'autograd']:
                return torch.flatten(tensor).unsqueeze(1).type(torch.FloatTensor)
            elif mode == 'mat':
                return tensor.type(torch.FloatTensor)
            
        _solver_form = {}
        if grids is None:
            grids = self.grids
            default_domain = True
        else:
            default_domain = False
        for term_idx, term in enumerate(equation.structure):
            if term_idx != equation.target_idx:
                if term_idx < equation.target_idx:
                    weight = equation.weights_final[term_idx]
                else:
                    weight = equation.weights_final[term_idx-1]
                if not np.isclose(weight, 0, rtol = self.coeff_tol):
                    _solver_form[term.name] = self._term_solver_form(term, grids, default_domain, variables)
                    _solver_form[term.name]['coeff'] = _solver_form[term.name]['coeff'] * weight
                    _solver_form[term.name]['coeff'] = adjust_shape(_solver_form[term.name]['coeff'], mode = mode)
                    #torch.flatten(_solver_form[term.name]['coeff']).unsqueeze(1).type(torch.FloatTensor)

        free_coeff_weight = torch.from_numpy(np.full_like(a=grids[0],
                                                          fill_value=equation.weights_final[-1]))
        free_coeff_weight = adjust_shape(free_coeff_weight, mode = mode)
        free_coeff_term = {'coeff': free_coeff_weight,
                           'term': [None],
                           'pow': 0,
                           'var': [0,]}
        _solver_form['C'] = free_coeff_term

        target_weight = torch.from_numpy(np.full_like(a=grids[0], fill_value=-1.))
        target_form = self._term_solver_form(equation.structure[equation.target_idx], grids, default_domain, variables)
        target_form['coeff'] = target_form['coeff'] * target_weight
        target_form['coeff'] = adjust_shape(target_form['coeff'], mode = mode)
        print(f'target_form shape is {target_form["coeff"].shape}')

        _solver_form[equation.structure[equation.target_idx].name] = target_form

        return _solver_form

    def use_grids(self, grids=None):
        if grids is None and self.grids is None:
            _, self.grids = global_var.grid_cache.get_all()
        elif grids is not None:
            if len(grids) != len(global_var.grid_cache.get_all()[1]):
                raise ValueError(
                    'Number of passed grids does not match the problem')
            self.grids = grids
            

    def form(self, grids=None, mode = 'NN'):
        self.use_grids(grids=grids)
        equation_forms = []

        for equation in self.adaptee.vals:
            equation_forms.append((equation.main_var_to_explain,
                                   self._equation_solver_form(equation, variables=self.variables,
                                                              grids=grids, mode = mode)))
        return equation_forms


class SolverAdapter(object):
    def __init__(self, net=None, fft_params: dict = None,
                 use_cache: bool = True, var_number: int = 1, use_fourier: bool = None):
        dim_number = global_var.grid_cache.get('0').ndim
        
        self.domain = Domain()
        print(f'dimensionality is {dim_number}')
        
        if net is None:
            if dim_number == 1:
                FFL = Fourier_embedding(**fft_params)
                linear_inputs = FFL.out_features                
                
                hidden_neurons = 128
            else:
                hidden_neurons = 112

        L_default, M_default = 4, 10
        if use_fourier:
            if fft_params is None:
                if dim_number == 1:
                   fft_params = {'L' : [L_default], 
                                 'M' : [M_default]}
                else:
                   fft_params = {'L' : [L_default] + [None,] * (dim_number - 1), 
                                 'M' : [M_default] + [None,] * (dim_number - 1)}
            net_default = [Fourier_embedding(**fft_params),]
        else:
            net_default = []            
               
        self.net = torch.nn.Sequential(net_default + [torch.nn.Linear(linear_inputs, hidden_neurons),
                                                      torch.nn.Tanh(),
                                                      torch.nn.Linear(hidden_neurons, hidden_neurons),
                                                      torch.nn.Tanh(),
                                                      torch.nn.Linear(hidden_neurons, hidden_neurons),
                                                      torch.nn.Tanh(),
                                                      torch.nn.Linear(hidden_neurons, var_number)
                                                      ])
        
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
        
        self.use_cache = use_cache
        self.prev_solution = None

    def set_solver_params(self, lambda_bound=None, verbose: bool = None, gamma: float = None,
                          lr_decay: int = 400, derivative_points: int = None, learning_rate: float = None,
                          eps: float = None, tmin: int = None, tmax: int = None,
                          use_cache: bool = None, cache_verbose: bool = None,
                          patience: int = None, loss_oscillation_window : int = None,
                          no_improvement_patience: int = None,
                          save_always: bool = None, print_every: bool = 5000, optimizer_mode = None, 
                          model_randomize_parameter: bool = None, step_plot_print: bool = None,
                          step_plot_save: bool = True, image_save_dir: str = None, tol: float = None):

        params = {'lambda_bound': lambda_bound, 'verbose': verbose, 'gamma': gamma, 
                  'lr_decay': lr_decay, 'derivative_points': derivative_points,
                  'learning_rate': learning_rate, 'eps': eps, 'tmin': tmin,
                  'tmax': tmax, 'use_cache': use_cache, 'cache_verbose': cache_verbose,
                  'patience' : patience, 'loss_oscillation_window' : loss_oscillation_window,
                  'no_improvement_patience' : no_improvement_patience, 'save_always': save_always,
                  'print_every': print_every, 'optimizer_mode': optimizer_mode,
                  'model_randomize_parameter': model_randomize_parameter, 'step_plot_print': step_plot_print,
                  'step_plot_save': step_plot_save, 'image_save_dir': image_save_dir, 'tol': tol}

        for param_key, param_vals in params.items():
            if param_vals is not None:
                try:
                    if param_vals is 'None':
                        self._solver_params[param_key] = None
                    else:
                        self._solver_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
    
    def set_compiling_params(self, mode: str = None, lambda_operator: float = None, 
                             lambda_bound : float = None, normalized_loss_stop: bool = None,
                             h: float = None, inner_order: str = None, bounary_order: str = None,
                             weak_form: List[Callable] = None, tol: float = None):
        compiling_params = {'mode' : mode, 'lambda_operator' : lambda_operator, 'lambda_bound' : lambda_bound,
                            'normalized_loss_stop' : normalized_loss_stop, 'h' : h, 'inner_order' : inner_order,
                            'bounary_order' : bounary_order, 'weak_form' : weak_form, 'tol' : tol}
        
        for param_key, param_vals in compiling_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._solver_params[param_key] = None
                    else:
                        self._solver_params[param_key] = param_vals
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
                        self._solver_params[param_key] = None
                    else:
                        self._solver_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
    
    def set_cache_params(self, use_cache: bool = None, cache_verbose: bool = None, 
                         cache_model: Sequential = None, model_randomize_parameter: Union[int, float] = None,
                         clear_cache: bool = None):
        cache_params = {'use_cache' : use_cache, 'cache_verbose' : cache_verbose, 'cache_model' : cache_model,
                        'model_randomize_parameter' : model_randomize_parameter, 'clear_cache' : clear_cache}

        for param_key, param_vals in cache_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._solver_params[param_key] = None
                    else:
                        self._solver_params[param_key] = param_vals
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
                        self._solver_params[param_key] = None
                    else:
                        self._solver_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
                    
    def set_plotting_params(self, save_every: int = None, print_every: int = None, title: str = None,
                            img_dir: str = None):
        plotting_params = {'save_every' : save_every, 'print_every' : print_every, 'print_every' : print_every,
                           'img_dir' : img_dir}
        for param_key, param_vals in plotting_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._solver_params[param_key] = None
                    else:
                        self._solver_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
    
    # def set_param(self, param_key: str, value):
        # self._solver_params[param_key] = value

    @staticmethod
    def create_domain(self, variables: List[str], grids : List[np.ndarray]) -> Domain:
        assert len(variables) == len(grids), f'Number of passed variables {len(variables)} does not \
                                               match number of grids {len(grids)}.'
        domain = Domain('uniform')
        for idx, var_name in enumerate(variables):
            domain.variable(var_name, torch.tensor(), n_points)
        

    def solve_epde_system(self, system: SoEq, grids: list=None, boundary_conditions=None, 
                          mode='NN', data=None):
        system_interface = SystemSolverInterface(system_to_adapt=system)

        system_solver_forms = system_interface.form(grids = grids, mode = mode)
        
        if boundary_conditions is None:
            op_gen = PregenBOperator(system=system,
                                     system_of_equation_solver_form=[sf_labeled[1] for sf_labeled
                                                                     in system_solver_forms])
            op_gen.generate_default_bc(vals = data, grids = grids)
            boundary_conditions = op_gen.conditions
            if not (isinstance(boundary_conditions, list) and isinstance(boundary_conditions[0], BOPElement)):
                raise ValueError('Incorrect boundary conditions generated in the solver interface.')
            
        bconds_combined = Conditions()
        for cond in boundary_conditions:
            bconds_combined.operator(bnd = cond['bnd_loc'], operator = cond['bnd_op'], 
                                     value = cond['bnd_val'])

        if grids is None:
            _, grids = global_var.grid_cache.get_all()

        return self.solve(equations=[form[1] for form in system_solver_forms], grid=grids,
                          boundary_conditions=boundary_conditions, mode = mode)

    def solve(self, equations, grid=None, boundary_conditions=None, mode = 'NN'): #: List[] =None
        print('Grid is ', type(grid), grid.shape)
        if isinstance(equations, SolverEquation):
            self.equations = equations
        else:
            self.equations = SolverEquation()
            for form in equations:
                self.equations.add(form)

        cb_cache = cache.Cache(**self._cache_params)
        cb_early_stops = early_stopping.EarlyStopping(**self._early_stopping_params)
        cb_plots = plot.Plots(**self._ploter_params)
        
        optimizer = Optimizer(**self._optimizer_params)
        
        model = Model(net = self.net, domain, equation, conditions)
        
        self.prev_solution = solver.Solver(grid, self.equations, self.model, mode).solve(**self._solver_params)
        return self.prev_solution