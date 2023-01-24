'''

'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

from typing import Callable, Union
from types import FunctionType

VAL_TYPES = Union[FunctionType, int, float, torch.Tensor, np.ndarray]
BASE_SOLVER_PARAMS = {'lambda_bound' : 10, 'verbose' : False,
                      'learning_rate' : 1e-3, 'eps' : 1e-5, 'tmin' : 1000,
                      'tmax' : 1e5, 'use_cache' : True, 'cache_verbose' : True, 
                      'save_always' : False, 'print_every' : None, 
                      'model_randomize_parameter' : 1e-6, 'step_plot_print' : False, 
                      'step_plot_save' : False, 'image_save_dir' : None}

from functools import singledispatchmethod, singledispatch

from epde.structure.main_structures import Equation, SoEq
import epde.globals as global_var

from epde.solver.input_preprocessing import Equation as SolverEquation
import epde.solver.solver as solver

class PregenBOperator(object):
    def __init__(self, system : SoEq, system_of_equation_solver_form : list):
        self.system = system
        self.equation_sf = [eq for eq in system_of_equation_solver_form]
        self.variables = list(system.vars_to_describe)
        # print('Varibales:', self.variables)
        # self.max_ord = self.max_deriv_orders
        
    def demonstrate_required_ords(self):
        linked_ords = list(zip([eq.main_var_to_explain for eq in self.system], 
                               self.max_deriv_orders))
        print(f'Orders, required by an equation, are as follows: {linked_ords}')
        
    @property
    def conditions(self):
        return self._bconds
    
    @conditions.setter
    def conditions(self, conds : list):
        self._bconds = []
        if len(conds) != int(sum([value.sum() for value in self.max_deriv_orders.values()])):
            raise ValueError('Number of passed boundry conditions does not match requirements of the system.')
        for condition in conds:
            if isinstance(condition, BOPElement):
                self._bconds.append(condition())
            else:
                print('condition is ', type(condition), condition)
                raise NotImplementedError('In-place initialization of boundary operator has not been implemented yet.')
    
    @property
    def max_deriv_orders(self):
        return self.get_max_deriv_orders(self.equation_sf, self.variables)
    
    @staticmethod
    def get_max_deriv_orders(system_sf : list, variables : list = ['u',]) -> np.ndarray:
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
        def get_equation_requirements(equation_sf, variables = ['u',]):
            raise NotImplementedError('Single-dispatch called in generalized form')
        
        @get_equation_requirements.register
        def _(equation_sf : dict, variables = ['u',]) -> dict:# dict = {u : 0}):
            dim = global_var.grid_cache.get('0').ndim
            if len(variables) == 1:
                print('processing a single variable')                
                var_max_orders = np.zeros(dim)
                for term in equation_sf.values():
                    if isinstance(term['power'], list):
                        for deriv_factor in term['term']:
                            orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                               in np.arange(dim)])
                            var_max_orders = np.maximum(var_max_orders, orders)
                    else:
                        orders = np.array([count_factor_order(term['term'], ax) for ax
                                           in np.arange(dim)])
                var_max_orders = {variables[0] : np.maximum(var_max_orders, orders)}
                return var_max_orders
            else:
                var_max_orders = {var_key : np.zeros(dim) for var_key in variables}
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
        def _(equation_sf : list, variables = ['u',]):
            raise NotImplementedError('TODO: add equation list form processing') # TODO
            
        eq_forms = []
        for equation_form in system_sf:
            eq_forms.append(get_equation_requirements(equation_form, variables))
        
        max_orders = {var : np.maximum.accumulate([eq_list[var] for eq_list in eq_forms])[-1]
                      for var in variables}    # TODO
        return max_orders
    
    def generate_default_bc(self, vals : Union[np.ndarray, dict] = None, grids : list = None, 
                            allow_high_ords : bool = False):
        # Implement allow_high_ords - selection of derivatives from 
        required_bc_ord = self.max_deriv_orders
        assert set(self.variables) == set(required_bc_ord.keys()), 'Some conditions miss required orders.'
        assert (len(self.variables) == 1) == isinstance(vals, dict), ''

        grid_cache = global_var.initial_data_cache
        tensor_cache = global_var.initial_data_cache
        
        if vals is None:
            val_keys = {key : (key, (1.0,)) for key in self.variables}

        if grids is None:
            _, grids = grid_cache.get_all()
            
        relative_bc_location = {0 : (), 1 : (0,), 2 : (0, 1), 
                                3 : (0., 0.5, 1.), 4 : (0., 1/3., 2/3., 1.)}
        
        bconds = []
        tensor_shape = grids[0].shape

        def get_boundary_ind(tensor_shape, axis, rel_loc):
            return tuple(np.meshgrid(*[np.arange(shape) if dim_idx != axis else min(int(rel_loc * shape), shape-1)
                       for dim_idx, shape in enumerate(tensor_shape)], indexing = 'ij'))        
        
        for var_idx, variable in enumerate(self.variables):            
            for ax_idx, ax_ord in enumerate(required_bc_ord[variable]):
                for loc in relative_bc_location[ax_ord]:
                    indexes = get_boundary_ind(tensor_shape, ax_idx, rel_loc = loc)
                    coords = np.array([grids[idx][indexes] for idx in np.arange(len(tensor_shape))]).T
                    if coords.ndim > 2:
                        coords.squeeze()
                        
                    if vals is None:
                        bc_values = tensor_cache.get(val_keys[variable])[indexes]
                    else:
                        bc_values = vals[indexes]
                        
                    bc_values = np.expand_dims(bc_values, axis = 0).T
                    coords = torch.from_numpy(coords).type(torch.FloatTensor)
                    
                    bc_values = torch.from_numpy(bc_values).type(torch.FloatTensor)
                    operator = BOPElement(axis = ax_idx, key = variable, coeff = 1, term = [None], 
                                          power = 1, var = var_idx, rel_location=loc)
                    operator.set_grid(grid = coords)
                    operator.values = bc_values
                    bconds.append(operator) # )(rel_location=loc)
        self.conditions = bconds


class BOPElement(object):
    def __init__(self, axis : int, key : str, coeff : float = 1., term : list = [None], 
                 power : Union[list, int] = 1, var : Union[list, int] = 1, rel_location : float = 0.):
        self.axis = axis
        self.key = key
        self.coefficient = coeff
        self.term = term
        self.power = power
        self.variables = var
        self.location = rel_location
        
        self.status = {'boundary_location_set'  : False,
                       'boundary_values_set'    : False}
    
    def set_grid(self, grid : torch.Tensor):
        self.grid = grid
        self.status['boundary_location_set'] = True
    
    @property
    def operator_form(self):
        form = {
                'coeffs' : self.coefficient,
                self.key   : self.term,
                'pow'  : self.power,
                'var'    : self.variables
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
            raise TypeError(f'Incorrect type of coefficients. Must be a type from list {VAL_TYPES}.')
        
        
    def __call__(self, values : VAL_TYPES = None, boundary : list = None):
        if not self.vals_set and values is not None:
            self.values = values
            self.status['boundary_values_set'] = True
        elif not self.vals_set and values is None:
            raise ValueError('No location passed into the BOP.')
        if boundary is None and self.location is not None:
            _, all_grids = global_var.grid_cache.get_all() # str(self.axis)

            abs_loc = self.location * all_grids[0].shape[self.axis]            
            if all_grids[0].ndim > 1:
                boundary = np.array(all_grids[:self.axis] + all_grids[self.axis+1:])
                if isinstance(values, FunctionType):
                    raise NotImplementedError # TODO: evaluation of BCs passed as functions or lambdas 
                boundary = torch.from_numpy(np.expand_dims(boundary, axis = self.axis)).float()  #.reshape(bnd_shape))

                boundary = torch.cartesian_prod(boundary, 
                                                torch.from_numpy(np.array([abs_loc,], dtype=np.float64))).float()
                boundary = torch.moveaxis(boundary, source = 0, destination = self.axis).resize()
            else:
                boundary = torch.from_numpy(np.array([[abs_loc,],])) # TODO: work from here
                # boundary = torch.expand_dims(boundary, axis = 0)
            print('boundary.shape', boundary.shape, boundary.ndim)
            
        elif boundary is None and self.location is None:
            raise ValueError('No location passed into the BOP.')
            
    
        # boundary = torch.reshape(boundary, ())
        
        form = self.operator_form
        boundary_operator = {form[0] : form[1]}

        boundary_value = self.values # TODO: inspect boundary value setter with an arbitrary function in symb form
        
        print('Output of bc:')
        print(boundary, boundary_value)
        return boundary, boundary_operator, boundary_value

class BoundaryConditions(object):
    def __init__(self, grids = None, partial_operators : dict = []):
        self.grids_set = (grids is not None)
        if grids is not None:
            self.grids = grids
        self.operators = partial_operators

    def form_operator(self):
        return [list(bcond()) for bcond in self.operators.values()]


def solver_formed_grid(training_grid = None):
    if training_grid is None: 
        keys, training_grid = global_var.grid_cache.get_all()
    else:
        keys, _ = global_var.grid_cache.get_all()
        
    assert len(keys) == training_grid[0].ndim, 'Mismatching dimensionalities'
    
    training_grid = np.array(training_grid).reshape((len(training_grid), -1))
    return torch.from_numpy(training_grid).T.type(torch.FloatTensor)

class SystemSolverInterface(object):
    def __init__(self, system_to_adapt : SoEq):
        self.variables = list(system_to_adapt.vars_to_describe)
        self.adaptee = system_to_adapt
        # assert self.adaptee.weights_final_evald

        self.grids = None
    
    # @staticmethod
    # def old_term_solver_form(term, grids):
    #     deriv_orders = []
    #     deriv_powers = []
    #     derivs_detected = False
        
    #     try:
    #         coeff_tensor = np.ones_like(global_var.grid_cache.get('0'))
    #     except KeyError:
    #         raise NotImplementedError('No cache implemented')
    #     for factor in term.structure:
    #         if factor.is_deriv:
    #             for param_idx, param_descr in factor.params_description.items():
    #                 if param_descr['name'] == 'power': power_param_idx = param_idx
    #             deriv_orders.append(factor.deriv_code); deriv_powers.append(factor.params[power_param_idx])
    #             derivs_detected = True
    #         else:
    #             coeff_tensor = coeff_tensor * factor.evaluate(grids = grids)
    #     if not derivs_detected:
    #        deriv_powers = [0]; deriv_orders = [[None,],]
    #     if len(deriv_powers) == 1:
    #         deriv_powers = deriv_powers[0]
    #         deriv_orders = deriv_orders[0]
            
    #     coeff_tensor = torch.from_numpy(coeff_tensor)
    #     return [coeff_tensor, deriv_orders, deriv_powers]

    @staticmethod
    def _term_solver_form(term, grids, variables : list = ['u',]):
        deriv_orders = []
        deriv_powers = []
        deriv_vars = []
        derivs_detected = False
        
        try:
            coeff_tensor = np.ones_like(grids[0])#global_var.grid_cache.get('0'))
        except KeyError:
            raise NotImplementedError('No cache implemented')
        for factor in term.structure:
            if factor.is_deriv:
                for param_idx, param_descr in factor.params_description.items():
                    if param_descr['name'] == 'power': power_param_idx = param_idx
                deriv_orders.append(factor.deriv_code); deriv_powers.append(factor.params[power_param_idx])
                try:
                    cur_deriv_var = variables.index(factor.ftype)
                except ValueError:
                    raise ValueError(f'Variable family of passed derivative {variables}, other than {cur_deriv_var}')
                derivs_detected = True
                
                deriv_vars.append(cur_deriv_var)
            else:
                coeff_tensor = coeff_tensor * factor.evaluate(grids = grids)
        if not derivs_detected:
           deriv_powers = [0]; deriv_orders = [[None,],], 
        if len(deriv_powers) == 1:
            deriv_powers = deriv_powers[0]
            deriv_orders = deriv_orders[0]
            
        coeff_tensor = torch.from_numpy(coeff_tensor)
        
        res = {'const'  : coeff_tensor,
               'term'   : deriv_orders,
               'power'  : deriv_powers,
               'var'    : deriv_vars}
        
        return res

    @singledispatchmethod
    def set_boundary_operator(self, operator_info):
        raise NotImplementedError()
        
    def _equation_solver_form(self, equation, variables, grids = None):
        # TODO: fix performance on other, than default grids.
        if grids is None:
            grids = self.grids
        _solver_form = {}
        for term_idx, term in enumerate(equation.structure):
            if term_idx != equation.target_idx:
                _solver_form[term.name] = self._term_solver_form(term, grids, variables)
                if term_idx < equation.target_idx:
                    weight = equation.weights_final[term_idx]
                else:
                    weight = equation.weights_final[term_idx-1]
                _solver_form[term.name]['const'] = _solver_form[term.name]['const'] * weight
                _solver_form[term.name]['const'] = torch.flatten(_solver_form[term.name]['const']).unsqueeze(1).type(torch.FloatTensor)

        free_coeff_weight = torch.from_numpy(np.full_like(a = grids[0], # global_var.grid_cache.get('0'), 
                                                          fill_value = equation.weights_final[-1]))
        free_coeff_weight = torch.flatten(free_coeff_weight).unsqueeze(1).type(torch.FloatTensor)
        free_coeff_term = {'const'  : free_coeff_weight,
                           'term'   : [None],
                           'power'  : 0,
                           'var'    : [0,]}
        _solver_form['C'] = free_coeff_term

        target_weight = torch.from_numpy(np.full_like(a = grids[0], #global_var.grid_cache.get('0'), 
                                                      fill_value = -1.))               
        target_form = self._term_solver_form(equation.structure[equation.target_idx], grids, variables)
        target_form['const'] = target_form['const'] * target_weight
        target_form['const'] = torch.flatten(target_form['const']).unsqueeze(1).type(torch.FloatTensor)        
        
        _solver_form[equation.structure[equation.target_idx].name] = target_form
        return _solver_form

    def use_grids(self, grids = None):
        if grids is None and self.grids is None:
            _, self.grids = global_var.grid_cache.get_all()
        elif grids is None :
            if len(grids) != len(global_var.grid_cache.get_all()[1]):
                raise ValueError('Number of passed grids does not match the problem')
            self.grids = grids

    def form(self, grids = None):
        self.use_grids(grids = grids)
        equation_forms = []

        for equation in self.adaptee.vals:  # Deleted enumeration
            equation_forms.append((equation.main_var_to_explain, 
                                   self._equation_solver_form(equation, variables = self.variables,
                                                              grids = grids)))
        return equation_forms


class SolverAdapter(object):
    def __init__(self, model = None, use_cache : bool = True, var_number : int = 1, 
                 dim_number : int = 1):
        if model is None:
            model = torch.nn.Sequential(
                                        torch.nn.Linear(dim_number, 100),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(100, 100),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(100, 100),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(100, 100),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(100, var_number),
                                        )
        self.model = model

        self._solver_params = dict()
        _solver_params = BASE_SOLVER_PARAMS

        self.set_solver_params(**_solver_params)
        self.use_cache = use_cache
        self.prev_solution = None

    def set_solver_params(self, lambda_bound = None, verbose : bool = None, learning_rate : float = None, 
                          eps : float = None, tmin : int = None, tmax : int = None, 
                          use_cache : bool = None, cache_verbose : bool = None, 
                          save_always : bool = None, print_every : bool = None, 
                          model_randomize_parameter : bool = None, step_plot_print : bool = None, 
                          step_plot_save : bool = None, image_save_dir : str = None):
        #TODO: refactor
        params = {'lambda_bound' : lambda_bound, 'verbose' : verbose,
                  'learning_rate' : learning_rate, 'eps' : eps, 'tmin' : tmin,
                  'tmax' : tmax, 'use_cache' : use_cache, 'cache_verbose' : cache_verbose, 
                  'save_always' : save_always, 'print_every' : print_every, 
                  'model_randomize_parameter' : model_randomize_parameter, 'step_plot_print' : step_plot_print, 
                  'step_plot_save' : step_plot_save, 'image_save_dir' : image_save_dir}
        
        print('params: ', params)
        for param_key, param_vals in params.items():
            if params is not None:
                try:
                    self._solver_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')

    def set_param(self, param_key : str, value):
        self._solver_params[param_key] = value

    def solve_epde_system(self, system : SoEq, grids : list = None, boundary_conditions = None):
        system_interface = SystemSolverInterface(system_to_adapt = system)

        system_solver_forms = system_interface.form(grids)
        if boundary_conditions is None:
            op_gen = PregenBOperator(system = system, 
                                     system_of_equation_solver_form = [sf_labeled[1] for sf_labeled 
                                                                       in system_solver_forms])
            op_gen.generate_default_bc()
            boundary_conditions = op_gen.conditions

        if grids is None:
            _, grids = global_var.grid_cache.get_all()

        return self.solve(system_form = [form[1] for form in system_solver_forms], grid = grids,
                          boundary_conditions = boundary_conditions)


    @staticmethod
    def convert_grid(grid):
        assert isinstance(grid, (list, tuple)), 'Convertion of the tensor can be only held from tuple or list.'
        dimensionality = len(grid)
        
        if grid[0].ndim == dimensionality:
            conv_grid = np.stack([subgrid.reshape(-1) for subgrid in grid])
            conv_grid = torch.from_numpy(conv_grid).float().T
        else:
            conv_grid = [torch.from_numpy(subgrid) for subgrid in grid]
            conv_grid = torch.cartesian_prod(*conv_grid).float()
        return conv_grid


    def solve(self, system_form = None, grid = None, boundary_conditions = None):
        if isinstance(grid, (list, tuple)):
            grid = self.convert_grid(grid)
        self.equation = SolverEquation(grid, system_form, boundary_conditions).set_strategy('NN')
        
        self.prev_solution = solver.Solver(grid, self.equation, self.model, 'NN').solve(**self._solver_params)
        return self.prev_solution
