'''

'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

from typing import Callable, Union
from types import FunctionType

VAL_TYPES = Union[FunctionType, int, float, torch.Tensor, np.ndarray]

from functools import singledispatchmethod, singledispatch

from epde.structure.main_structures import Equation, SoEq
import epde.globals as global_var

from TEDEouS.input_preprocessing import Equation as SolverEquation
import TEDEouS.solver as solver

class PregenBOperator(object):
    def __init__(self, system : SoEq, system_of_equation_solver_form : list):
        self.system = system
        self.equation_sf = [eq for eq in system_of_equation_solver_form]
        self.max_ord = self.max_deriv_orders()
        self.variables = system.vars_to_describe
        
    def demonstrate_required_ords(self):
        linked_ords = list(zip([eq.main_var_to_explain for eq in self.system], 
                               self.max_deriv_orders(self.equation_sf)))
        print(f'Orders, required by an equation, are as follows: {linked_ords}')
        
    @property
    def conditions(self):
        return self._bconds
    
    @conditions.setter
    def conditions(self, conds : list):
        def count_var_bc(conds, var : int):
            return sum([cond[1]['var'] == var or var in cond[1]['var'] for cond in conds])
        
        self._bconds = []
        if len(conds) != sum(self.max_deriv_orders(self.equation_sf)):
            raise ValueError('Number of passed boundry conditions does not match requirements of the system.')
        for condition in conds:
            if isinstance(condition, BOPElement):
                self._bconds.append(condition())
            else:
                # self.bconds = BOPElement(key, coeff)
                raise NotImplementedError('In-place initialization of boundary operator has not been implemented yet.')
        
        if self.max_deriv_orders(self.equation_sf) != [count_var_bc(self._bconds, v) for v in np.arange(v)]: # TODO: correct check
            raise ValueError('Numbers of conditions do not match requirements of equations.')
            
    
    
    def parse_equation(self, conditions : list, full_domain : bool = True, grids = None):
        required_orders = self.max_deriv_orders(self.equation_sf)
        assert len(conditions)
    
    @staticmethod
    def max_deriv_orders(system_sf : list, variables : list = ['u',]) -> np.ndarray:
        def count_factor_order(obj, deriv_ax):
            if obj is None:
                return 0
            else:
                return obj.count(deriv_ax)
        
        @singledispatch
        def get_equation_requirements(equation_sf, variables = ['u',]):
            raise NotImplementedError('Single-dispatch called in generalized form')
        
        @get_equation_requirements.register
        def _(equation_sf : dict, variables = ['u',]):# dict = {u : 0}):
            dim = global_var.grid_cache.get('0').ndim
            if len(variables) > 1:
                var_max_orders = np.zeros(dim)
                for term in equation_sf.values():
                    if isinstance(term[2], list):
                        for deriv_factor in term[1]:
                            orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                               in np.arange(dim)])
                            var_max_orders = np.maximum(var_max_orders, orders)
                    else:
                        orders = np.array([count_factor_order(term[1], ax) for ax
                                           in np.arange(dim)])
                var_max_orders = {variables[0] : np.maximum(var_max_orders, orders)}
            else:
                var_max_orders = {var_key : np.zeros(dim) for var_key in variables}
                if list(var_max_orders.keys()) != list(equation_sf.keys()):
                    raise ValueError('Variables are not ordered correctily or differ from the solver form.')
                for term_key, symb_form in equation_sf.items():
                    if isinstance(symb_form['var'], list):
                        assert len(symb_form['term']) == len(symb_form['var'])
                        for factor_idx, factor in enumerate([count_factor_order(symb_form['term'], ax) for ax
                                                            in np.arange(dim)]):
                            var_orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                                   in np.arange(dim)])
                            var_key = symb_form['var'][factor_idx]
                            var_max_orders[var_key] = np.maximum(var_max_orders[var_key], var_orders)
                    elif isinstance(symb_form['var'], int):
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
        
        max_orders = {var : np.accumulate([eq_list[var] for eq_list in eq_forms])
                      for var in variables}    # TODO
        return max_orders
    
    def generate_default_bc(self, grids : list = None, allow_high_ords : bool = True):
        required_bc_ord = self.max_deriv_orders()
        
        if grids is None:
            grids = global_var.grid_cache.get_all()

        relative_bc_location

        # TODO: finish
            
        
        
        

class BOPElement(object):
    def __init__(self, key : str, coeff : float = 1., term : list = [None], 
                 power : Union[list, int] = 1, var : Union[list, int] = 1):
        self.key = key
        self.coefficient = coeff
        self.term = term
        self.power = power
        self.variables = var
        
        self.status = {'boundary_location_set'  : False,
                       'boundary_operator_set'  : False,
                       'boundary_values_set'    : False}
    
    @property
    def operator_form(self):
        form = {
                'coeffs' : self.coefficient,
                self.key   : self.term,
                'pow'  : self.powers,
                'var'    : self.variables
                }
        return self.key, form
        
    # @property
    # def coefficient(self):
    #     if isinstance(self._coefficient, FunctionType):
    #         assert self.grid_set, 'Tring to evaluate variable coefficent without a proper grid.'
    #         res = self._coefficient(self.grids)
    #         assert res.shape == self.grids[0].shape
    #         return torch.from_numpy(res)
    #     else:
    #         return self._coefficient
    
    # @coefficient.setter
    # def coefficient(self, coeff):
    #     if isinstance(coeff, (FunctionType, int, float, torch.Tensor)):
    #         self._coefficient = coeff
    #     elif isinstance(coeff, np.ndarray):
    #         if self.grids is not None and np.shape(coeff) != np.shape(self.grids[0]):
    #             raise ValueError('Numpy array of coefficients does not match the grid.')
    #         self._coefficient = torch.from_numpy(coeff)
    #     else:
    #         raise TypeError(f'Incorrect type of coefficients. Must be a type from list {VAL_TYPES}.')
    
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
        
        
    def __call__(self, values : VAL_TYPES = None, boundary : list = None, 
                 rel_location : float = None):
        if not self.vals_set and values is not None:
            self.values = values
        elif values is None:
            raise ValueError('No location passed into the BOP.')
        if boundary is None and rel_location is not None:
            # try:
            _, all_grids = global_var.grid_cache.get_all() # str(self.axis)
            
            with np.moveaxis(all_grids[0], source = self.axis, destination = 0)[0, ...] as tmp:
                bnd_shape = (tmp.size, np.squeeze(tmp))
            boundary = np.array(all_grids[:self.axis] + all_grids[self.axis+1:])
            if isinstance(values, FunctionType):
                raise NotImplementedError # TODO: evaluation of BCs passed as functions or lambdas 
            boundary = torch.from_numpy(boundary.reshape(bnd_shape))
            
            # boundary = np.squeeze(general_grid[rel_location * general_grid.shape[0], ...])
            boundary = torch.cartesian_prod(boundary, torch.from_numpy(np.array([0], dtype=np.float64))).float()
            boundary = torch.moveaxis(boundary, source = 0, destination = self.axis).resize()
            
            
        elif boundary is None and rel_location is None:
            raise ValueError('No location passed into the BOP.')
            
    
        # boundary = torch.reshape(boundary, ())
        
        form = self.operator_form
        boundary_operator = {form[0] : form[1]}

        boundary_value = self.values # TODO: inspect boundary value setter with an arbitrary function in symb form

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
        self.variables = system_to_adapt.vars_to_describe
        self.adaptee = system_to_adapt
        assert self.adaptee.weights_final_evald

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
            coeff_tensor = np.ones_like(global_var.grid_cache.get('0'))
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
                
                deriv_vars.append(cur_deriv_var + 1)
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

        free_coeff_weight = torch.from_numpy(np.full_like(a = global_var.grid_cache.get('0'), 
                                                          fill_value = equation.weights_final[-1]))
        free_coeff_weight = torch.flatten(free_coeff_weight).unsqueeze(1).type(torch.FloatTensor)
        free_coeff_term = {'const'  : free_coeff_weight,
                           'term'   : [None],
                           'power'  : 0,
                           'var'    : 1}
        _solver_form['C'] = free_coeff_term

        target_weight = torch.from_numpy(np.full_like(a = global_var.grid_cache.get('0'), 
                                                          fill_value = -1.))               
        target_form = self._term_solver_form(equation.structure[equation.target_idx], grids, variables)
        target_form['const'] = target_form['const'] * target_weight
        target_form['const'] = torch.flatten(target_form['const']).unsqueeze(1).type(torch.FloatTensor)        
        
        _solver_form[equation.structure[equation.target_idx].name] = target_form
        return _solver_form

    def use_grids(self, grids = None):
        if grids is None and self.grids is None:
            self.grids = global_var.grid_cache.get_all()
        elif grids is None :
            if len(grids) != len(global_var.grid_cache.get_all()):
                raise ValueError('Number of passed grids does not match the problem')
            self.grids = grids

    def form(self, grids = None):
        self.use_grids(grids = grids)
        equation_forms = []

        for equation in self.adaptee.vals:  # Deleted enumeration
            equation_forms.append((equation.main_var_to_explain, self._equation_solver_form(equation, grids = grids)))
        return equation_forms


class SolverAdapter(object):
    def __init__(self, model = None, use_cache : bool = True):
        if model is None:
            model = torch.nn.Sequential(
               torch.nn.Linear(1, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 1),
            )
        self.default_model = model


        self._solver_params = {'model' : self.default_model, 'learning_rate' : 1e-3, 'eps' : 1e-5, 'tmin' : 1000,
                               'tmax' : 1e5, 'use_cache' : True, 'cache_verbose' : True, 
                               'save_always' : False, 'print_every' : False, 
                               'model_randomize_parameter' : 1e-6, 'step_plot_print' : False, 
                               'step_plot_save' : False, 'image_save_dir' : None}

        self.set_solver_params()
        self.use_cache = use_cache
        self.prev_solution = None

    # def use_default(self, key, vals, base_vals):
    #     self._params = key


    # def set_solver_params(self, params = {'model' : None, 'learning_rate' : None, 'eps' : None, 'tmin' : None, 
    #                                       'tmax' : None, 'use_cache' : None, 'cache_verbose' : None, 
    #                                       'save_always' : None, 'print_every' : None, 
    #                                       'model_randomize_parameter' : None, 'step_plot_print' : None, 
    #                                       'step_plot_save' : None, 'image_save_dir' : None}):

    def set_solver_params(self, model = None, learning_rate : float = None, eps : float = None, 
                          tmin : int = None, tmax : int = None, use_cache : bool = None, cache_verbose : bool = None, 
                          save_always : bool = None, print_every : bool = None, 
                          model_randomize_parameter : bool = None, step_plot_print : bool = None, 
                          step_plot_save : bool = None, image_save_dir : str = None):
        params = {'model' : model, 'learning_rate' : learning_rate, 'eps' : eps, 'tmin' : tmin,
                  'tmax' : tmax, 'use_cache' : use_cache, 'cache_verbose' : cache_verbose, 
                  'save_always' : save_always, 'print_every' : print_every, 
                  'model_randomize_parameter' : model_randomize_parameter, 'step_plot_print' : step_plot_print, 
                  'step_plot_save' : step_plot_save, 'image_save_dir' : image_save_dir}
        
        if model is None:
            model = self.default_model 
        for param_key, param_vals in params.items():
            if params is not None:
                try:
                    self._solver_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')

    def set_param(self, param_key, value):
        self._solver_params[param_key] = value

    def solve_epde_system(self, system : SoEq, grids : list = None, boundary_conditions = None):
        system_interface = SystemSolverInterface(system_to_adapt = system)

        system_solver_forms = system_interface.form(grids)
        if boundary_conditions is None:
            bop_gen = PregenBOperator(system = system, s_of_equation_solver_form = [sf_labeled[1] for sf_labeled in system_solver_forms])
        
        if grids is None:
            grids = global_var.grid_cache.get_all()

        return self.solve(system_form = [form[1] for form in system_solver_forms], grid = grids)


    def solve(self, system_form = None, grid = None, boundary_conditions = None):
        if system_form is None and grid is None and boundary_conditions is None:
            self.equation = SolverEquation(grid, system_form, boundary_conditions).set_strategy('NN')
        self.prev_solution = solver.Solver(grid, self.equation, self.model, 'NN').solver(**self._solver_params)
        return self.prev_solution