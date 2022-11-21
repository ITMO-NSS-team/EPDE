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


class PregenOperator(object):
    def __init__(self, system : SoEq, s_of_equation_solver_form : list):
        self.system = system
        self.equation_sf = [eq for eq in s_of_equation_solver_form]
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
        
        if self.max_deriv_orders(self.equation_sf) != [count_var_bc(self._bconds, v) for v in np.arange(v)]:
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
    
    def form_operator(self):
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
        if boundary is None and rel_location is not None:
            # try:
            _, all_grids = global_var.grid_cache.get_all() # str(self.axis)
            
            with np.moveaxis(all_grids[0], source = self.axis, destination = 0)[0, ...] as tmp:
                bnd_shape = (tmp.size, np.squeeze(tmp))
            boundary = torch.from_numpy(np.array(all_grids[:self.axis] + all_grids[self.axis+1:]).reshape(bnd_shape))
            
            # boundary = np.squeeze(general_grid[rel_location * general_grid.shape[0], ...])
            boundary = torch.cartesian_prod(boundary, torch.from_numpy(np.array([0], dtype=np.float64))).float()
            boundary = torch.moveaxis(boundary, source = 0, destination = self.axis).resize()
            
            
        elif boundary is None and rel_location is None:
            raise ValueError('No location passed into the BOP.')
            
        # boundary = torch.reshape(boundary, ())
        
        return boundary, boundary_operator, boundary_value

class BoundaryCondition(object):
    def __init__(self, grids = None, partial_operators : dict = []):
        self.grids_set = (grids is not None)
        if grids is not None:
            self.grids = grids
        self.operators = partial_operators

    def form_operator(self):
        return 


def solver_formed_grid(training_grid = None):
    if training_grid is None: 
        keys, training_grid = global_var.grid_cache.get_all()
    else:
        keys, _ = global_var.grid_cache.get_all()
        
    assert len(keys) == training_grid[0].ndim, 'Mismatching dimensionalities'
    
    training_grid = np.array(training_grid).reshape((len(training_grid), -1))
    return torch.from_numpy(training_grid).T.type(torch.FloatTensor)

class SolverAdapter(object):
    def __init__(self, system_to_adapt : SoEq):
        self.variables = system_to_adapt.vars_to_describe
        self.adaptee = system_to_adapt
        assert self.adaptee.weights_final_evald
    
    @staticmethod
    def old_term_solver_form(term, grids):
        deriv_orders = []
        deriv_powers = []
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
                derivs_detected = True
            else:
                coeff_tensor = coeff_tensor * factor.evaluate(grids = grids)
        if not derivs_detected:
           deriv_powers = [0]; deriv_orders = [[None,],]
        if len(deriv_powers) == 1:
            deriv_powers = deriv_powers[0]
            deriv_orders = deriv_orders[0]
            
        coeff_tensor = torch.from_numpy(coeff_tensor)
        return [coeff_tensor, deriv_orders, deriv_powers]

    @staticmethod
    def term_solver_form(term, grids, variables : list = ['u',]):
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
        
    # def old_equation_solver_form(self, equation, grids):
    #     _solver_form = []
    #     for term_idx in range(len(equation.structure)):
    #         if term_idx != equation.target_idx:
    #             term_form = self.term_solver_form(equation.structure[term_idx], grids)  #equation.structure[term_idx].solver_form
    #             if term_idx < equation.target_idx:
    #                 weight = equation.weights_final[term_idx] 
    #             else:
    #                 weight = equation.weights_final[term_idx-1]
    #             term_form[0] = term_form[0] * weight
    #             term_form[0] = torch.flatten(term_form[0]).unsqueeze(1).type(torch.FloatTensor)
    #             _solver_form.append(term_form)
                
    #     free_coeff_weight = torch.from_numpy(np.full_like(a = global_var.grid_cache.get('0'), 
    #                                                       fill_value = equation.weights_final[-1]))
    #     free_coeff_weight = torch.flatten(free_coeff_weight).unsqueeze(1).type(torch.FloatTensor)
    #     target_weight = torch.from_numpy(np.full_like(a = global_var.grid_cache.get('0'), 
    #                                                       fill_value = -1.))            
    #     target_form = equation.structure[equation.target_idx].solver_form
    #     target_form[0] = target_form[0] * target_weight
    #     target_form[0] = torch.flatten(target_form[0]).unsqueeze(1).type(torch.FloatTensor)
        
    #     _solver_form.append([free_coeff_weight, [None], 0])
    #     _solver_form.append(target_form)
    #     return _solver_form    
        
    def equation_solver_form(self, equation, variables, grids = None):
        if grids is None:
            grids = self.grids
        _solver_form = {}
        for term_idx, term in enumerate(equation.structure):
            if term_idx != equation.target_idx:
                _solver_form[term.name] = self.term_solver_form(term, grids, variables)
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
        target_form = self.term_solver_form(equation.structure[equation.target_idx], grids, variables)
        target_form['const'] = target_form['const'] * target_weight
        target_form['const'] = torch.flatten(target_form['const']).unsqueeze(1).type(torch.FloatTensor)        
        
        _solver_form[equation.structure[equation.target_idx].name] = target_form
        return _solver_form

    def use_grid(self, grids):
        if grids is None:
            self.grids = global_var.grid_cache.get_all()
        else:
            if len(self.grids) != len(global_var.grid_cache.get_all()):
                raise ValueError('Number of passed grids does not match the problem')
            self.grids = grids

    def form(self, full_domain : bool = True, grids : list = None):
        equation_forms = []
        bconds = []
        
        for idx, equation in enumerate(self.adaptee.vals):
            equation_forms.append(self.equation_solver_form(equation, grids = grids))
            bconds.append(equation.boundary_conditions(full_domain = full_domain, grids = grids, 
                                                       index = idx))
        
        return equation_forms, solver_formed_grid(grids), bconds