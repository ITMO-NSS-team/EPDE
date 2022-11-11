'''

'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

from typing import Callable, Union
from types import FunctionType
COEFF_TYPES = Union[FunctionType, int, float, torch.Tensor, np.ndarray]

from functools import singledispatchmethod

from epde.structure.main_structures import Equation, SoEq
import epde.globals as global_var

class PregenOperator(object):
    def __init__(self, system : SoEq, s_of_equation_solver_form : list):
        self.equation_sf = [eq for eq in s_of_equation_solver_form]
        self.max_ord = max_deriv_orders()
        
    def demonstrate_required_ords(self):
        linked_ords = list(zip([eq.main_var_to_explain for eq in system], 
                               self.max_deriv_orders(self.equation_sf)))
        print(f'Orders, required by an equation, are as follows: {linked_ords}')
        
    @property
    def conditions(self):
        return self._conditions
    
    @conditions.setter
    def conditions(self, cond : dict):
        assert len(conditions) == sum(self.max_deriv_orders(self.equation_sf))
        
        
    def parse_equation(self, conditions : list, full_domain : bool = True, grids = None):
        required_orders = self.max_deriv_orders(self.equation_sf)
        assert len(conditions)
        
    @staticmethod
    def max_deriv_orders(system_sf : list) -> np.ndarray:
        for equation_form in system_sf
            max_orders = np.zeros(global_var.grid_cache.get('0').ndim)
    
            def count_order(obj, deriv_ax):
                if obj is None:
                    return 0
                else:
                    return obj.count(deriv_ax)
    
            for term in equation_sf:
                if isinstance(term[2], list):
                    for deriv_factor in term[1]:
                        orders = np.array([count_order(deriv_factor, ax) for ax
                                           in np.arange(max_orders.size)])
                        max_orders = np.maximum(max_orders, orders)
                else:
                    orders = np.array([count_order(term[1], ax) for ax
                                       in np.arange(max_orders.size)])
                    max_orders = np.maximum(max_orders, orders)
            if np.max(max_orders) > 4:
                raise NotImplementedError('The current implementation allows does not allow higher orders of equation, than 2.')
            return max_orders
    
    

class BOPElement(object):
    def __init__(self, key : str, coeff : COEFF_TYPES, term : list = [None], 
                 power : Union[list, int] = 1, var : Union[list, int] = 0):
        self.key = key
        self.coefficient = coeff
        self.term = term
        self.power = power
        self.variables = var
        
        self.set_grids(grids)
    
    def set_grids(self, grids : list):
        assert isinstance(grids[0], np.ndarray), 'Grids have to be in formats of numpy.ndarray.'
        self.grids = grids
    
    def form_operator(self):
        form = {
                'coeffs' : self.coefficient,
                self.key   : self.term,
                'pow'  : self.powers,
                'var'    : self.variables
                }
        return self.key, form
        
    @property
    def coefficient(self):
        if isinstance(self._coefficient, FunctionType):
            assert self.grid_set, 'Tring to evaluate variable coefficent without a proper grid.'
            res = self._coefficient(self.grids)
            assert res.shape == self.grids[0].shape
            return torch.from_numpy(res)
        else:
            return self._coefficient
    
    @coefficient.setter
    def coefficient(self, coeff):
        if isinstance(coeff, (FunctionType, int, float, torch.Tensor)):
            self._coefficient = coeff
        elif isinstance(coeff, np.ndarray):
            if self.grids is not None and np.shape(coeff) != np.shape(self.grids[0]):
                raise ValueError('Numpy array of coefficients does not match the grid.')
            self._coefficient = torch.from_numpy(coeff)
        else:
            raise TypeError(f'Incorrect type of coefficients. Must be a type from list {coeff_types}.')
        

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
    def __init__(self, system_to_adapt):
        self.adaptee = system_to_adapt
        assert self.adaptee.weights_final_evald      
    
    @staticmethod
    def term_solver_form(term, grids):
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
    
    @singledispatchmethod
    def set_boundary_operator(self, operator_info):
        
        
    def equation_solver_form(self, equation, grids):
        _solver_form = []
        for term_idx in range(len(equation.structure)):
            if term_idx != equation.target_idx:
                term_form = self.term_solver_form(equation.structure[term_idx], grids)  #equation.structure[term_idx].solver_form
                if term_idx < equation.target_idx:
                    weight = equation.weights_final[term_idx] 
                else:
                    weight = equation.weights_final[term_idx-1]
                term_form[0] = term_form[0] * weight
                term_form[0] = torch.flatten(term_form[0]).unsqueeze(1).type(torch.FloatTensor)
                _solver_form.append(term_form)
                
        free_coeff_weight = torch.from_numpy(np.full_like(a = global_var.grid_cache.get('0'), 
                                                          fill_value = equation.weights_final[-1]))
        free_coeff_weight = torch.flatten(free_coeff_weight).unsqueeze(1).type(torch.FloatTensor)
        target_weight = torch.from_numpy(np.full_like(a = global_var.grid_cache.get('0'), 
                                                          fill_value = -1.))            
        target_form = equation.structure[equation.target_idx].solver_form
        target_form[0] = target_form[0] * target_weight
        target_form[0] = torch.flatten(target_form[0]).unsqueeze(1).type(torch.FloatTensor)
        
        _solver_form.append([free_coeff_weight, [None], 0])
        _solver_form.append(target_form)
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