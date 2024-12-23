from typing import Union, List, Dict
from types import FunctionType
from functools import singledispatch

import numpy as np
import torch

import epde.globals as global_var
from epde.structure.main_structures import SoEq

VAL_TYPES = Union[FunctionType, int, float, torch.Tensor, np.ndarray]


def get_max_deriv_orders(system_sf: List[Union[Dict[str, Dict]]], variables: List[str] = ['u',]) -> dict:
    def count_factor_order(factor_code, deriv_ax):
        if factor_code is None or isinstance(factor_code, tuple):
            return 0
        else:
            if isinstance(factor_code, list):
                return factor_code.count(deriv_ax)
            elif isinstance(factor_code, int):
                return 1 if factor_code == deriv_ax else 0
            else:
                raise TypeError(f'Incorrect type of the input. Got {type(factor_code), factor_code}, expecting int or list')

    @singledispatch
    def get_equation_requirements(equation_sf, variables=['u',]):
        raise NotImplementedError(
            'Single-dispatch called in generalized form')

    @get_equation_requirements.register
    def _(equation_sf: dict, variables=['u',]) -> dict:  # dict = {u: 0}):
        dim = global_var.grid_cache.get('0').ndim
        if len(variables) == 1:
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
                    var_max_orders = np.maximum(var_max_orders, orders)
            return {variables[0]: var_max_orders}
        else:
            var_max_orders = {var_key: np.zeros(dim) for var_key in variables}
            for term_key, symb_form in equation_sf.items():
                if isinstance(symb_form['var'], list):
                    assert len(symb_form['term']) == len(symb_form['var'])
                    for factor_idx, deriv_factor in enumerate(symb_form['term']):
                        var_orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                                in np.arange(dim)])
                        if isinstance(symb_form['var'][factor_idx], int):
                            var_key = symb_form['var'][factor_idx] #- 1
                        else:
                            var_key = 0
                            var_orders = 0 # Such tokens do not increase order of the DE
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
            'TODO: add equation list form processing') 

    eq_forms = []
    for equation_form in system_sf:
        eq_forms.append(get_equation_requirements(equation_form, variables))

    max_orders = {var: np.maximum.accumulate([eq_list[var] for eq_list in eq_forms])[-1]
                    for var in variables}  # TODO
    return max_orders

class BOPElement(object):
    def __init__(self, axis: int, key: str, coeff: float = 1., term: list = [None], 
                 power: Union[Union[List[int], int]] = 1, var: Union[List[int], int] = 1,
                 rel_location: float = 0., device = 'cpu'):
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
        
        self._device = device

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
            return torch.from_numpy(res).to(self._device)
        else:
            return self._values

    @values.setter
    def values(self, vals):
        if isinstance(vals, (FunctionType, int, float, torch.Tensor)):
            self._values = vals
            self.vals_set = True
        elif isinstance(vals, np.ndarray):
            self._values = torch.from_numpy(vals).to(self._device)
            self.vals_set = True
        else:
            raise TypeError(
                f'Incorrect type of coefficients. Must be a type from list {VAL_TYPES}.')

    def __call__(self, values: VAL_TYPES = None) -> dict:
        if not self.vals_set and values is not None:
            self.values = values
            self.status['boundary_values_set'] = True
        elif not self.vals_set and values is None:
            raise ValueError('No location passed into the BOP.')
        if self.grid is not None:
            boundary = self.grid
        elif self.grid is None and self.location is not None:
            _, all_grids = global_var.grid_cache.get_all(mode = 'torch')

            abs_loc = self.location * all_grids[0].shape[self.axis]
            if all_grids[0].ndim > 1:
                boundary = np.array(all_grids[:self.axis] + all_grids[self.axis+1:])
                if isinstance(values, FunctionType):
                    raise NotImplementedError  # TODO: evaluation of BCs passed as functions or lambdas
                boundary = torch.from_numpy(np.expand_dims(boundary, axis=self.axis)).to(self._device).float()

                boundary = torch.cartesian_prod(boundary,
                                                torch.from_numpy(np.array([abs_loc,], dtype=np.float64)).to(self._device)).float()
                boundary = torch.moveaxis(boundary, source=0, destination=self.axis).resize()
            else:
                boundary = torch.from_numpy(np.array([[abs_loc,],])).to(self._device).float() # TODO: work from here
            
        elif boundary is None and self.location is None:
            raise ValueError('No location passed into the BOP.')
            
        form = self.operator_form
        boundary_operator = {form[0]: form[1]}
        
        boundary_value = self.values
        
        return {'bnd_loc' : boundary.to(self._device), 'bnd_op' : boundary_operator, 
                'bnd_val' : boundary_value.to(self._device), 
                'variables' : self.variables, 'type' : 'operator'}

class PregenBOperator(object):
    def __init__(self, system: SoEq, system_of_equation_solver_form: list): #, device = 'cpu'
        self.system = system
        self.equation_sf = [eq for eq in system_of_equation_solver_form]
        self.variables = list(system.vars_to_describe)

    def demonstrate_required_ords(self):
        linked_ords = list(zip([eq.main_var_to_explain for eq in self.system],
                                self.max_deriv_orders))

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
        return get_max_deriv_orders(self.equation_sf, self.variables)

    def generate_default_bc(self, vals: Union[np.ndarray, dict] = None, grids: List[np.ndarray] = None,
                            allow_high_ords: bool = False, required_bc_ord: List[int] = None):
        # Implement allow_high_ords - selection of derivatives from
        if required_bc_ord is None:
            required_bc_ord = self.max_deriv_orders
        assert set(self.variables) == set(required_bc_ord.keys()), 'Some conditions miss required orders.'

        grid_cache = global_var.initial_data_cache
        tensor_cache = global_var.initial_data_cache

        if vals is None:
            val_keys = {key: (key, (1.0,)) for key in self.variables}

        if grids is None:
            _, grids = grid_cache.get_all(mode = 'torch')

        device = global_var.grid_cache._device
        # assert self._device
        device_on_cpu = (device  == 'cpu')
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

                    if device_on_cpu:
                        coords = np.array([grids[idx][indexes].detach().numpy() for idx in np.arange(len(tensor_shape))]).T
                    else:
                        coords = np.array([grids[idx][indexes].detach().cpu().numpy()
                                           for idx in np.arange(len(tensor_shape))]).T
                    if coords.ndim > 2:
                        coords = coords.squeeze()

                    if vals is None:
                        bc_values = tensor_cache.get(val_keys[variable])[indexes]
                    else:
                        bc_values = vals[indexes]

                    bc_values = np.expand_dims(bc_values, axis=0).T
                    coords = torch.from_numpy(coords).to(device).float()

                    bc_values = torch.from_numpy(bc_values).to(device).float() # TODO: set devices for all torch objs
                    operator = BOPElement(axis=ax_idx, key=variable, coeff=1, term=[None],
                                          power=1, var=var_idx, rel_location=loc, device=device)
                    operator.set_grid(grid=coords)
                    operator.values = bc_values
                    bconds.append(operator)
        print('Types of conds:', [type(cond) for cond in bconds])
        self.conditions = bconds


class BoundaryConditions(object):
    def __init__(self, grids=None, partial_operators: dict = []):
        self.grids_set = (grids is not None)
        if grids is not None:
            self.grids = grids
        self.operators = partial_operators

    def form_operator(self):
        return [list(bcond()) for bcond in self.operators.values()]
