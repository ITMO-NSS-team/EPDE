from typing import Union, List, Dict, Tuple, Callable
import copy
from functools import reduce
from warnings import warn

import torch
import numpy as np
from scipy.integrate import ode, solve_ivp

from epde.solver.data import Domain, Conditions
from epde.structure.main_structures import SoEq
from epde.integrate.bop import PregenBOperator, BOPElement, get_max_deriv_orders
from epde.integrate.interface import SystemSolverInterface

def get_terms_der_order(equation: Dict, variable_idx: int) -> np.ndarray:
    '''
    Get the highest orders of the ``variable_idx``-th variable derivative in the equation terms.
    '''
    term_max_ord = np.zeros(len(equation))
    for term_idx, term_dict in enumerate(equation.values()):

        if isinstance(term_dict['var'], list) and len(term_dict['var']) > 1:
            max_ord = 0
            for arg_idx, deriv_ord in enumerate(term_dict['term']):
                if isinstance(term_dict['pow'][arg_idx], (int, float)) and term_dict['var'][arg_idx] == variable_idx:
                    max_ord = max(max_ord, len([var for var in deriv_ord if var is not None]))
            term_max_ord[term_idx] = max_ord
        elif isinstance(term_dict['var'], int):
            if isinstance(term_dict['pow'], (int, float)) and term_dict['var'] == variable_idx:
                term_max_ord[term_idx] = max(0, len([var for var in term_dict['term'] if var is not None]))
        elif isinstance(term_dict['var'], list) and len(term_dict['var']) == 1:
            if isinstance(term_dict['var'][0], (int, float)):
                term_var = term_dict['var'][0]
            elif isinstance(term_dict['var'][0], (list, tuple)):
                term_var = term_dict['var'][0][0]
            if (isinstance(term_dict['pow'], (int, float)) or (isinstance(term_dict['pow'], (list, tuple)) 
                                                               and len(term_dict['pow']) == 1)) and term_var == variable_idx:
                term_max_ord[term_idx] = max(0, len([var for var in term_dict['term'] if var is not None and var != [None,]]))
        pass

    return term_max_ord

def get_higher_order_coeff(equation: Dict, orders: np.ndarray, var: int) -> Tuple[List]:
    def transform_term(term: Dict, deriv_key: list, var: int) -> Dict:
        term_filtered = copy.deepcopy(term)
        if (isinstance(term['var'], int) and term['var'] == var) or (isinstance(term['var'], list) 
                                                                     and len(term['var']) == 1 and term['var'][0] == var):
            term_filtered['term'] = [None,]
            term_filtered['pow'] = 0
        else:
            term_idx = [der_var for idx, der_var in enumerate(term_filtered['term']) 
                        if der_var == deriv_key and term_filtered['pow'][idx] == var][0]
            term_filtered['term'][term_idx] = [None,]
            term_filtered['pow'][term_idx] = 0
        return term_filtered            

    denom_terms = []
    numer_terms = []
    for term_idx, term in enumerate(equation.values()):
        if orders[term_idx] == np.max(orders):
            denom_terms.append(transform_term(term, deriv_key=[0,]*int(np.max(orders)), var=var))
        else:
            numer_terms.append(term)
    return [denom_terms, numer_terms]

def get_eq_order(equation, variables: List[str]):
    eq_var = 0; eq_orders = np.zeros(len(equation))
    for var_idx in range(len(variables)):
        orders = get_terms_der_order(equation=equation, variable_idx=var_idx)
        if np.max(orders) > np.max(eq_orders):
            eq_var = var_idx; eq_orders = orders
    return eq_var, eq_orders

def replace_operator(term: Dict, variables: List):
    '''

    Variables have to be in form of [(0, [None]), (0, [0,]), (0, [0, 0]), (0, [0, 0, 0]), (1, [None,]), ... ]
    where the list elements are factors, taken as derivatives: (variable, differentiations), and the index in list
    matches the index of dynamics operator output.

    '''
    term_ = copy.deepcopy(term)
    if isinstance(term_['var'], list) and len(term_['var']) > 1:
        for arg_idx, deriv_ord in enumerate(term_['term']):
            if isinstance(term_['var'][arg_idx], (tuple, list)):
                continue
            term_['var'][arg_idx]  = variables.index((term_['var'][arg_idx], deriv_ord))
            term_['term'][arg_idx] = [None,]
    elif isinstance(term['var'], int) or (isinstance(term_['var'], list) and len(term_['var']) == 1):
        if isinstance(term['var'], int):
            term_var = term_['var']
        else:
            term_var = term_['var'][0]
        if isinstance(term['pow'], (int, float)):
            term_['var']  = variables.index((term_var, term_['term']))
            term_['term'] = [None,]
    return term_

class ImplicitEquation(object):
    def __init__(self, system: List, grid: np.ndarray, variables: List[str]):
        self.grid_dict = grid

        # print(f'Solved system is {system}')

        self._dynamics_operators = []
        self._var_order = []
        self._vars_with_eqs = {}

        for var, order in [get_eq_order(equation, variables) for equation in system]:
            self._var_order.extend([(var, [None,])] + [(var, [0,]*(idx+1)) for idx in range(int(np.max(order))-1)])
            if len(self._vars_with_eqs) == 0:
                self._vars_with_eqs[int(np.max(order)) - 1] = (var, order)
            else:
                self._vars_with_eqs[list(self._vars_with_eqs.keys())[-1] + int(np.max(order))] = (var, order)

        for var_idx, var in enumerate(self._var_order):
            if var_idx in self._vars_with_eqs.keys():
                operator = get_higher_order_coeff(equation = system[self._vars_with_eqs[var_idx][0]],
                                                  orders = self._vars_with_eqs[var_idx][1], 
                                                  var = self._vars_with_eqs[var_idx][0])
                operator[0] = [replace_operator(denom_term, self._var_order) for denom_term in operator[0]]
                operator[1] = [replace_operator(numer_term, self._var_order) for numer_term in operator[1]]
            else:
                operator = [None, self.create_first_ord_eq(var_idx + 1)]
            self._dynamics_operators.append(operator)

    def parse_cond(self, conditions: List[Union[BOPElement, dict]]):
        cond_val = np.full(shape = len(self._dynamics_operators), fill_value=np.inf)
        for cond in conditions:
            if isinstance(cond, BOPElement):
                assert isinstance(cond.variables, int), 'Boundary operator has to contain only a single variable.'
                try:
                    var = self._var_order.index((cond.variables, cond.axis))
                except ValueError:
                    print(f'Missing {cond.variables, cond.axis} from the list of variables {self._var_order}')
                    raise RuntimeError()
                cond_val[var] = cond.values
            else:
                op_form = list(cond['bnd_op'].values())[0]
                term_key = [op_key for op_key in list(op_form.keys()) if op_key not in ['coeff', 'pow', 'var']][0]
                try:
                    # print(f'term key {term_key}')
                    var = self._var_order.index((op_form['var'], op_form[term_key]))
                except ValueError:
                    print(f'Missing {(op_form["var"], op_form[term_key])} from the list of variables {self._var_order}')
                    raise RuntimeError()
                cond_val[var] = cond['bnd_val']

        assert np.sum(np.inf == cond_val) == 0, 'Not enough initial conditions were passed.'
        return cond_val

    def __call__(self, t, y):
        values = np.empty(len(self._dynamics_operators))
        for idx, operator in enumerate(self._dynamics_operators):
            if operator[0] is None:
                denom = 1
            else:
                denom = [self.term_callable(term, t, y) for term in operator[0]]
                if np.isclose(denom, 0):
                    raise ZeroDivisionError('Denominator in the dynamics operator is close to zero.')
            numerator = [self.term_callable(term, t, y) for term in operator[1]]
            values[idx] = -1*np.sum(numerator)/np.sum(denom)
        return values

    @property
    def grid_dict(self):
        return self._grid_rounded

    @grid_dict.setter
    def grid_dict(self, grid_points):
        self._grid_step = grid_points[1] - grid_points[0]
        digits = np.floor(np.log10(self._grid_step/2.)-1)
        self._grid_rounded = {np.round(grid_val, -int(digits)): idx 
                              for idx, grid_val in np.ndenumerate(grid_points)}

    def create_first_ord_eq(self, var: int) -> List[Tuple]:
        '''
        Example of order: np.array([3., 0., 0.]) for third ord eq. 
        '''
        return [{'coeff' : -1.,
                 'term'  : [None,],
                 'pow'   : 1,
                 'var'   : var},]

    def merge_coeff(self, coeff: np.ndarray, t: float):
        try:
            return self.grid_dict[t]
        except KeyError:
            for grid_loc, grid_idx in self.grid_dict.items():
                if grid_loc < t and grid_loc + self._grid_step > t:
                    # print('Search in ', grid_loc, grid_loc + self._grid_step)
                    left_loc, right_loc = grid_loc, grid_loc + self._grid_step
                    left_idx, right_idx = grid_idx[0], grid_idx[0] + 1
                    break
            val = coeff[left_idx] + (t - left_loc) / (right_loc - left_loc) * (coeff[right_idx] - coeff[left_idx])
            return val

    def term_callable(self, term: Dict, t, y):
        def call_ann_token(token_nn: torch.nn.Sequential, arguments: list,
                           t: float, y: np.ndarray):
            return token_nn[torch.from_numpy(y[tuple(arguments)]).reshape((-1, 1))].detach().numpy() # Hereby, the ANN does not explicitly depend on time

        if isinstance(term['coeff'], Callable):
            k = term['coeff'](t)
        elif isinstance(term['coeff'], torch.nn.Sequential):
            k = term['coeff'](torch.from_numpy(t).reshape((1, 1).float()))
        elif isinstance(term['coeff'], np.ndarray):
            k = self.merge_coeff(term['coeff'], t)
        else:
            k = term['coeff']
        
        if not isinstance(term['var'], (list, tuple)) or len(term['pow']) == 1:
            term_var = [term['var'],]
        else:
            term_var = term['var']
        if isinstance(term['var'], (list, tuple)):
            term_pow = term['pow']
        else:
            term_pow = [term['pow'],]
        
        values = []
        for var_idx, var in enumerate(term_var):  
            if isinstance(var, int):
                if isinstance(term_pow[var_idx], (int, float)):
                    val = y[var]**term_pow[var_idx]
                elif isinstance(term_pow[var_idx], torch.nn.Sequential):
                    val = call_ann_token(term_pow[var_idx], var, t, y)
                else:                
                    val = term_pow[var_idx](y[var])
            elif isinstance(var, (tuple, list)):
                if isinstance(term_pow[var_idx], torch.nn.Sequential):
                    val = call_ann_token(term_pow[var_idx], var, t, y)
                elif isinstance(term_pow[var_idx], (int, float)):
                    assert len(var) == 1, 'Incorrect number of arguments'
                    val = y[list(var)]**term_pow[var_idx]
                    if isinstance(val, np.ndarray): val = val[0]
                    
                    # print(values[-1], type(values[-1]))
                else:               
                    val = term_pow[var_idx](*y[list(var)])
                    if isinstance(val, torch.Tensor):
                        val = val.item()

            values.append(val)
            pass

        return reduce(lambda x, z: x*z, values, k)


class OdeintAdapter(object):
    def __init__(self, method: str = 'Radau'):
        self._solve_method = method
        pass # TODO: implement hyperparameters setup, according to the problem specifics

    def solve_epde_system(self, system: Union[SoEq, dict], grids: list=None, boundary_conditions=None,
                          mode='NN', data=None, vars_to_describe = ['u'], *args, **kwargs):
        if isinstance(system, SoEq):
            system_interface = SystemSolverInterface(system_to_adapt=system)
            system_solver_forms = system_interface.form(grids = grids, mode = mode)
        elif isinstance(system, list):
            system_solver_forms = system
        else:
            raise TypeError('Incorrect input into the Odeint Adapter.')

        if boundary_conditions is None:
            op_gen = PregenBOperator(system=system,
                                     system_of_equation_solver_form=[sf_labeled[1] for sf_labeled
                                                                     in system_solver_forms])
            op_gen.generate_default_bc(vals = data, grids = grids)
            boundary_conditions = op_gen.conditions

        if isinstance(system, SoEq):
            vars_to_describe = system.vars_to_describe
            
        return self.solve(equations = [sf_labeled[1] for sf_labeled in system_solver_forms], domain = grids[0], 
                          boundary_conditions = boundary_conditions, vars = vars_to_describe)
        # Add condition parser and control function args parser



    def solve(self, equations, domain: Union[Domain, np.ndarray],
              boundary_conditions: List[BOPElement] = None, vars: List[str] = ['x',], *args, **kwargs):
        if not isinstance(equations, list):
            raise RuntimeError('Incorrect type of equations passed into odeint solver.')
        self._implicit_equation = ImplicitEquation(equations, domain, vars)
        if isinstance(domain, Domain): 
            grid = domain.build().detach().numpy().reshape(-1)
        else:
            grid = domain.detach().numpy().reshape(-1)

        initial_cond = self._implicit_equation.parse_cond(boundary_conditions)
        solution = solve_ivp(fun = self._implicit_equation, t_span = (grid[0], grid[-1]), y0=initial_cond,
                             t_eval = grid, method = self._solve_method)
        if not solution.success:
            warn(f'Numerical solution of ODEs has did not converge. The error message is {solution.message}')
        return 0, solution.y.T
