from typing import List
from functools import singledispatchmethod

import numpy as np
import torch

from epde.evaluators import simple_function_evaluator
from epde.structure.main_structures import SoEq
import epde.globals as global_var

def make_eval_func(eval_func, eval_func_kwargs):
    """
    Creates a wrapper function for evaluation.
    
    This method generates a lambda function that calls the provided
    evaluation function with the given keyword arguments. This allows
    for pre-configuring the evaluation function with specific settings
    without modifying its original signature.
    
    Args:
      eval_func: The evaluation function to be wrapped.
      eval_func_kwargs: Keyword arguments to be passed to the evaluation function.
    
    Returns:
      A lambda function that calls `eval_func` with the provided keyword arguments.
    """
    return lambda *args: eval_func(*args, **eval_func_kwargs)

class SystemSolverInterface(object):
    def __init__(self, system_to_adapt: SoEq, coeff_tol: float = 1.e-9, device = 'cpu'):
        """
        Initializes the SystemDescription object.
        
        This method prepares the SystemDescription object for describing a system,
        extracting its variables and setting up necessary parameters.
        
        Args:
            system_to_adapt: The system to be adapted and described.
            coeff_tol: Tolerance for coefficients.
            device: The device to use for computation (e.g., 'cpu', 'cuda').
        
        Fields:
            variables (list): A list of variables to describe, extracted from the input system.
            adaptee (SoEq): The system to be adapted, stored for later use.
            grids (None): Placeholder for grids, initialized to None.
            coeff_tol (float): Tolerance for coefficients, used in various calculations.
            _device (str): The device to use for computation.
        
        Returns:
            None.
        """
        self.variables = list(system_to_adapt.vars_to_describe)
        self.adaptee = system_to_adapt
        self.grids = None
        self.coeff_tol = coeff_tol

        self._device = device

    @staticmethod
    def _term_solver_form(term, grids, default_domain, variables: List[str] = ['u',], 
                          device = 'cpu') -> dict:
        """
        Transforms a symbolic term into a solver-compatible dictionary format.
        
        This method processes a symbolic term, extracting derivative information
        and coefficient values to create a dictionary suitable for use by a numerical solver.
        It handles both constant coefficients and coefficients that are functions of the grid.
        
        Args:
            term: The symbolic term to be transformed.
            grids: The grid points on which the term is evaluated.
            default_domain: A boolean indicating whether to use the default domain.
            variables: A list of variable names. Defaults to ['u'].
        
        Returns:
            dict: A dictionary containing the transformed term information, with the following keys:
                - 'coeff': The coefficient tensor or scalar value.
                - 'term': A list of derivative orders.
                - 'pow': A list of derivative powers.
                - 'var': A list of derivative variable indices.
        """
        deriv_orders = []
        deriv_powers = []
        deriv_vars = []
        derivs_detected = False

        try:
            coeff_tensor = torch.ones_like(grids[0]).to(device)
            
        except KeyError:
            raise NotImplementedError('No cache implemented')
        for factor in term.structure:
            if factor.is_deriv:
                for param_idx, param_descr in factor.params_description.items():
                    if param_descr['name'] == 'power':
                        power_param_idx = param_idx
                deriv_orders.append(factor.deriv_code)
                if factor.evaluator._evaluator != simple_function_evaluator:
                    if factor.evaluator._evaluator._single_function_token:
                        eval_func = factor.evaluator._evaluator._evaluation_functions_torch 
                    else:
                        eval_func = factor.evaluator._evaluator._evaluation_functions_torch[factor.label]
                    if not isinstance(eval_func, torch.nn.Sequential):
                        # print(f'for term {factor.name} eval func {eval_func} is non')
                        eval_func_kwargs = dict()
                        for key in factor.evaluator._evaluator.eval_fun_params_labels:
                            for param_idx, param_descr in factor.params_description.items():
                                if param_descr['name'] == key:
                                    eval_func_kwargs[key] = factor.params[param_idx]
                        # print(f'eval_func_kwargs for {factor.name} are {eval_func_kwargs}')
                        lbd_eval_func = make_eval_func(eval_func, eval_func_kwargs)
                    deriv_powers.append(lbd_eval_func)
                else:
                    deriv_powers.append(factor.params[power_param_idx])
                try:
                    if isinstance(factor.variable, str):
                        cur_deriv_var = variables.index(factor.variable)
                    elif isinstance(factor.variable, int) or (isinstance(factor.variable, (list, tuple)) and
                                                              isinstance(factor.variable[0], int)):
                        cur_deriv_var = factor.variable
                    elif isinstance(factor.variable, (list, tuple)) and isinstance(factor.variable[0], str):
                        cur_deriv_var = [variables.index(var_elem) for var_elem in factor.variable]
                except ValueError:
                    raise ValueError(
                        f'Variable family of passed derivative {variables}, other than {factor.variable}')
                derivs_detected = True

                deriv_vars.append(cur_deriv_var)
            else:
                grid_arg = None if default_domain else grids
                coeff_tensor = coeff_tensor * factor.evaluate(grids=grid_arg, torch_mode = True).to(device)
        if not derivs_detected:
            deriv_powers = [0,]
            deriv_orders = [[None,],]
        if len(deriv_powers) == 1:
            deriv_powers = [deriv_powers[0],]
            deriv_orders = [deriv_orders[0],]

        if deriv_vars == []:
            if isinstance(deriv_powers, int) and deriv_powers != 0:
                raise Exception('Something went wrong with parsing an equation for solver')
            # elif isinstance(deriv_powers, list) and all([spec_power != 0 for spec_power in deriv_powers]):
            #     raise Exception('Something went wrong with parsing an equation for solver')
            else:
                deriv_vars = [0,]

        if torch.all(torch.isclose(coeff_tensor.reshape(-1)[0], coeff_tensor.reshape(-1))):
            coeff_tensor = coeff_tensor.reshape(-1)[0].item()

        res = {'coeff': coeff_tensor,
               'term': deriv_orders,
               'pow': deriv_powers,
               'var': deriv_vars}

        # print(f'Translated {term.name} to "term" {deriv_orders}, "pow" {deriv_powers}, "var" {deriv_vars} ')
        return res

    @singledispatchmethod
    def set_boundary_operator(self, operator_info):
        """
        Sets the boundary operator.
        
        Args:
            operator_info: Information about the boundary operator.
        
        Returns:
            None.
        
        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError()

    def _equation_solver_form(self, equation, variables, grids=None, mode = 'NN') -> dict:
        """
        Forms the equation for the solver.
        
        This method takes an equation, variables, and grids, and transforms
        them into a dictionary suitable for the solver. It iterates through
        the terms of the equation, calculates their contributions, and
        organizes them into a structured format.
        
        Args:
            equation: The equation to be solved.
            variables: The variables involved in the equation.
            grids: The grids on which the equation is defined. Defaults to None,
                in which case the grids from the `self` object are used.
            mode: The mode of operation ('NN', 'autograd', or 'mat'). Defaults to 'NN'.
        
        Returns:
            dict: A dictionary representing the equation in a solver-friendly format.
                The dictionary contains terms of the equation as keys, and each term
                is represented by a dictionary containing its coefficient, term
                structure, power, and variable information.
        """
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
            if isinstance(grids[0], np.ndarray):
                grids = [torch.from_numpy(subgrid).to(self._device) for subgrid in grids]            
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
                    if isinstance(_solver_form[term.name]['coeff'], torch.Tensor):
                        _solver_form[term.name]['coeff'] = adjust_shape(_solver_form[term.name]['coeff'], mode = mode)

        free_coeff_weight = equation.weights_final[-1] #torch.full_like(input=grids[0], fill_value=equation.weights_final[-1]).to(self._device)

        # free_coeff_weight = adjust_shape(free_coeff_weight, mode = mode)
        free_coeff_term = {'coeff': free_coeff_weight,
                           'term': [None],
                           'pow': 0,
                           'var': [0,]}
        _solver_form['C'] = free_coeff_term

        target_weight = -1 # torch.full_like(input = grids[0], fill_value = -1.).to(self._device)

        target_form = self._term_solver_form(equation.structure[equation.target_idx], grids, default_domain, variables)
        target_form['coeff'] = target_form['coeff'] * target_weight
        # target_form['coeff'] = adjust_shape(target_form['coeff'], mode = mode)
        # print(f'target_form shape is {target_form["coeff"].shape}')

        _solver_form[equation.structure[equation.target_idx].name] = target_form

        return _solver_form

    def use_grids(self, grids=None): # 
        if grids is None and self.grids is None:
        """
        Uses provided grids or retrieves them from the global grid cache.
        
        If grids are not provided and the instance's grids are also None,
        it retrieves all grids from the global grid cache. If grids are
        provided, it checks if the number of provided grids matches the
        number of grids in the global grid cache. If the provided grids
        are NumPy arrays, they are converted to PyTorch tensors and moved
        to the device specified by `self._device`.
        
        Args:
            grids: A list of grids to use. If None, grids are retrieved from
                the global grid cache.
        
        Returns:
            None.
        
        Class Fields Initialized:
            grids (list of torch.Tensor): The grids used by the instance.
                Initialized with the provided grids or retrieved from the
                global grid cache.
        """
            _, self.grids = global_var.grid_cache.get_all(mode = 'torch')
        elif grids is not None:
            if len(grids) != len(global_var.grid_cache.get_all(mode = 'torch')[1]):
                raise ValueError(
                    'Number of passed grids does not match the problem')
            if isinstance(grids[0], np.ndarray):
                grids = [torch.from_numpy(subgrid).to(self._device) for subgrid in grids]
            self.grids = grids
            

    def form(self, grids=None, mode = 'NN'):
        """
        Forms the equations into a solver-friendly format.
        
        Transforms the equations stored in the adaptee into a format suitable
        for a numerical solver, associating each main variable with its
        corresponding equation form.
        
        Args:
          grids: Optional grid data to be used in the equation solving process.
          mode: String indicating the solving mode, defaults to 'NN'.
        
        Returns:
          A list of tuples, where each tuple contains the main variable to
          explain and its corresponding equation form ready for the solver.
        """
        self.use_grids(grids=grids)
        equation_forms = []

        for equation in self.adaptee.vals:
            equation_forms.append((equation.main_var_to_explain,
                                   self._equation_solver_form(equation, variables=self.variables,
                                                              grids=grids, mode = mode)))
        return equation_forms
