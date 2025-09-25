#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:26:56 2021

@author: mike_ubuntu
"""
from abc import ABC
from collections import OrderedDict
from typing import Union, Callable, List, Tuple

import numpy as np
import torch

from epde.supplementary import define_derivatives
from epde.preprocessing.derivatives import preprocess_derivatives

import epde.globals as global_var
from epde.interface.token_family import TokenFamily
from epde.cache.cache import upload_simple_tokens, prepare_var_tensor  # np_ndarray_section,

from epde.evaluators import CustomEvaluator, EvaluatorTemplate, trigonometric_evaluator, \
     simple_function_evaluator, const_evaluator, const_grad_evaluator, grid_evaluator, \
     velocity_evaluator, velocity_grad_evaluators, phased_sine_evaluator, sign_evaluator

class PreparedTokens(ABC):
    """
    Base class for tokens that have a corresponding class definition within the system.
    
    
        Attributes:
            _token_family  (`TokenFamily`): the family of functions to which the token belongs
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the `PreparedTokens` object.
        
                The `PreparedTokens` class manages a specific family of tokens, in this case, 'Placeholder' tokens. This initialization prepares the token family for use in the equation discovery process. By creating a `TokenFamily` instance with the 'Placeholder' type, the system ensures that placeholder tokens are properly handled during the search for the best equation structure.
        
                Args:
                    *args: Variable length argument list.
                    **kwargs: Arbitrary keyword arguments.
        
                Returns:
                    None.
        """
        self._token_family = TokenFamily(token_type='Placeholder')

    @property
    def token_family(self):
        """
        Returns the token family associated with the layer.
        
        This property provides access to the token family, which defines the building blocks for constructing equation candidates.
        It ensures that the token family is fully configured before being accessed, guaranteeing that all necessary components for equation discovery are available.
        
        Args:
            None
        
        Raises:
            AttributeError: If the token family's evaluator or parameters have not been set,
                            indicating that the equation search space is not yet fully defined.
        
        Returns:
            TokenFamily: The token family object, representing the set of allowed operations and variables in the equation search.
        """
        if not (self._token_family.evaluator_set and self._token_family.params_set):
            raise AttributeError(f'Some attributes of the token family have not been declared.')
        return self._token_family

class ArbitraryDataFunction(PreparedTokens):
    """
    Class for tokens, representing arbitrary functions of the modelled variable passed in `var_name` or its derivatives.
    """

    def __init__(self, token_type: str, var_name: str, token_labels: list,
                 evaluator: Union[CustomEvaluator, EvaluatorTemplate, Callable],
                 params_ranges: dict, params_equality_ranges: dict = None, unique_specific_token=True, 
                 unique_token_type=True, meaningful=True, non_default_power = False,
                 deriv_solver_orders: list = [[None,],]): # Add more intuitive method of declaring solver orders
        """
        Class for tokens, representing arbitrary functions of the modelled variable passed in `var_name` or its derivatives.  
        """        
        self._token_family = TokenFamily(token_type = token_type, variable = var_name,
                                         family_of_derivs=True)

        self._token_family.set_status(demands_equation=False, meaningful=meaningful,
                                      unique_specific_token=unique_specific_token, unique_token_type=unique_token_type,
                                      s_and_d_merged=False, non_default_power = non_default_power)

        self._token_family.set_params(token_labels, params_ranges, params_equality_ranges,
                                      derivs_solver_orders=deriv_solver_orders)  
        self._token_family.set_evaluator(evaluator)

class DerivSignFunction(PreparedTokens):
    """
    Class for representing the derivative sign function.
    
        This class represents the sign function of a derivative of a variable.
    """

    def __init__(self, token_type: str, var_name: str, token_labels: list, unique_specific_token=True, 
                #  evaluator: Union[CustomEvaluator, EvaluatorTemplate, Callable], 
                #  params_ranges: dict, params_equality_ranges: dict = None,
                 unique_token_type=True, meaningful=True, non_default_power = False,
                 deriv_solver_orders: list = [[None,],]): # Add more intuitive method of declaring solver orders
        """
        Class for tokens, representing arbitrary functions of the modelled variable passed in `var_name` or its derivatives.  
        """        
        self._token_family = TokenFamily(token_type = token_type, variable = var_name,
                                         family_of_derivs=True)

        self._token_family.set_status(demands_equation=False, meaningful=meaningful,
                                      unique_specific_token=unique_specific_token, unique_token_type=unique_token_type,
                                      s_and_d_merged=False, non_default_power = non_default_power)

        params_ranges = OrderedDict([('power', (1, 1))])
        params_equality_ranges = {'power': 0}
        self._token_family.set_params(token_labels, params_ranges, params_equality_ranges,
                                      derivs_solver_orders=deriv_solver_orders)  
        self._token_family.set_evaluator(sign_evaluator)

class DataPolynomials(PreparedTokens):
    """
    Class for generating and managing polynomial features from input data.
    
        Attributes:
            max_power: The maximum power to which the input data will be raised.
    """

    def __init__(self, var_name: str, max_power: int = 1):
        """
        Initializes a family of tokens representing power products of a specified variable.
        
                This class creates tokens that represent the variable raised to different powers,
                up to a specified maximum. These tokens are designed to be included in the pool
                of potential terms when searching for differential equations. By using powers of
                the variable, the search space is expanded to include polynomial relationships.
        
                Args:
                    var_name (str): The name of the variable to be exponentiated.
                    max_power (int, optional): The maximum power to which the variable will be raised. Defaults to 1.
        
                Returns:
                    None
        """
        self._token_family = TokenFamily(token_type=f'poly of {var_name}', variable = var_name,
                                         family_of_derivs=True)
        
        def latex_form(label, **params):
            '''
            Parameters
            ----------
            label : str
                label of the token, for which we construct the latex form.
            **params : dict
                dictionary with parameter labels as keys and tuple of parameter values 
                and their output text forms as values.

            Returns
            -------
            form : str
                LaTeX-styled text form of token.
            '''            
            if '/' in label:
                label = label[:label.find('x')+1] + '_' + label[label.find('x')+1:]
                label = label.replace('d', r'\partial ').replace('/', r'}{')
                label = r'\frac{' + label + r'}'
                                
            if params['power'][0] > 1:
                label = r'\left(' + label + r'\right)^{{{0}}}'.format(params["power"][1])
            return label
        
        self._token_family.set_latex_form_constructor(latex_form)
        self._token_family.set_status(demands_equation=False, meaningful=True,
                                      unique_specific_token=True, unique_token_type=True,
                                      s_and_d_merged=False, non_default_power = True)
        self._token_family.set_params([var_name,], OrderedDict([('power', (1, max_power))]), 
                                      {'power': 0}, [[None,],])
        self._token_family.set_evaluator(simple_function_evaluator)
        
class DataSign(PreparedTokens):
    """
    Class for signing data using a specified method.
    """

    def __init__(self, var_name: str, max_power: int = 1):
        """
        Represents a family of tokens that are polynomial powers of a single variable. This allows the search space to include polynomial terms of the modeled variable.
        
                Args:
                    var_name (str): The name of the independent variable to be exponentiated.
                    max_power (int): The maximum power to which the variable will be raised.  This defines the range of polynomial terms to consider.
        
                Returns:
                    None
        """
        raise NotImplementedError('TBD.')
        self._token_family = TokenFamily(token_type=f'poly of {var_name}', variable = var_name,
                                         family_of_derivs=True)
        
        def latex_form(label, **params):
            '''
            Parameters
            ----------
            label : str
                label of the token, for which we construct the latex form.
            **params : dict
                dictionary with parameter labels as keys and tuple of parameter values 
                and their output text forms as values.

            Returns
            -------
            form : str
                LaTeX-styled text form of token.
            '''            
            if '/' in label:
                label = label[:label.find('x')+1] + '_' + label[label.find('x')+1:]
                label = label.replace('d', r'\partial ').replace('/', r'}{')
                label = r'\frac{' + label + r'}'
                                
            if params['power'][0] > 1:
                label = r'\left(' + label + r'\right)^{{{0}}}'.format(params["power"][1])
            return label
        
        self._token_family.set_latex_form_constructor(latex_form)
        self._token_family.set_status(demands_equation=False, meaningful=True,
                                      unique_specific_token=True, unique_token_type=True,
                                      s_and_d_merged=False, non_default_power = True)
        self._token_family.set_params([var_name,], OrderedDict([('power', (1, max_power))]), 
                                      {'power': 0}, [[None,],])
        self._token_family.set_evaluator(simple_function_evaluator)    

class ControlVarTokens(PreparedTokens):
    """
    Represents a family of control variable tokens for use in symbolic regression.
    
        This class manages the token type, variable names, derivative orders, and
        evaluation functions associated with control variables in a symbolic
        regression context. It supports both single and multiple control components
        and prepares data for evaluation using neural networks or custom functions.
    
        Attributes:
            _token_family (TokenFamily): An instance of the TokenFamily class, configured for control tokens.
                It manages the token type, variable names, derivative orders, and evaluation functions.
            _token_family.ftype (str): The type of the token family, set to 'ctrl'.
            _token_family.variable (str): The variable associated with the token family.
            _token_family.family_of_derivs (bool): A flag indicating whether the token family includes derivatives (True).
            _token_family.evaluator_set (bool): A flag indicating whether the evaluator has been set (True after initialization).
            _token_family.params_set (bool): A flag indicating whether the parameters have been set (True after initialization).
            _token_family.cache_set (bool): A flag indicating whether the cache has been set (False by default).
            _token_family.deriv_evaluator_set (bool): A flag indicating whether the derivative evaluator has been set (True by default).
    """

    def __init__(self, sample: Union[np.ndarray, List[np.ndarray]], ann: torch.nn.Sequential = None, 
                 var_name: Union[str, List[str]] = 'ctrl', arg_var: List[Tuple[Union[int, List]]] = [(0, [None,]),], 
                 eval_torch: Union[Callable, dict] = None, eval_np: Union[Callable, dict] = None, device:str = 'cpu'):
        """
        Initializes the object, configuring the token family for control variables and setting up the evaluation functions.
        
                This setup is crucial for representing and evaluating control variables within the equation discovery process. It prepares the data and neural network (if provided) for use in the evolutionary search for differential equations. The evaluation functions determine how well the control variable, potentially learned by a neural network, fits the data.
        
                Args:
                    sample: Sample data, can be a NumPy array or a list of NumPy arrays.
                    ann: A PyTorch Sequential model for the control neural network.
                    var_name: Name(s) of the control variable(s). Can be a string or a list of strings.
                    arg_var: List of tuples, where each tuple contains the variable index and derivative orders.
                    eval_torch: Custom evaluation function for PyTorch tensors. Can be a callable or a dictionary.
                    eval_np: Custom evaluation function for NumPy arrays. Can be a callable or a dictionary.
                    device: The device to run the neural network on ('cpu' or 'cuda').
                
                Fields:
                    _token_family (TokenFamily): An instance of the TokenFamily class, configured for control tokens.
                        It manages the token type, variable names, derivative orders, and evaluation functions.
                    _token_family.ftype (str): The type of the token family, set to 'ctrl'.
                    _token_family.variable (str): The variable associated with the token family.
                    _token_family.family_of_derivs (bool): A flag indicating whether the token family includes derivatives (True).
                    _token_family.evaluator_set (bool): A flag indicating whether the evaluator has been set (True after initialization).
                    _token_family.params_set (bool): A flag indicating whether the parameters have been set (True after initialization).
                    _token_family.cache_set (bool): A flag indicating whether the cache has been set (False by default).
                    _token_family.deriv_evaluator_set (bool): A flag indicating whether the derivative evaluator has been set (True by default).
                
                Returns:
                    None
        """
        vars, der_ords = zip(*arg_var)
        if isinstance(sample, List):
            assert isinstance(var_name, List), 'Both samples and var names have to be set as Lists or single elements.'
            num_ctrl_comp = len(var_name)
        else:
            num_ctrl_comp = 1

        token_params = OrderedDict([('power', (1, 1)),])
        
        equal_params = {'power': 0}

        self._token_family = TokenFamily(token_type = 'ctrl', variable = vars,
                                         family_of_derivs=True)

        self._token_family.set_status(demands_equation=False, meaningful=True,
                                      unique_specific_token=True, unique_token_type=True,
                                      s_and_d_merged=False, non_default_power = False)
        if isinstance(var_name, str): var_name = [var_name,]
        self._token_family.set_params(var_name, token_params, equal_params,
                                      derivs_solver_orders=[der_ords for label in var_name])
        
        def nn_eval_torch(*args, **kwargs):
            if isinstance(args[0], torch.Tensor):
                inp = torch.cat([torch.reshape(tensor, (-1, 1)) for tensor in args], dim = 1).to(device)
            else:
                inp = torch.cat([torch.reshape(torch.Tensor([elem,]), (-1, 1)) for elem in args], dim = 1).to(device)
            return global_var.control_nn.net(inp).to(device)

        def nn_eval_np(*args, **kwargs):
            return nn_eval_torch(*args, **kwargs).detach().cpu().numpy()

        if eval_np    is None: eval_np = nn_eval_np
        if eval_torch is None: eval_torch = nn_eval_torch

        eval = CustomEvaluator(evaluation_functions_np = eval_np,
                               evaluation_functions_torch = eval_torch,
                               eval_fun_params_labels = ['power'])

        global_var.reset_control_nn(n_control = num_ctrl_comp, ann = ann, 
                                    ctrl_args = arg_var, device = device)
        if isinstance(sample, np.ndarray):
            global_var.tensor_cache.add(tensor = sample, label = (var_name[0], (1.0,)))
        else:
            for idx, var_elem in enumerate(var_name):
                global_var.tensor_cache.add(tensor = sample[idx], label = (var_elem, (1.0,)))

        self._token_family.set_evaluator(eval)

class TrigonometricTokens(PreparedTokens):
    """
    Class for prepared tokens, that belongs to the trigonometric family
    """

    def __init__(self, freq: tuple = (np.pi/2., 2*np.pi), dimensionality=1):
        """
        Initializes a family of trigonometric tokens for equation discovery.
        
                This method sets up the building blocks for representing trigonometric functions
                within the equation search space. It defines the token family, its parameters
                (frequency and dimensionality), and the evaluation function. This setup allows
                the evolutionary algorithm to explore trigonometric terms as potential components
                of the discovered differential equations.
        
                Args:
                    freq (`tuple`): optional, default - (pi/2., 2*pi)
                        Interval for the frequency parameter within the trigonometric token.
                        This range constrains the frequencies that can be explored during the
                        equation discovery process.
                    dimensionality (`int`): optional, default - 1
                        The dimensionality of the input data that the trigonometric token operates on.
                        This specifies the number of independent variables the token can accept.
        
                Returns:
                    None
        """
        assert freq[1] > freq[0] and len(freq) == 2, 'The tuple, defining frequncy interval, shall contain 2 elements with first - the left boundary of interval and the second - the right one. '

        self._token_family = TokenFamily(token_type='trigonometric')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True,
                                      meaningful=False)
            
        def latex_form(label, **params):
            '''
            Parameters
            ----------
            label : str
                label of the token, for which we construct the latex form.
            **params : dict
                dictionary with parameter labels as keys and tuple of parameter values 
                and their output text forms as values.

            Returns
            -------
            form : str
                LaTeX-styled text form of token.
            '''
            form = label + r'^{{{0}}}'.format(params["power"][1]) + \
                    r'(' + params["freq"][1] + r' x_{' + params["dim"][1] + r'})'
            return form
        
        self._token_family.set_latex_form_constructor(latex_form)
        trig_token_params = OrderedDict([('power', (1, 1)),
                                         ('freq', freq),
                                         ('dim', (0, dimensionality))])
        print(f'trig_token_params: VALUES = {trig_token_params["dim"]}')
        freq_equality_fraction = 0.05  # fraction of allowed frequency interval, that is considered as the same
        trig_equal_params = {'power': 0, 'freq': (freq[1] - freq[0]) / freq_equality_fraction,
                             'dim': 0}
        self._token_family.set_params(['sin', 'cos'], trig_token_params, trig_equal_params)
        self._token_family.set_evaluator(trigonometric_evaluator)


class PhasedSine1DTokens(PreparedTokens):
    """
    Represents a family of phased sine wave tokens in 1D.
    
        This class sets up a token family for phased sine waves, defines their
        parameters, and configures how they are evaluated and represented in
        LaTeX format.
    
        Attributes:
            _token_family (TokenFamily): An instance of the TokenFamily class, configured for 'phased_sine_1d' tokens.
            _token_family.ftype (str): The type of token family, set to 'phased_sine_1d'.
            _token_family.variable (None): The variable associated with the token family, initialized to None.
            _token_family.family_of_derivs (bool): A flag indicating if the token family includes derivatives, initialized to False.
            _token_family.evaluator_set (bool): A flag indicating if the evaluator is set, initialized to False.
            _token_family.params_set (bool): A flag indicating if the parameters are set, initialized to False.
            _token_family.cache_set (bool): A flag indicating if the cache is set, initialized to False.
            _token_family.deriv_evaluator_set (bool): A flag indicating if the derivative evaluator is set, initialized to True.
            _token_family.latex_form_constructor (Callable): A callable object used to construct LaTeX representations of tokens, initialized by `set_latex_form_constructor`.
    """

    def __init__(self, freq: tuple = (np.pi/2., 2*np.pi)):
        """
        Initializes the object, configuring a token family for phased sine waves.
        
                This setup defines the token's parameters (frequency, phase, power), their evaluation method,
                and their representation in LaTeX format, which is essential for symbolic manipulation and display
                of discovered equations. The frequency range is specified to constrain the search space
                during the equation discovery process.
        
                Args:
                    freq (tuple): A tuple containing the minimum and maximum frequency values.
        
                Returns:
                    None
        """
        self._token_family = TokenFamily(token_type='phased_sine_1d')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True,
                                      meaningful=False)
        
        sine_token_params = OrderedDict([('power', (1, 1)),#tuple([(1, 1) for idx in range(dimensionality)])),
                                         ('freq', freq),
                                         ('phase', (0., 1.))])

        freq_equality_fraction = 0.05  # fraction of allowed frequency interval, that is considered as the same

        def latex_form(label, **params):
            '''
            Parameters
            ----------
            label : str
                label of the token, for which we construct the latex form.
            **params : dict
                dictionary with parameter labels as keys and tuple of parameter values 
                and their output text forms as values.

            Returns
            -------
            form : str
                LaTeX-styled text form of token.
            '''            
            pwr_sign = r'^{{{0}}}'.format(params["power"][1]) if params["power"][0] != 1 else ''
            return label + pwr_sign + r'(' + params["freq"][1] + r' x_{1} + ' \
                   + params["phase"][1] + r')'
        
        self._token_family.set_latex_form_constructor(latex_form)
        sine_equal_params = {'power': 0, 'freq': (freq[1] - freq[0]) / freq_equality_fraction,
                             'phase': 0.05}
        self._token_family.set_params(['sine',], sine_token_params, sine_equal_params)
        self._token_family.set_evaluator(phased_sine_evaluator)        


class GridTokens(PreparedTokens):
    """
    Class for prepared tokens, that describe family of grids as values
    """

    def __init__(self, labels = ['t',], max_power: int = 1, dimensionality=1):
        """
        Initializes a `GridTokens` object, defining a family of tokens representing grid-based features.
        
                This class sets up the token family with specified labels, maximum power, and dimensionality. 
                It configures how these tokens are represented in LaTeX format and prepares them for evaluation within the EPDE framework.
                The grid tokens are designed to capture spatial or temporal relationships within the data, enabling the discovery of differential equations that incorporate these relationships.
        
                Args:
                    labels (`list` of `str`): Labels for each dimension of the grid, plus a base label. The length of the list must be equal to dimensionality + 1.
                    max_power (`int`): The maximum power to which a grid token can be raised. Defaults to 1.
                    dimensionality (`int`): The number of dimensions of the grid tokens. Defaults to 1.
        
                Returns:
                    None
        """
        assert len(labels) == dimensionality + 1, 'Incorrect labels for grids.'
        
        self._token_family = TokenFamily(token_type='grids')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True,
                                      meaningful=True, non_default_power = True)

        def latex_form(label, **params):
            '''
            Parameters
            ----------
            label : str
                label of the token, for which we construct the latex form.
            **params : dict
                dictionary with parameter labels as keys and tuple of parameter values 
                and their output text forms as values.

            Returns
            -------
            form : str
                LaTeX-styled text form of token.
            '''
            form = label            
            if params['power'][0] > 1:
                form = r'(' + form + r')^{{{0}}}'.format(params["power"][0])
            return form
        

        self._token_family.set_latex_form_constructor(latex_form)
        grid_token_params = OrderedDict([('power', (1, max_power)), ('dim', (0, dimensionality))])

        grid_equal_params = {'power': 0, 'dim': 0}
        self._token_family.set_params(labels, grid_token_params, grid_equal_params)
        self._token_family.set_evaluator(grid_evaluator)


class LogfunTokens(PreparedTokens):
    """
    Represents a collection of tokens with associated log probabilities.
    
        This class is designed to store and manage a sequence of tokens,
        along with their corresponding log probabilities. It provides
        functionality for accessing and manipulating these tokens and
        probabilities.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the `LogfunTokens` class.
        
        This method is intentionally left unimplemented to enforce its abstract nature.
        Subclasses must provide their own implementation.
        
        Args:
            *args: Variable length argument list.  These arguments would typically
                define the specific tokens and their associated functionalities
                for a particular equation discovery task.
            **kwargs: Arbitrary keyword arguments. These keyword arguments might
                configure the token behavior or influence the equation search process.
        
        Returns:
            None.
        
        Raises:
            NotImplementedError: Always raised to indicate that the base class
                `LogfunTokens` should not be instantiated directly.  This ensures
                that concrete token sets are defined in subclasses, tailoring the
                equation search to specific problem domains.
        """
        raise NotImplementedError


class CustomTokens(PreparedTokens):
    """
    Class for customer tokens
    """

    def __init__(self, token_type: str, token_labels: list,
                 evaluator: Union[CustomEvaluator, EvaluatorTemplate, Callable],
                 params_ranges: dict, params_equality_ranges: dict = None, dimensionality: int = 1,
                 unique_specific_token=True, unique_token_type=True, meaningful=False, non_default_power = False):
        
        """
        Initializes a custom token family for equation discovery.
        
                This method sets up a new token family with specific properties,
                an evaluation function, and parameter ranges. This allows the system
                to explore different equation structures by combining these custom tokens.
                The parameters control the behavior and constraints of the token within
                the evolutionary search process.
        
                Args:
                    token_type (`str`): The unique identifier for this token family.
                    token_labels (`list`): Names of the parameters associated with this token.
                    evaluator (`CustomEvaluator|EvaluatorTemplate|Callable`):
                        The function used to evaluate the token's contribution to the equation.
                    params_ranges (`dict`): Defines the valid ranges for each parameter of the token.
                    params_equality_ranges (`dict`, optional): Acceptable deviations for parameters
                        when considering tokens as equivalent. Defaults to None.
                    dimensionality (`int`): Number of dimensions for token parameters. Defaults to 1.
                    unique_specific_token (`bool`, optional): If True, only one instance of this
                        specific token (with the same parameters) can exist in a term. Defaults to True.
                    unique_token_type (`bool`, optional): If True, only one token from this family
                        can exist in a term. Defaults to True.
                    meaningful (`bool`, optional): A flag indicating whether this token is considered
                        significant in the equation. Defaults to False.
                    non_default_power (`bool`, optional): A flag indicating whether token power is non-default. Defaults to False.
        
                Returns:
                    None
        """
        self._token_family = TokenFamily(token_type=token_type)
        self._token_family.set_status(unique_specific_token=unique_specific_token,
                                      unique_token_type=unique_token_type, 
                                      meaningful=meaningful,
                                      non_default_power = non_default_power)
        default_param_eq_fraction = 0.5
        if params_equality_ranges is not None:
            for param_key, interval in params_ranges.items():
                if param_key not in params_equality_ranges.keys():
                    if isinstance(interval[0], float):
                        params_equality_ranges[param_key] = (interval[1] - interval[0]) * default_param_eq_fraction
                    elif isinstance(interval[0], int):
                        params_equality_ranges[param_key] = 0
        else:
            params_equality_ranges = dict()
            for param_key, interval in params_ranges.items():
                if isinstance(interval[0], float):
                    params_equality_ranges[param_key] = (interval[1] - interval[0]) * default_param_eq_fraction
                elif isinstance(interval[0], int):
                    params_equality_ranges[param_key] = 0

        self._token_family.set_params(token_labels, params_ranges, params_equality_ranges)
        self._token_family.set_evaluator(evaluator)


class CacheStoredTokens(CustomTokens):
    """
    A class to store and manage cached tokens.
    
        This class provides a mechanism to store and retrieve tokens,
        potentially optimizing performance by avoiding redundant computations.
    """

    def __init__(self, token_type: str, token_labels: list, token_tensors: dict, params_ranges: dict,
                 params_equality_ranges: Union[None, dict], dimensionality: int = 1,
                 unique_specific_token=True, unique_token_type=True, meaningful=False,
                 non_default_power = True):
        """
        Initializes a SimpleEquationToken object, preparing it for equation discovery.
        
                This constructor validates the provided token labels and tensors, then stores the tensors
                in a global cache for efficient access during the equation search process. This ensures
                that the building blocks for constructing equations are readily available.
        
                Args:
                    token_type: The type of the token.
                    token_labels: A list of labels for the tokens.
                    token_tensors: A dictionary mapping token labels to their corresponding tensors.
                    params_ranges: A dictionary defining the ranges for the token's parameters.
                    params_equality_ranges: A dictionary defining the equality ranges for the token's parameters, or None.
                    dimensionality: The dimensionality of the token (default: 1).
                    unique_specific_token: A boolean indicating whether the token should be unique (default: True).
                    unique_token_type: A boolean indicating whether the token type should be unique (default: True).
                    meaningful: A boolean indicating whether the token is meaningful (default: False).
                    non_default_power: A boolean indicating whether the token has a non-default power (default: True).
                
                Raises:
                    KeyError: If the labels of the tokens do not match the labels of the passed tensors,
                              ensuring data consistency for equation construction.
                
                Returns:
                    None
                
                Fields:
                    token_type (str): The type of the token.
                    token_labels (list): A list of labels for the tokens.
                    evaluator (function): The evaluator function for the token (set to simple_function_evaluator).
                    params_ranges (dict): A dictionary defining the ranges for the token's parameters.
                    params_equality_ranges (dict): A dictionary defining the equality ranges for the token's parameters, or None.
                    dimensionality (int): The dimensionality of the token.
                    unique_specific_token (bool): A boolean indicating whether the token should be unique.
                    unique_token_type (bool): A boolean indicating whether the token type should be unique.
                    meaningful (bool): A boolean indicating whether the token is meaningful.
                    non_default_power (bool): A boolean indicating whether the token has a non-default power.
        """
        if set(token_labels) != set(list(token_tensors.keys())):
            raise KeyError('The labels of tokens do not match the labels of passed tensors')
        upload_simple_tokens(list(token_tensors.keys()), global_var.tensor_cache, list(token_tensors.values()))
        super().__init__(token_type=token_type, token_labels=token_labels, evaluator=simple_function_evaluator,
                         params_ranges=params_ranges, params_equality_ranges=params_equality_ranges,
                         dimensionality=dimensionality, unique_specific_token=unique_specific_token,
                         unique_token_type=unique_token_type, meaningful=meaningful, non_default_power = non_default_power)


class ExternalDerivativesTokens(CustomTokens):
    """
    Represents tokens derived from external sources.
    
        This class handles tokens that are derivatives obtained from external data,
        managing their names, orders, and tensor representations.
    
        Attributes:
            token_type: The type of the token.
            token_labels: The labels for the tokens (derivative names).
            evaluator: The function used for evaluating the token. Set to `simple_function_evaluator`.
            params_ranges: Ranges for the parameters.
            params_equality_ranges: Ranges for parameter equality.
            dimensionality: The dimensionality of the data.
            unique_specific_token: Whether specific tokens should be unique.
            unique_token_type: Whether token types should be unique.
            meaningful: A flag indicating whether the token is meaningful.
    """

    def __init__(self, token_type: str, time_axis: int, base_token_label: list, token_tensor: np.ndarray,
                 max_orders: Union[int, tuple], deriv_method: str, deriv_method_kwargs: dict, params_ranges: dict,
                 params_equality_ranges: Union[None, dict], dimensionality: int = 1,
                 unique_specific_token=True, unique_token_type=True, meaningful=False):
        """
        Initializes a new instance of the `ExternalDerivativesTokens` class.
        
                This method orchestrates the creation of tokens representing derivatives,
                preparing them for use in the equation discovery process. It preprocesses
                the input tensor to compute derivatives, defines the names and orders
                of these derivatives, arranges the data into a suitable tensor format,
                and uploads these derivative tokens into the system's cache. This ensures
                that the derivative information is readily available for subsequent
                equation learning steps.
        
                Args:
                    token_type (str): The type of the token.
                    time_axis (int): The axis representing time.
                    base_token_label (list): The base label for the token.
                    token_tensor (np.ndarray): The tensor representing the token.
                    max_orders (Union[int, tuple]): The maximum order of derivatives. Can be an integer or a tuple.
                    deriv_method (str): The method used for calculating derivatives.
                    deriv_method_kwargs (dict): Keyword arguments for the derivative method.
                    params_ranges (dict): Ranges for the parameters.
                    params_equality_ranges (Union[None, dict]): Ranges for parameter equality.
                    dimensionality (int): The dimensionality of the data. Defaults to 1.
                    unique_specific_token (bool): Whether specific tokens should be unique. Defaults to True.
                    unique_token_type (bool): Whether token types should be unique. Defaults to True.
                    meaningful (bool): A flag indicating whether the token is meaningful. Defaults to False.
        
                Returns:
                    None
        """
        deriv_method_kwargs['max_order'] = max_orders
        # TODO: rewrite preprocessing of external derivatives to match selected preprocessing pipelines
        data_tensor, derivs_tensor = preprocess_derivatives(token_tensor, method=deriv_method,
                                                          method_kwargs=deriv_method_kwargs)
        deriv_names, deriv_orders = define_derivatives(base_token_label, dimensionality=token_tensor.ndim,
                                                       max_order=max_orders)

        derivs_stacked = prepare_var_tensor(token_tensor, derivs_tensor, time_axis)
        upload_simple_tokens(deriv_names, global_var.tensor_cache, derivs_stacked)

        super().__init__(token_type=token_type, token_labels=deriv_names,
                         evaluator=simple_function_evaluator, params_ranges=params_ranges,
                         params_equality_ranges=params_equality_ranges,
                         dimensionality=dimensionality, unique_specific_token=unique_specific_token,
                         unique_token_type=unique_token_type, meaningful=meaningful)


class ConstantToken(PreparedTokens):
    """
    Variety of tokens for keeping constanting variables
    """

    def __init__(self, values_range=(-np.inf, np.inf)):
        """
        Initializes a constant token, defining its properties within the equation discovery process.
        
                This method sets up the constant token with a specified range of possible values.
                The range is used to constrain the search space during the equation discovery process,
                ensuring that the constant values explored are within reasonable bounds.
                It also configures the token family, sets parameters for evaluation and differentiation,
                and associates the token with its corresponding evaluators.
        
                Args:
                    values_range (`tuple`): optional, default - (-inf, +inf)
                        Interval in which the value of the token can exist.
        
                Returns:
                    None
        """
        assert len(values_range) == 2 and values_range[0] < values_range[1], 'Range of the values has not been stated correctly.'
        self._token_family = TokenFamily(token_type='constants')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True,
                                      meaningful=False)
        const_token_params = OrderedDict([('power', (1, 1)),
                                          ('value', values_range)])
        if - values_range[0] == np.inf or values_range[0] == np.inf:
            value_equal_range = 0.1
        else:
            val_equality_fraction = 0.01
            value_equal_range = (values_range[1] - values_range[0]) / val_equality_fraction
        const_equal_params = {'power': 0, 'value': value_equal_range}
        print('Conducting init procedure for ConstantToken:')
        self._token_family.set_params(['const'], const_token_params, const_equal_params)
        print('Parameters set')
        self._token_family.set_evaluator(const_evaluator)
        print('Evaluator set')
        self._token_family.set_deriv_evaluator({'value': const_grad_evaluator})


class VelocityHEQTokens(PreparedTokens):
    """
    Сustom type of tokens for the equation of thermal conductivity
    """

    def __init__(self, param_ranges):
        """
        Initializes the VelocityAssumptionTokenFamilyGenerator.
        
                This constructor configures the token family for velocity assumption tokens.
                It defines the token's parameters, how it's evaluated, and how its gradient is calculated,
                which is essential for optimizing the equation discovery process. The configuration ensures
                that the generated tokens are suitable for representing velocity-related terms within
                the differential equations being discovered.
        
                Args:
                    param_ranges: A list of 15 tuples, where each tuple represents the
                        range (min, max) for a specific parameter. These ranges constrain
                        the parameter space during the equation search.
        
                Returns:
                    None
        
                Class Fields:
                    _token_family (TokenFamily): A TokenFamily object configured for
                        'velocity_assuption' tokens. It manages the token type,
                        uniqueness constraints, meaningfulness, parameters, evaluators,
                        and derivative evaluators. This family is used to generate
                        and manage tokens representing velocity-related terms in the
                        discovered equations.
        """
        assert len(param_ranges) == 15
        self._token_family = TokenFamily(token_type='velocity_assuption')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True,
                                      meaningful=False)

        opt_params = [('p' + str(idx+1), p_range) for idx, p_range in enumerate(param_ranges)]
        token_params = OrderedDict([('power', (1, 1)),] + opt_params)
        # print(token_params)

        p_equality_fraction = 0.05  # fraction of allowed frequency interval, that is considered as the same
        opt_params_equality = {'p' + str(idx+1): (p_range[1] - p_range[0]) / p_equality_fraction for idx, p_range in enumerate(param_ranges)}
        equal_params = {'power': 0}
        equal_params.update(opt_params_equality)
        self._token_family.set_params(['v'], token_params, equal_params)
        self._token_family.set_evaluator(velocity_evaluator)
        grad_eval_labeled = {'p'+str(idx+1): fun for idx, fun in enumerate(velocity_grad_evaluators)}
        self._token_family.set_deriv_evaluator(grad_eval_labeled, [])
