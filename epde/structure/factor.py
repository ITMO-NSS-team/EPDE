#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:16:43 2020

@author: mike_ubuntu
"""

import numpy as np
import copy
import torch
from typing import Callable
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import epde.globals as global_var
from epde.structure.Tokens import TerminalToken
from epde.supplementary import factor_params_to_str, train_ann, use_ann_to_predict, exp_form
from epde.evaluators import simple_function_evaluator

class EvaluatorContained(object):
    """
    Class for evaluating the values of tokens (factors within terms of a candidate equation). This evaluator supports the use of arbitrary functions in the equation.
    
    
        Attributes:
            _evaluator (`callable`): a function, which returns the vector of token values, evaluated on the studied area;
            params (`dict`): dictionary, containing parameters of the evaluator (like grid, on which the function is evaluated or matrices of pre-calculated function)
    
        Methods:
            set_params(**params)
                set the parameters of the evaluator, using keyword arguments
            apply(token, token_params)
                apply the defined evaluator to evaluate the token with specific parameters
    """


    def __init__(self, eval_function): # , eval_kwargs_keys={}
        self._evaluator = eval_function
        # self.eval_kwargs_keys = eval_kwargs_keys

    def apply(self, token, structural=False, func_args=None, torch_mode=False): # , **kwargs
        """
        Apply the defined evaluator to evaluate the token with specific parameters.

        Args:
        """
        Apply the evaluator to a given symbolic token.
        
                This method serves as an interface to the underlying evaluator, allowing the application of specific functions or operations represented by the token.
        
                Args:
                    token (`epde.main_structures.factor.Factor`): The symbolic representation of the token to be evaluated.
                    structural (`bool`): A flag indicating whether the evaluation is part of a structural analysis. Defaults to False.
                    func_args (`object`): Additional arguments to be passed to the evaluator function. Defaults to None.
                    torch_mode (`bool`): A flag indicating whether to use torch mode. Defaults to False.
        
                Returns:
                    `Any`: The result of applying the evaluator to the token.
        
                Raises:
                    `TypeError`: If the evaluator cannot be applied to the provided token.
        """
            token (`epde.main_structures.factor.Factor`): symbolic label of the specific token, e.g. 'cos';
        token_params (`dict`): dictionary with keys, naming the token parameters (such as frequency, axis and power for trigonometric function) 
            and values - specific values of corresponding parameters.

        Raises:
            `TypeError`
                If the evaluator could not be applied to the token.
        """
        # assert list(kwargs.keys()) == self.eval_kwargs_keys, f'Kwargs {kwargs.keys()} != {self.eval_kwargs_keys}'
        return self._evaluator(token, structural, func_args, torch_mode = torch_mode)


class Factor(TerminalToken):
    """
    Represents a factor in an equation.
    
        The Factor class encapsulates the properties and behavior of a factor within an equation,
        including its label, family type, associated variable, status, and evaluation logic.
    
        Class Methods:
        - __init__
        - variable
        - manual_reconst
        - ann_representation
        - predict_with_ann
        - reset_saved_state
        - status
        - set_parameters
        - __eq__
        - partial_equlaity
        - evaluator
        - evaluate
        - cache_label
        - name
        - latex_name
        - hash_descr
        - grids
        - use_grids_cache
        - __deepcopy__
        - use_cache
    
        Class Attributes:
        - __slots__
    """

    __slots__ = ['_params', '_params_description', '_hash_val', '_latex_constructor', 'label',
                 'ftype', '_variable', '_all_vars', 'grid_set', 'grid_idx', 'is_deriv', 'deriv_code',
                 'cache_linked', '_status', 'equality_ranges', '_evaluator', 'saved']

    def __init__(self, token_name: str, status: dict, family_type: str, latex_constructor: Callable,
                 variable: str = None, all_vars: list = None, randomize: bool = False, 
                 params_description=None, deriv_code=None, equality_ranges = None):
        """
        Initializes a `Factor` object, representing a term within a differential equation.
        
                This constructor sets up the `Factor` with its symbolic representation (`token_name`),
                family type, and any associated variables. It prepares the `Factor` for use in the
                equation discovery process, including potential randomization of parameters and caching
                of computed values to improve efficiency. The initialization also determines if the factor
                represents a derivative, storing the derivative code if applicable.
        
                Args:
                    token_name: The symbolic name of this factor (e.g., 'u', 'du/dx').
                    status: A dictionary containing status information about the factor, such as whether it requires a grid.
                    family_type: The type of family this factor belongs to (e.g., polynomial, trigonometric).
                    latex_constructor: A callable that converts the factor into a LaTeX representation for display.
                    variable: An optional variable associated with the factor (e.g., 'x' in 'du/dx'). Defaults to None.
                    all_vars: An optional list of all variables in the equation. Defaults to None.
                    randomize: A boolean indicating whether to randomize the factor's parameters. Defaults to False.
                    params_description: Description of parameters for randomization, including ranges and distributions.
                    deriv_code: Code for calculating the derivative of this factor, if it represents a derivative.
                    equality_ranges: Ranges for equality constraints on the parameters.
        
                Returns:
                    None
        
                Class Fields:
                    label: The name of the token (str).
                    ftype: The type of family (str).
                    _variable: An optional variable associated with the object (str).
                    _all_vars: An optional list of all variables (list).
                    status: A dictionary containing status information (dict).
                    grid_set: A boolean indicating whether the grid is set (bool). Initialized to False.
                    _hash_val: A random integer used for hashing (int).
                    _latex_constructor: A callable used to construct LaTeX representations (Callable).
                    is_deriv: A boolean indicating whether the object represents a derivative (bool).
                    deriv_code: Code for derivative calculation (any).
                    cache_linked: A boolean indicating whether the cache is linked (bool).
        
                The Factor class represents a building block of a differential equation. The initialization process prepares
                it for symbolic manipulation, numerical evaluation, and integration within the equation discovery workflow.
                Caching and parameter randomization are key aspects of this initialization, enabling efficient exploration
                of the solution space.
        """
        self.label = token_name
        self.ftype = family_type
        self._variable = variable
        self._all_vars = all_vars
        
        self.status = status
        self.grid_set = False
        self._hash_val = np.random.randint(0, 1e9)
        self._latex_constructor = latex_constructor

        self.is_deriv = not (deriv_code is None)
        self.deriv_code = deriv_code

        self.reset_saved_state()
        if global_var.tensor_cache is not None:
            self.use_cache()
        else:
            self.cache_linked = False

        if randomize:
            assert params_description is not None and equality_ranges is not None
            self.set_parameters(params_description,
                                equality_ranges, random=True)

            if self.status['requires_grid']:
                self.use_grids_cache()
    
    @property
    def variable(self):
        """
        Return the `Variable` associated with this `Factor`.
        
                If a specific `Variable` has been assigned, it is returned. Otherwise, the default feature type (`ftype`) is returned. This ensures that a valid feature representation is always available for use in equation construction and evaluation.
        
                Args:
                    self: The `Factor` instance.
        
                Returns:
                    The `Variable` or `ftype` associated with this `Factor`.
        """
        if self._variable is None:
            return self.ftype
        else:
            return self._variable
        
    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        """
        Reconstructs a specific attribute of the `Factor` object, enabling targeted adjustments to the equation discovery process.
        
                This method allows for the manual modification of a `Factor` object's attributes, providing a way to fine-tune the equation search based on prior knowledge or specific requirements. It is particularly useful when certain aspects of a factor need to be adjusted without re-running the entire discovery process.
        
                Args:
                    attribute: The name of the attribute to reconstruct.
                    value: The new value to assign to the attribute.
                    except_attrs: A dictionary of attributes to exclude during the reconstruction process.
        
                Raises:
                    ValueError: If the specified attribute is not supported for manual reconstruction.
        
                Returns:
                    None.
        """
        from epde.loader import obj_to_pickle, attrs_from_dict        
        supported_attrs = []
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')

    @property
    def ann_representation(self) -> torch.nn.modules.container.Sequential:
        """
        Return the ANN representation of the factor.
        
                This representation is crucial for efficiently evaluating the factor
                across different grid configurations. By caching the ANN representation,
                subsequent evaluations become significantly faster.
        
                Args:
                    self: The instance of the Factor class.
        
                Returns:
                    torch.nn.modules.container.Sequential: The ANN representation of the factor.
        """
        try:
            return self._ann_repr
        except AttributeError:
            _, grids = global_var.grid_cache.get_all()
            self._ann_repr = train_ann(grids = grids, data=self.evaluate())
            return self._ann_repr

    def predict_with_ann(self, grids: list):
        """
        Uses the ANN representation, learned during the equation discovery process, to predict values for given grids.
        
                This method leverages the pre-trained ANN model to generate predictions
                based on the provided input grids. It reshapes the grids, converts them
                to tensors, and feeds them into the ANN model. The output is then reshaped
                to match the original grid dimensions. This allows to evaluate discovered equation on unseen data.
        
                Args:
                    grids (list): A list of grids for which predictions are to be made.
        
                Returns:
                    np.ndarray: The predicted values, reshaped to match the dimensions of the input grids.
        """
        return use_ann_to_predict(self.ann_representation, grids)

    def reset_saved_state(self):
        """
        Resets the tracking of saved states.
        
                This method resets the 'saved' dictionary, marking both 'base' and
                'structural' states as unsaved. This ensures that the system accurately
                reflects whether modifications have occurred since the last save,
                allowing the evolutionary process to correctly identify changes and
                potentially trigger necessary actions like re-evaluation or storage.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        self.saved = {'base': False, 'structural': False}

    @property
    def status(self):
        """
        Returns the current status of the symbolic factor.
        
                This status reflects the stage of the factor within the equation discovery process,
                such as whether it's being actively considered, has been selected, or has been discarded.
        
                Returns:
                    str: The current status of the factor.
        """
        return self._status

    @status.setter
    def status(self, status_dict):
        """
        Configures the token's behavior within the equation discovery process.
        
                This method sets the internal status dictionary, which governs how the token interacts with other tokens during equation construction and simplification. This configuration is crucial for guiding the search process towards valid and meaningful equation structures.
        
                Args:
                    status_dict (dict): A dictionary defining the token's constraints and requirements.
                        Keys:
                            'mandatory' (bool): If True, a token from this family must be present in every term.
                            'unique_token_type' (bool): If True, only one token of this type can be present in a term.
                            'unique_specific_token' (bool): If True, this specific token can only appear once per term.
                            'requires_grid' (bool): If True, the token requires grid data for evaluation; otherwise, it will be loaded from cache.
        
                Returns:
                    None
        """
        self._status = status_dict

    def set_parameters(self, params_description: dict, equality_ranges: dict,
                       random=True, **kwargs):
        """
        Sets the parameters for the factor, either randomly within specified bounds or using provided values.
        
                This method initializes the factor's parameters, ensuring they are appropriately set for subsequent calculations and avoids issues with periodic parameters or other constraints.
        
                Args:
                    params_description (dict): A dictionary describing the parameters, including their names and bounds.
                    equality_ranges (dict): A dictionary defining ranges for equality constraints on the parameters.
                    random (bool, optional): Whether to randomly initialize the parameters within the specified bounds. Defaults to True. If False, parameters are taken from kwargs.
                    **kwargs: Keyword arguments providing specific values for the parameters when random is False.
        
                Returns:
                    None
        """
        _params_description = {}
        if not random:
            _params = np.empty(len(kwargs))
            if len(kwargs) != len(params_description):
                print('Not all parameters have been declared. Partial randomization TBD')
                print(f'kwargs {kwargs}, while params_descr {params_description}')
                raise ValueError('...')
            for param_idx, param_info in enumerate(kwargs.items()):
                _params[param_idx] = param_info[1]
                _params_description[param_idx] = {'name': param_info[0],
                                                  'bounds': params_description[param_info[0]]}
        else:
            _params = np.empty(len(params_description))
            for param_idx, param_info in enumerate(params_description.items()):
                if param_info[0] != 'power' or self.status['non_default_power']:
                    _params[param_idx] = (np.random.randint(param_info[1][0], param_info[1][1] + 1) if isinstance(param_info[1][0], int)
                                          else np.random.uniform(param_info[1][0], param_info[1][1])) if param_info[1][1] > param_info[1][0] else param_info[1][0]
                else:
                    _params[param_idx] = 1
                _params_description[param_idx] = {'name': param_info[0],
                                                  'bounds': param_info[1]}
        self.equality_ranges = equality_ranges
        super().__init__(number_params=_params.size, params_description=_params_description,
                         params=_params)
        if not self.grid_set:
            self.use_grids_cache()

    def __eq__(self, other):
        """
        Tests the equality of two factors.
        
        This method determines if two factors are equivalent by comparing their types,
        labels, and parameters. It checks if the parameters of the two factors
        are within the acceptable equality ranges defined for each parameter.
        
        Args:
            other (Factor): The factor to compare with this factor.
        
        Returns:
            bool: True if the factors are considered equal based on their type,
                label, and parameter values; False otherwise.
        
        Why:
            This ensures that factors representing the same underlying equation
            are treated as identical, which is crucial for the evolutionary
            search process to avoid redundant computations and to properly
            evaluate the fitness of different equation candidates.
        """
        if type(self) != type(other):
            return False
        elif self.label != other.label:
            return False
        elif any([abs(self.params[idx] - other.params[idx]) > self.equality_ranges[self.params_description[idx]['name']]
                  for idx in np.arange(self.params.size)]):
            return False
        else:
            return True
        
    def partial_equlaity(self, other):
        """
        Checks for similarity between two factors, focusing on key characteristics relevant to equation discovery.
        
                This method assesses whether two factors are similar enough to be considered equivalent within the context of equation search.
                It compares the 'label' (mathematical operation) and 'params' (parameters), ensuring they align within acceptable tolerances.
                The 'power' parameter is excluded from the comparison as it represents scaling and does not affect equation structure.
        
                Args:
                    other: The factor to compare with.
        
                Returns:
                    bool: True if the factors are similar, False otherwise.
        """
        for param_idx, param_info in self.params_description.items():
            if param_info['name'] == 'power':
                power_idx = param_idx
                break
            
        if type(self) != type(other):
            return False
        elif self.label != other.label:
            return False
        elif any([abs(self.params[idx] - other.params[idx]) > self.equality_ranges[self.params_description[idx]['name']]
                  for idx in np.arange(self.params.size) if idx != power_idx]):
            return False
        else:
            return True

    @property
    def evaluator(self):
        """
        Returns the evaluator object.
        This object is responsible for assessing the quality of the generated equations.
        
        Args:
            None
        
        Returns:
            The evaluator object used to score equation fitness.
        """
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator):
        """
        Sets the evaluator for the factor.
        
                The evaluator is responsible for calculating the loss and gradients
                associated with the factor during the equation discovery process.
                If the provided evaluator is self-contained (EvaluatorContained), it is used directly.
                Otherwise, the appropriate evaluator is extracted from the factor families
                within the provided evaluator, ensuring compatibility with the factor's type.
        
                Args:
                    evaluator: The evaluator to set. It can be either an EvaluatorContained instance
                        or an evaluator containing factor families.
        
                Returns:
                    None.
        
                Class Fields:
                    _evaluator (EvaluatorBase): The evaluator for the factor.
        """
        if isinstance(evaluator, EvaluatorContained):
            self._evaluator = evaluator
        else:
            factor_family = [family for family in evaluator.families if family.ftype == self.ftype][0]
            self._evaluator = factor_family._evaluator # TODO: fix calling private attribute
            
    def evaluate(self, structural=False, grids=None, torch_mode: bool = False):
        """
        Evaluates the token's value, leveraging cached results when available or applying the appropriate evaluator function. This process is central to constructing equation candidates and assessing their fitness against the observed data.
        
                Args:
                    structural (bool): A flag indicating whether to evaluate the structural form of the token. Structural evaluation might involve different simplifications or representations.
                    grids (optional): Grids to use for evaluation, typically representing the domain over which the equation is defined. Defaults to None.
                    torch_mode (bool): A flag indicating whether to use torch tensors for evaluation.
        
                Returns:
                    The evaluated value of the token, which can be a numerical value, a symbolic expression, or a tensor, depending on the evaluation context.
        """
        assert self.cache_linked, 'Missing linked cache.'
        if self.is_deriv and grids is not None:
            raise Exception(
                'Derivatives have to evaluated on the initial grid')

        key = 'structural' if structural else 'base'
        if (self.cache_label, structural) in global_var.tensor_cache and grids is None:
            # print(f'Asking for {self.cache_label} in tmode {torch_mode}')
            # print(f'From numpy cache of {global_var.tensor_cache.memory_structural["numpy"].keys()}')
            # print(f'And torch cache of {global_var.tensor_cache.memory_structural["torch"].keys()}')

            return global_var.tensor_cache.get(self.cache_label,
                                               structural=structural, torch_mode = torch_mode)
 
        else:
            if self.is_deriv and self.evaluator._evaluator != simple_function_evaluator:
                if grids is not None:
                    raise Exception('Data-reliant tokens shall not get grids as arguments for evaluation.')
                if isinstance(self.variable, str):
                    var = self._all_vars.index(self.variable)
                    func_arg = [global_var.tensor_cache.get(label=None, torch_mode=torch_mode,
                                                            deriv_code=(var, self.deriv_code)),]
                elif isinstance(self.variable, (list, tuple)):
                    func_arg = []
                    for var_idx, code in enumerate(self.deriv_code):
                        assert len(self.variable) == len(self.deriv_code)
                        func_arg.append(global_var.tensor_cache.get(label=None, torch_mode=torch_mode,
                                                                    deriv_code=(self.variable[var_idx], code)))

                value = self.evaluator.apply(self, structural=structural, func_args=func_arg, torch_mode=torch_mode)
            else:
                value = self.evaluator.apply(self, structural=structural, func_args=grids, torch_mode=torch_mode)
            if grids is None:
                if self.is_deriv and self.evaluator._evaluator == simple_function_evaluator:
                    full_deriv_code = (self._all_vars.index(self.variable), self.deriv_code)
                else:
                    full_deriv_code = None      

                if key == 'structural' and self.status['structural_and_defalut_merged']:
                    self.saved[key] = global_var.tensor_cache.add(self.cache_label, value, structural=False, 
                                                                  deriv_code=full_deriv_code)                    
                    global_var.tensor_cache.use_structural(use_base_data=True,
                                                           label=self.cache_label)
                elif key == 'structural' and not self.status['structural_and_defalut_merged']:
                    global_var.tensor_cache.use_structural(use_base_data=False,
                                                           label=self.cache_label,
                                                           replacing_data=value)
                else:
                    self.saved[key] = global_var.tensor_cache.add(self.cache_label, value, structural=False, 
                                                                  deriv_code=full_deriv_code)
            return value

    @property
    def cache_label(self):
        """
        Caches a unique string representation of the factor's parameters.
        
                This representation is used to efficiently identify and reuse previously computed results for factors with the same parameter values. This avoids redundant calculations when exploring the search space of possible equation structures.
        
                Args:
                    self: The factor object.
        
                Returns:
                    str: A string that uniquely identifies the factor based on its label and parameters.
        """
        cache_label = factor_params_to_str(self)
        return cache_label

    @property
    def name(self):
        """
        Generates a string representation of the factor, including its label and parameter values.
        
                This provides a human-readable identifier for the factor, useful for tracking its configuration during the equation discovery process. The string includes the factor's label and a dictionary-like representation of its parameters.
        
                Args:
                    self: The object instance.
        
                Returns:
                    str: A formatted string containing the label and parameter values.
        """
        form = self.label + '{'
        for param_idx, param_info in self.params_description.items():
            form += param_info['name'] + ': ' + str(self.params[param_idx])
            if param_idx < len(self.params_description.items()) - 1:
                form += ', '
        form += '}'
        return form

    @property
    def latex_name(self):
        """
        Returns the LaTeX representation of the factor.
        
                This representation is crucial for displaying the discovered equation in a human-readable format.
                If a LaTeX constructor is defined, it uses the constructor and parameter descriptions to generate the LaTeX string,
                formatting the parameters with appropriate scientific notation.
                Otherwise, it returns the factor's name.
        
                Args:
                    self: The Factor instance.
        
                Returns:
                    str: The LaTeX representation of the factor.
        """
        if self._latex_constructor is not None:
            params_dict = {}
            for param_idx, param_info in self.params_description.items():
                mnt, exp = exp_form(self.params[param_idx], 3)
                exp_str = r'\cdot 10^{{{0}}} '.format(str(exp)) if exp != 0 else ''

                params_dict[param_info['name']] = (self.params[param_idx], str(mnt) + exp_str)
            return self._latex_constructor(self.label, **params_dict)
        else:
            return self.name # other implementations are possible
    
    @property
    def hash_descr(self) -> int:
        """
        Returns the precomputed hash value of the factor.
        
        This method returns the precomputed hash value, which is calculated
        during object initialization. This hash is used for efficient comparison
        and caching of factors within the equation discovery process. By precomputing
        the hash, we avoid redundant calculations when evaluating the fitness
        of different equation candidates.
        
        Args:
            self: The object instance.
        
        Returns:
            int: The precomputed hash value of the factor.
        """
        return self._hash_val

    @property
    def grids(self):
        """
        Retrieves all grid configurations currently stored.
        
                This provides access to the explored search space,
                allowing inspection of different equation structures
                evaluated during the discovery process.
        
                Returns:
                    A list of grid configurations representing the search space.
        """
        _, grids = global_var.grid_cache.get_all()
        return grids

    def use_grids_cache(self):
        """
        Uses a precomputed grid based on the 'dim' parameter to speed up calculations.
        
                This method searches for a 'dim' parameter within the defined parameters. If found, it uses the corresponding value to select a pre-calculated grid, avoiding redundant grid generation. If 'dim' is absent, a default grid (index 0) is used. This optimization is crucial for efficient equation discovery.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        
                Class Fields Initialized:
                    grid_idx (int): The index of the grid to use, derived from the 'dim' parameter or 0 if 'dim' is not found.
                    grid_set (bool): A flag indicating whether the grid index has been set (True after this method is called).
        """
        dim_param_idx = np.inf
        dim_set = False
        for param_idx, param_descr in self.params_description.items():
            if param_descr['name'] == 'dim':
                dim_param_idx = param_idx
                dim_set = True
        self.grid_idx = int(self.params[dim_param_idx]) if dim_set else 0
        self.grid_set = True

    def __deepcopy__(self, memo=None):
        """
        Creates a deep copy of the Factor object.
        
                This method ensures that when a Factor object is copied, all its attributes,
                including those defined in `__slots__`, are also copied independently. This is
                crucial for maintaining the integrity of equation structures during evolutionary
                operations, preventing unintended modifications to shared components.
        
                Args:
                    memo (dict, optional): A dictionary used by `copy.deepcopy` to keep track of
                        objects that have already been copied, preventing infinite recursion
                        when dealing with circular references. Defaults to None.
        
                Returns:
                    Factor: A new Factor object that is a deep copy of the original.
        """
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct

        new_struct.__dict__.update(self.__dict__)

        attrs_to_avoid_copy = []
        for k in self.__slots__:
            try:
                if k not in attrs_to_avoid_copy:
                    if not isinstance(k, list):
                        setattr(new_struct, k, copy.deepcopy(
                            getattr(self, k), memo))
                    else:
                        temp = []
                        for elem in getattr(self, k):
                            temp.append(copy.deepcopy(elem, memo))
                        setattr(new_struct, k, temp)
                else:
                    setattr(new_struct, k, None)
            except AttributeError:
                pass

        return new_struct

    def use_cache(self):
        """
        Enables the caching mechanism for storing and reusing previously computed factor values.
        
        This optimization can significantly reduce computational cost when the same factor is encountered multiple times during the equation discovery process.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Class Fields:
            cache_linked (bool): A boolean indicating whether the cache is linked.
                Initialized to True when this method is called.
        """
        self.cache_linked = True
