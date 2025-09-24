#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:39:18 2020

@author: mike_ubuntu
"""

import numpy as np
import itertools
from typing import Union, Callable, List
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import epde.globals as global_var
from epde.structure.factor import Factor, EvaluatorContained


def constancy_hard_equality(tensor, epsilon=1e-7):
    """
    Checks if a tensor can be considered constant based on a hard equality threshold.
    
    This method verifies if the difference between the maximum and minimum values
    in the tensor is less than a predefined epsilon. It helps to identify
    tensors that exhibit near-constant behavior, which is useful for
    simplifying equation discovery by identifying terms that might be
    effectively constant within the given data.
    
    Args:
        tensor (np.ndarray): The input tensor to be checked.
        epsilon (float): The tolerance value for determining near-equality.
    
    Returns:
        bool: True if the tensor is approximately constant, False otherwise.
    """
    return np.abs(np.max(tensor) - np.min(tensor)) < epsilon


class TokenFamily(object):
    """
    Represents a category of tokens used as building blocks for constructing equation terms.
    
    
        Attributes:
            _deriv_evaluators (`dict`): dict containing the derivatives by each of the token parameter, where elements are the functions, used in the evaluator, or the evaluators.
            ftype (`string`): the symbolic name of the token family (e.g. 'logarithmic', 'trigonometric', etc.)
            status (`dict`): dictionary, containing markers, describing the token properties. Key - property, value - bool variable:
                'mandatory' - if True, a token from the family must be present in every term;
                'unique_token_type' - if True, only one token of the family can be present in the term;
                'unique_specific_token' - if True, a specific token can be present only once per term;
            family_of_derivs (`bool`): flag about the presence of derivatives in the token family
            evaluator_set (`bool`): flag about the existence of a method for evaluation
            params_set (`bool`): flag, that exist params in that token fami;y
            cache_set (`bool`): flag, that exist cache for the token family
            deriv_exaluator_set (`bool`): flag about the existing a evaluator for derivatives
            _evaluator (`EvaluatorContained object`): Evaluator, which is used to get values of the tokens from that family;
            tokens (`list of strings`): List of function names, describing all of the functions, belonging to the family. E.g. for 'trigonometric' token type, this list will be ['sin', 'cos']
            token_params (`OrderedDict`): Available range for token parameters. Ordered dictionary with key - token parameter name, and value - tuple with 2 elements:
                (lower boundary, higher boundary), while type of boundaries describes the avalable token params: 
                if int - the parameters will be integer, if float - float.
            equality_ranges (`dict`): error for equality of token parameters, key is name of parameter
            derivs_ords (`dict`):  keys for derivatides for `solver` for each token in family
            opt_param_labels (`list`): name of parameters of tokens, that used in elements functions 
            test_token ():
            test_evaluation ():
    
        Methods:
            set_status(demands_equation = False, meaningful = False, 
                    s_and_d_merged = True, unique_specific_token = False, 
                    unique_token_type = False, requires_grid = False))
                Method to set the markers of the token status;
    
            set_params(tokens, token_params)
                Method to set the list of tokens, present in the family, and their parameters;
    
            set_evaluator(eval_function)
                Method to set the evaluator for the token family & its parameters;
    
            test_evaluator()
                Method to test, if the evaluator and tokens are set properly
    
            evaluate(token, token_params)
                Method, which uses the specific token evaluator to evaluate the passed token with its parameters
    """


    def __init__(self, token_type: str, variable:str = None, family_of_derivs: bool = False):
        """
        Initialize the token family.
        
        This initialization is crucial for defining the basic characteristics of a token family, 
        such as its type and whether it includes derivatives. This information is used later 
        to construct valid equation structures and evaluate their fitness against the data.
        
        Args:
            token_type (`str`): The name of the token family; must be unique among other families.
            variable (`str`, optional): variable name for token. Defaults to None.
            family_of_derivs (`bool`, optional): A flag indicating the existence of derivatives within this token family. Defaults to False.
        
        Returns:
            None
        """

        self.ftype = token_type
        self.variable = variable
        
        self.family_of_derivs = family_of_derivs
        self.evaluator_set = False
        self.params_set = False
        self.cache_set = False
        self.deriv_evaluator_set = True
        self.set_latex_form_constructor()

    def __len__(self):
        """
        Returns the number of tokens forming the equation family.
        
        This is essential for assessing the complexity and size of the equation family 
        during the evolutionary search for the best equation.
        
        Args:
            self: The Family instance.
        
        Returns:
            int: The number of tokens in the family.
        """
        assert self.params_set, 'Familiy is not fully initialized.'
        return len(self.tokens)

    def set_status(self, demands_equation=False, meaningful=False,
                   s_and_d_merged=True, unique_specific_token=False,
                   unique_token_type=False, requires_grid=False, non_default_power=False):
        """
        Sets the constraints and requirements for tokens within this family, influencing how they are used in equation discovery.
        
                Args:
                    demands_equation (`bool`, optional): Indicates if the presence of this token family imposes constraints on the equation structure. Defaults to `False`.
                    meaningful (`bool`, optional): If `True`, ensures that at least one token from this family is present in each term of the equation. Defaults to `False`.
                    unique_token_type (`bool`, optional): If `True`, allows only one token of this family to be present in each term. Defaults to `False`.
                    unique_specific_token (`bool`, optional): If `True`, restricts a specific token from this family to appear only once per term. Defaults to `False`.
                    s_and_d_merged (`bool`, optional): Determines whether the default values of the token are used directly as structural values. Defaults to `True`.
                    requires_grid (`bool`, optional): Specifies if a spatial grid is necessary for evaluating tokens from this family. Defaults to `False`.
                    non_default_power (`bool`, optional): Indicates whether power values other than 1 are treated as distinct tokens during initialization. Defaults to `False`.
        """
        self.status = {}
        self.status['demands_equation'] = demands_equation
        self.status['meaningful'] = meaningful
        self.status['structural_and_defalut_merged'] = s_and_d_merged
        self.status['unique_specific_token'] = unique_specific_token
        self.status['unique_token_type'] = unique_token_type
        self.status['requires_grid'] = requires_grid
        self.status['non_default_power'] = non_default_power

    def set_params(self, tokens, token_params, equality_ranges, derivs_solver_orders=None):
        """
        Define the token family, its parameters, and derivative orders if applicable.
        
                This method configures the token family with essential information for equation discovery. 
                It specifies the tokens (functions) belonging to the family, their parameter ranges, and, 
                if the family represents derivatives, the corresponding solver orders. This setup is crucial 
                for generating and evaluating candidate equations within the search space.
        
                Args:
                    tokens (`list of strings`): List of function names, describing all of the functions, belonging to the family. 
                        E.g., for a 'trigonometric' token type, this list will be ['sin', 'cos'].
                    token_params (`OrderedDict`): Available range for token parameters. Ordered dictionary with key - token parameter name, 
                        and value - tuple with 2 elements: (lower boundary, higher boundary). The type of boundaries (int or float) 
                        determines the parameter type.
                    equality_ranges (`dict`): Error tolerance for equality of token parameters; key is the name of the parameter.
                    derivs_solver_orders (`list`): Keys for derivatives in `int` format for the solver. Required only if the token family 
                        represents derivatives.
        
                Returns:
                    None
        """

        assert bool(derivs_solver_orders is not None) == bool(
            self.family_of_derivs), 'Solver form must be set for derivatives, and only for them.'

        self.tokens = tokens
        self.token_params = token_params
        if self.family_of_derivs:
            self.derivs_ords = {token: derivs_solver_orders[idx] for idx, token in enumerate(tokens)}
        self.params_set = True
        self.equality_ranges = equality_ranges

        # if self.family_of_derivs:
        #     print(f'self.tokens is {self.tokens}')
        #     print(f'Here, derivs order is {self.derivs_ords}')
        if self.evaluator_set:
            self.test_evaluator()

    def set_evaluator(self, eval_function, suppress_eval_test=True):
        """
        Define how tokens within this family are evaluated.
        
                This method configures the evaluation process for tokens belonging to this family. It's essential for translating symbolic tokens into numerical values, enabling the equation discovery process. By setting the evaluator, the framework knows how to interpret and quantify the contribution of each token within a candidate equation.
        
                Args:
                    eval_function (`function or EvaluatorContained object`): Function, used in the evaluator, or the evaluator
                    suppress_eval_test (`boolean`): if True, run `test_evaluator` for testing of method for evaluating token
        """
        if isinstance(eval_function, EvaluatorContained):
            self._evaluator = eval_function
        else:
            self._evaluator = EvaluatorContained(eval_function)
        self.evaluator_set = True
        if self.params_set and not suppress_eval_test:
            self.test_evaluator()

    def set_deriv_evaluator(self, eval_functions, suppress_eval_test=True): # eval_kwargs_keys=[], 
        """
        Define the evaluator for the derivatives of the token family and its parameters

        Args:
        """
        Define the evaluator for the derivatives of the token family with respect to its parameters.
        
                This method configures how the derivatives of the token family are calculated, which is crucial for refining the equation discovery process by understanding the sensitivity of the equation to changes in its parameters.
        
                Args:
                    eval_functions (`dict|EvaluatorContained`): A dictionary where keys are parameter names and values are either functions or `EvaluatorContained` instances that compute the derivatives with respect to those parameters.
                    suppress_eval_test (`boolean`): If `True`, the `test_evaluator` method is skipped after setting the evaluator.  This is useful to suppress tests during initial setup or when testing is not immediately required.
        
                Returns:
                    None
        """
            eval_functions (`dict|EvaluatorContained`): Dict containing the derivatives by each of the token parameter, where elements are the functions, used in the evaluator, or the evaluators.
                Keys represent the parameter name, and values are the corresponding functions.
            eval_kwargs_keys (`list`): The parameters for evaluator; must contain params_names (names of the token parameters) &
                param_equality (for each of the token parameters, range in which it considered as the same)
            suppress_eval_test (`boolean`): if True, run `test_evaluator` for testing of method for evaluating token
        """
        self._deriv_evaluators = {}
        for param_key, eval_function in eval_functions.items():
            if isinstance(eval_function, EvaluatorContained):
                _deriv_evaluator = eval_function
            else:
                # print('Setting evaluator kwargs:', eval_kwargs_keys)
                _deriv_evaluator = EvaluatorContained(eval_function) # , eval_kwargs_keys
            self._deriv_evaluators[param_key] = _deriv_evaluator
        self.opt_param_labels = list(eval_functions.keys())
        self.deriv_evaluator_set = True
        if self.params_set and not suppress_eval_test:
            self.test_evaluator(deriv=True)

    def set_latex_form_constructor(self, latex_constructor: Callable = None):
        """
        Sets the method to be used for constructing the LaTeX representation of the token family.
        
        This allows customization of how token families are displayed in a human-readable format,
        which is particularly useful when exploring and presenting discovered equation structures.
        
        Args:
            latex_constructor (Callable, optional): A callable that, when called, returns the LaTeX representation of the object. Defaults to None.
        
        Returns:
            None.
        """
        self.latex_constructor = latex_constructor

    def test_evaluator(self, deriv=False):
        """
        Tests the evaluator's functionality with a newly created token.
        
        This method creates a test token, disables scaling, and evaluates it using the configured evaluator.
        It's crucial for verifying that the token set and evaluator are correctly set up to produce valid results
        before running the equation discovery process. The test ensures that the evaluator can process tokens
        generated by the family without errors.
        
        Args:
            deriv (bool, optional): If True, tests the derivative evaluators instead of the main evaluator. Defaults to False.
        
        Returns:
            None
        
        Raises:
            Exception: If the evaluator fails to process the test token, indicating a problem with the evaluator or token configuration.
        """
        _, self.test_token = self.create()
        self.test_token.use_cache()
        if self.status['requires_grid']:
            self.test_token.use_grids_cache()
        print(self.test_token.grid_idx, self.test_token.params)
        self.test_token.scaled = False
        if deriv:
            for _deriv_evaluator in self._deriv_evaluators.values():
                self.test_evaluation = _deriv_evaluator.apply(self.test_token)
        else:
            # print('Test in the evaluator:', self._evaluator.eval_kwargs_keys)
            self.test_evaluation = self._evaluator.apply(self.test_token)
        print('Test evaluation performed correctly')

    def chech_constancy(self, **tfkwargs):
        """
        Method to identify and remove constant tokens within the defined domain.
        
                This method iterates through the tokens, evaluates their constancy based on cached data,
                and removes those identified as constant from both the token list and the cache.
                This ensures that the equation search focuses on non-constant tokens, improving efficiency
                and accuracy in discovering the underlying differential equation.
        
                Args:
                    tfkwargs (`dict`): Additional keyword arguments for the test function (currently unused).
        
                Returns:
                    None
        """
        assert self.params_set
        constant_tokens_labels = []
        for label in self.tokens:
            data_label = (label, (1.0,))
            data = global_var.tensor_cache.memory_default["numpy"].get(data_label)
            try:
                constancy = np.isclose(np.min(data), np.max(data))
            except TypeError:
                print(f"No {label} data in cache!")
                continue
            if constancy:
                constant_tokens_labels.append(label)

        for label in constant_tokens_labels:
            print(f'Function {label} is assumed to be constant in the studied domain. \
                          Removed from the equaton search.')
            data_label = (label, (1.0,))
            self.tokens.remove(label)
            global_var.tensor_cache.delete_entry(data_label)

    def evaluate(self, token):
        """
        Evaluate the token using a predefined evaluator.
        
        This method attempts to apply a pre-configured evaluator to the given token.
        The evaluation process is crucial for assessing the token's contribution to the overall equation discovery process.
        It ensures that each token is properly assessed for its relevance and impact on the equation's ability to fit the observed data.
        
        Args:
            token: The token to be evaluated.
        
        Returns:
            The result of the evaluator applied to the token.
        
        Raises:
            NotImplementedError: If the method is called directly from the TokenFamily class.
            TypeError: If the evaluator function or its parameters are not set before application.
        """
        raise NotImplementedError('Method has been moved to the Factor class.')
        if self.evaluator_set:
            return self._evaluator.apply(token)
        else:
            raise TypeError(
                'Evaluator function or its parameters not set before evaluator application.')

    def create(self, label=None, token_status: dict = None, all_vars: List[str] = None,
               create_derivs: bool = False, **factor_params):
        """
        Creates a factor representing a specific element (token) within this token family.
        
                This method is responsible for generating a new factor, which is a building block
                in the equation discovery process. The created factor is based on the specified
                label (token name) and incorporates derivative information if needed. The method
                also handles the allocation of resources (tokens) based on the family's constraints
                and sets the factor's parameters, either randomly or based on provided values.
                This ensures that the generated factors adhere to the defined structure and
                constraints of the token family, contributing to the construction of valid
                equation candidates.
        
                Args:
                    label (`str`): The name of the token to create. If `None`, a token is randomly selected
                        from the available tokens in this family, subject to usage constraints.
                    token_status (`dict`): A dictionary containing information about the usage of each token
                        in this family. This is used to track how many times each token has been used,
                        its maximum allowed usage, and whether it is allowed to be used. This argument
                        is ignored if `label` is not `None`. Example: `{'token1': (2, 5, True), 'token2': (0, 3, False)}`.
                    all_vars (`List[str]`): List of all variables.
                    create_derivs (`bool`): A flag indicating whether to create a derivative factor. If `True`,
                        the selected token must have derivative information associated with it. Defaults to `False`.
                    **factor_params: Keyword arguments that will be passed as parameters to created Factor.
        
                Returns:
                    `tuple`: A tuple containing:
                        - `dict`: A dictionary representing the tokens blocked after creating the factor.
                            This indicates which tokens are now unavailable for further use due to constraints.
                        - `Factor`: The newly created factor object, representing the selected token and its
                            associated parameters and derivative information.
        """
        if token_status is None or token_status == {}:
            token_status = {label: (0, self.token_params['power'][1], False)
                            for label in self.tokens}
        if label is None:
            try:
                if create_derivs:
                    label = np.random.choice([token for token in self.tokens
                                              if (not token_status[token][0] + 1 > token_status[token][1]
                                                  and self.derivs_ords[token][0] is not None)])
                else:
                    label = np.random.choice([token for token in self.tokens
                                              if not token_status[token][0] + 1 > token_status[token][1]])
            except ValueError:
                raise ValueError("'a' cannot be empty unless no samples are taken")

        if self.family_of_derivs:
            factor_deriv_code = self.derivs_ords[label]
        else:
            factor_deriv_code = None
        new_factor = Factor(token_name=label, deriv_code=factor_deriv_code, status=self.status,
                            family_type=self.ftype, variable = self.variable, all_vars = all_vars,
                            latex_constructor = self.latex_constructor)

        if self.status['unique_token_type']:
            occupied_by_factor = {token: self.token_params['power'][1] for token in self.tokens}
        elif self.status['unique_specific_token']:
            occupied_by_factor = {label: self.token_params['power'][1]}
        else:
            occupied_by_factor = {label: 1}
        if len(factor_params) == 0:
            new_factor.set_parameters(params_description=self.token_params,
                                      equality_ranges=self.equality_ranges,
                                      random=True)
        else:
            new_factor.set_parameters(params_description=self.token_params,
                                      equality_ranges=self.equality_ranges,
                                      random=False,
                                      **factor_params)
        new_factor.evaluator = self._evaluator

        return occupied_by_factor, new_factor

    def cardinality(self, token_status: Union[dict, None] = None):
        """
        Calculates the number of tokens in the family that can still accommodate new factors,
        considering their current usage.
        
        Args:
           token_status (`dict`, optional): A dictionary containing usage information for each token in the family.
                The dictionary should map token labels to a tuple containing:
                - The number of factors currently using the token.
                - The maximum number of factors allowed to use the token.
                - A flag indicating whether the token is permitted for use.
                If `None` or empty, it's assumed that no tokens are currently in use. Defaults to `None`.
        
        Returns:
            `int`: The number of tokens in the family that have available capacity for new factors.
        
        Why:
            This method determines the remaining capacity within the token family, guiding the evolutionary process
            to efficiently explore equation structures by considering token availability.
        """
        if token_status is None or token_status == {}:
            token_status = {label: (0, self.token_params['power'][1], False)
                            for label in self.tokens}
        return len([token for token in self.tokens if token_status[token][0] < token_status[token][1]])

    def evaluate_all(self, all_vars: List[str]):
        """
        Applies the evaluation method to all tokens within this family, exploring different parameter combinations to identify optimal configurations. This ensures comprehensive coverage of the token family's parameter space, facilitating the discovery of equation structures that accurately represent the underlying dynamics.
        
                Args:
                    all_vars (List[str]): A list of all variable names used in the equation.
        
                Returns:
                    None
        """
        for token_label in self.tokens:
            params_vals = []
            for param_label, param_range in self.token_params.items():
                if param_label != 'power' and isinstance(param_range[0], int):
                    params_vals.append(np.arange(param_range[0], param_range[1] + 1))
                elif param_label == 'power':
                    params_vals.append([1,])
                else:
                    params_vals.append(np.random.uniform(param_range[0], param_range[1]))
            params_sets = list(itertools.product(*params_vals))
            for params_selection in params_sets:
                params_sets_labeled = dict(zip(list(self.token_params.keys()), params_selection))

                _, generated_token = self.create(token_label, all_vars=all_vars, **params_sets_labeled)
                generated_token.use_cache()
                if self.status['requires_grid']:
                    generated_token.use_grids_cache()
                generated_token.scaled = False
                _ = generated_token.evaluate()
                print(generated_token.cache_label)
                if generated_token.cache_label not in global_var.tensor_cache.memory_default['numpy'].keys():
                    raise KeyError('Generated token somehow was not stored in cache.')


class TFPool(object):
    """
    Manages a collection of token families. This class provides functionality for storing, accessing, and manipulating these families, facilitating the construction and optimization of equation structures.
    
    
    Args:
       families (`list`): toen families that using in that run
    """
    
    def __init__(self, families: list):
        """
        Initializes a TFPool instance with a list of symbolic equation families.
        
        This initialization prepares the pool to manage and evolve families of equations,
        facilitating the search for the best equation structures that fit the data.
        The pool serves as a central point for multi-objective optimization.
        
        Args:
            families: A list of family objects, each representing a set of candidate equations.
        
        Returns:
            None
        
        Why:
            The TFPool needs to be initialized with equation families to start the evolutionary process
            of discovering differential equations from data. These families represent the initial
            population of candidate solutions that will be evolved and optimized.
        """
        self.families = families
    
    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        """
        Reconstructs a specific object attribute manually.
        
        This method allows for the targeted modification of an object's internal state,
        bypassing the typical reconstruction process. It is used to fine-tune the object's
        configuration, especially when dealing with attributes that are not automatically
        handled during standard loading or initialization procedures. This is particularly
        useful for attributes that influence the equation discovery process.
        
        Args:
            attribute (str): The name of the attribute to reconstruct.
            value: The new value to assign to the attribute.
            except_attrs (dict): A dictionary of attributes to exclude during the reconstruction process.
        
        Raises:
            ValueError: If the specified attribute is not supported.
        
        Returns:
            None.
        """
        from epde.loader import obj_to_pickle, attrs_from_dict        
        supported_attrs = []
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')
    
    @property
    def families_meaningful(self):
        """
        Filters and retrieves token families marked as meaningful.
        
        This method iterates through the available token families and returns only those that have been 
        designated as 'meaningful' based on their status. This selection is crucial for focusing on the 
        most relevant equation structures during the equation discovery process, improving efficiency and accuracy.
        
        Args:
            None
        
        Returns:
            list: A list of token families where each family's status indicates it is meaningful.
        """
        return [family for family in self.families if family.status['meaningful']]

    @property
    def families_demand_equation(self):
        """
        Return:
                list: A list of token families that require individual equation discovery.
        
                This property identifies token families within the pool that are marked as requiring a dedicated equation.
                These families are processed separately during the equation discovery phase to ensure accurate and specific modeling.
                This is needed to allow the framework to discover more complex relationships between different token families.
                Args:
                    None
        """
        return [family for family in self.families if family.status['demands_equation']]

    @property
    def families_supplementary(self):
        """
        Returns a list of token families that have been identified as not contributing significantly to the equation discovery process. These families are excluded from consideration during the search for governing differential equations to improve efficiency and focus on more relevant terms.
        
                Args:
                    None
        
                Returns:
                    list: A list of `TokenFamily` objects that are considered supplementary.
        """
        return [family for family in self.families if not family.status['meaningful']]

    @property
    def families_equationless(self):
        """
        Gets token families that are not required to be present in the discovered equations.
        
                Args:
                    None
        
                Returns:
                    list: A list of token families. These families represent optional components in the equation discovery process, allowing the algorithm to explore simpler models where these terms might not be necessary to accurately represent the data. This helps to avoid overfitting and identify the most parsimonious equation structure.
        """
        return [family for family in self.families if not family.status['demands_equation']]

    @property
    def labels_overview(self):
        """
        Provides an overview of the token families, indicating the maximum allowed usage for each token within its family.
        
                This information is crucial for configuring the search space of the evolutionary algorithm, ensuring that the generated equation structures adhere to predefined constraints on token usage.
        
                Returns:
                    list: A list of tuples, where each tuple contains the token family name and the maximum allowed power for tokens in that family.
        """
        overview = []
        for family in self.families:
            overview.append((family.tokens, family.token_params['power'][1]))
        return overview

    def families_cardinality(self, meaningful_only: bool = False,
                             token_status: Union[dict, None] = None):
        """
        Calculates the available capacity for adding new factors within each token family.
        
                This method determines how many more factors can be accommodated in each family,
                considering either all families or only the meaningful ones. This information
                is crucial for the evolutionary process, guiding the creation of new equation candidates
                by ensuring that the token families are not over-saturated.
        
                Args:
                    meaningful_only (`bool`): If True, considers only the meaningful families.
                    token_status (`dict`, optional): A dictionary containing usage information for all tokens
                        belonging to the families. Defaults to None.
        
                Returns:
                    `numpy.ndarray`: An array of integers, where each element represents the number of
                    available slots for new factors in the corresponding family.
        """
        if meaningful_only:
            return np.array([family.cardinality(token_status) for family in self.families_meaningful])
        else:
            return np.array([family.cardinality(token_status) for family in self.families])

    def create(self, label=None, create_meaningful: bool = False, token_status=None,
               create_derivs: bool = False, **kwargs) -> Union[str, Factor]:
        """
        Creates a token (either a variable or a constant) by sampling from available token families.
        
        This method facilitates the generation of equation components during the evolutionary search process.
        It selects a token family based on predefined probabilities and then delegates the actual token creation
        to the chosen family. This ensures diversity in the generated equations while adhering to the constraints
        defined by the token families.
        
        Args:
            label (str, optional): If provided, creates a token with the specified name. Defaults to None.
            create_meaningful (bool, optional): If True, selects a token from the 'meaningful' families.
                Defaults to False. Meaningful families typically represent variables that directly contribute
                to the equation's structure.
            token_status (dict, optional): A dictionary containing information about the status of each token family.
                This can be used to influence the selection probabilities. Defaults to None.
            create_derivs (bool, optional): A flag indicating whether derivative tokens should be created.
                Defaults to False.
            **kwargs: Additional keyword arguments that are passed to the `create` method of the selected token family.
        
        Returns:
            Union[str, Factor]: The created token, which can be either a string (if a simple token is created)
            or a Factor object (representing a more complex expression).
        
        Raises:
            ValueError: If `create_meaningful` is True and there are no meaningful families available.
            Exception: If the specified `label` is found in more than one token family or in none.
        """
        if label is None:
            if create_meaningful:
                if np.sum(self.families_cardinality(True, token_status)) == 0:
                    raise ValueError(
                        'Tring to create a term from an empty pool')

                probabilities = (self.families_cardinality(True, token_status) /
                                 np.sum(self.families_cardinality(True, token_status)))
                return np.random.choice(a=self.families_meaningful,
                                        p=probabilities).create(label=None,
                                                                token_status=token_status,
                                                                create_derivs=create_derivs,
                                                                all_vars = [family.variable for family in 
                                                                            self.families_demand_equation], 
                                                                **kwargs)
            else:
                probabilities = (self.families_cardinality(False, token_status) /
                                 np.sum(self.families_cardinality(False, token_status)))
                return np.random.choice(a=self.families,
                                        p=probabilities).create(label=None,
                                                                token_status=token_status,
                                                                create_derivs=create_derivs,
                                                                all_vars = [family.variable for family in 
                                                                            self.families_demand_equation],                                                                 
                                                                **kwargs)
        else:
            token_families = [family for family in self.families if label in family.tokens]
            if len(token_families) > 1:
                print([family.tokens for family in token_families])
                raise Exception(
                    'More than one family contains token with desired label.')
            elif len(token_families) == 0:
                raise Exception(
                    'Desired label does not match tokens in any family.')
            else:
                return token_families[0].create(label=label, token_status=token_status,
                                                all_vars = [family.variable for family in self.families_demand_equation],
                                                **kwargs)

    def create_from_family(self, family_label: str, token_status=None, **kwargs):
        """
        Create a new factor (token) based on a specified family.
        
        This method selects a family of factors based on the provided label and uses it to create a new factor.
        This is useful for constructing more complex equation structures by combining factors from different families.
        
        Args:
            family_label (str): The label of the family to create the factor from.
            token_status (dict, optional): Information about the status of all families. Defaults to None.
            **kwargs: Additional keyword arguments passed to the family's create method.
        
        Returns:
            Factor: The newly created factor.
        """
        # print([f.ftype for f in self.families], family_label)
        family = [f for f in self.families if family_label == f.ftype][0]
        return family.create(label=None, token_status=token_status, 
                             all_vars = [family.variable for family in self.families_demand_equation], 
                             **kwargs)

    def create_with_var(self, variable: str, token_status=None, **kwargs):
        """
        Creates a token (Factor) associated with a specific variable, selecting from available families based on token availability.
        
                This method is used to generate new equation terms by probabilistically choosing a family (representing a type of term) 
                that matches the given variable. The selection is weighted by the number of existing tokens within each family, 
                favoring families with more tokens. This helps to diversify the generated equations and explore different term combinations.
        
                Args:
                    variable (str): The variable that the created token should be associated with.
                    token_status (dict, optional): Information about the status of all token families. Defaults to None.
                    **kwargs: Additional keyword arguments to be passed to the family's `create` method.
        
                Returns:
                    Factor: A new token (Factor) created from the selected family.
        """
        # print([f.ftype for f in self.families], family_label)
        assert variable is not None, 'Can not create token with a specific variable for '
        families = [f for f in self.families if variable == f.variable]

        while True:
            try:
                probabilities = np.array([len(f.tokens) for f in families])
                family = np.random.choice(families, p = probabilities/probabilities.sum())                
                return family.create(label=None, token_status=token_status, 
                                     all_vars = [family.variable for family in self.families_demand_equation],
                                     **kwargs)
            except ValueError:
                families.remove(family)
                
    def __add__(self, other):
        """
        Combines two TFPool objects to create a new pool containing all families.
        
                This operation is essential for expanding the search space of possible equation structures by merging different sets of equation components.
        
                Args:
                    other (TFPool): The TFPool object to merge with.
        
                Returns:
                    TFPool: A new TFPool object containing the combined families from both input pools.
        """
        return TFPool(families=self.families + other.families)

    def __len__(self):
        """
        Returns the number of equation families stored in the pool.
        
                This reflects the diversity of equation structures being explored.
        
                Args:
                    self: The TFPool instance.
        
                Returns:
                    int: The number of equation families.
        """
        return len(self.families)

    def get_families_by_label(self, label):
        """
        Retrieves the `TokenFamily` that contains a specific token label.
        
        This method is used to locate the family to which a token belongs, 
        ensuring that each token is uniquely associated with a single family 
        within the pool. This is crucial for maintaining the structural 
        integrity of the equation discovery process.
        
        Args:
            label (`str`): The name of the token to search for within the families.
        
        Returns:
            `TokenFamily`: The family containing the specified token label.
        
        Raises:
            ValueError: If more than one family contains the same token label, indicating a potential ambiguity in the token assignments.
            IndexError: If no family contains the specified token label, suggesting that the token is not properly associated with any family.
        """
        containing_families = [family for family in self.families
                               if label in family.tokens]
        if len(containing_families) > 1:
            raise ValueError('More than one families contain the same tokens.')
        try:
            return containing_families[0]
        except IndexError:
            print(label, [family.tokens for family in self.families])
            raise IndexError('No family for token.')
