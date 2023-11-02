#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:39:18 2020

@author: mike_ubuntu
"""

import numpy as np
import itertools
from typing import Union, Callable

import pickle

import epde.globals as global_var
from epde.structure.factor import Factor


def constancy_hard_equality(tensor, epsilon=1e-7):
    # print(np.abs(np.max(tensor) - np.min(tensor)), epsilon, type(np.abs(np.max(tensor) - np.min(tensor))),  type(epsilon))
    return np.abs(np.max(tensor) - np.min(tensor)) < epsilon


class EvaluatorContained(object):
    """
    Class for evaluator of token (factor of the term in the sought equation) values with arbitrary function

    Attributes:
        _evaluator (`callable`): a function, which returns the vector of token values, evaluated on the studied area;
        params (`dict`): dictionary, containing parameters of the evaluator (like grid, on which the function is evaluated or matrices of pre-calculated function)

    Methods:
        set_params(**params)
            set the parameters of the evaluator, using keyword arguments
        apply(token, token_params)
            apply the defined evaluator to evaluate the token with specific parameters
    """

    def __init__(self, eval_function, eval_kwargs_keys={}):  # , deriv_eval_function = None
        self._evaluator = eval_function
        # if deriv_eval_function is not None: self._deriv_evaluator = deriv_eval_function
        self.eval_kwargs_keys = eval_kwargs_keys

    def apply(self, token, structural=False, grids=None, **kwargs):
        """
        Apply the defined evaluator to evaluate the token with specific parameters.

        Args:
            token (`epde.main_structures.factor.Factor`): symbolic label of the specific token, e.g. 'cos';
        token_params (`dict`): dictionary with keys, naming the token parameters (such as frequency, axis and power for trigonometric function) 
            and values - specific values of corresponding parameters.

        Raises:
            `TypeError`
                If the evaluator could not be applied to the token.
        """
        assert list(kwargs.keys()) == self.eval_kwargs_keys
        return self._evaluator(token, structural, grids, **kwargs)


class TokenFamily(object):
    """
    Class for the type (family) of tokens, from which the tokens are taken as factors in the terms of the equation

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

        set_evaluator(eval_function, **eval_params)
            Method to set the evaluator for the token family & its parameters;

        test_evaluator()
            Method to test, if the evaluator and tokens are set properly

        evaluate(token, token_params)
            Method, which uses the specific token evaluator to evaluate the passed token with its parameters
    """

    def __init__(self, token_type: str, family_of_derivs: bool = False):
        """
        Initialize the token family;

        Args:
            token_type (`string`): The name of the token family; must be unique among other families.
            family_of_derivs (`bool`): tha flag about the existence of a derivative in the fanily of token
        """

        self.ftype = token_type
        self.family_of_derivs = family_of_derivs
        self.evaluator_set = False
        self.params_set = False
        self.cache_set = False
        self.deriv_evaluator_set = True
        self.set_latex_form_constructor()

    def __len__(self):
        assert self.params_set, 'Familiy is not fully initialized.'
        return len(self.tokens)

    def set_status(self, demands_equation=False, meaningful=False,
                   s_and_d_merged=True, unique_specific_token=False,
                   unique_token_type=False, requires_grid=False):
        """
        Set the status of the elements of the token family; 

        Args:
            demands_equation (`boolean`): default - False
                flag about the existence of restrictions for equation
            meaningful (`boolean`):if True, a token from the family must be present in every term
            unique_token_type (`boolean`): if True, only one token of the family can be present in the term
            unique_specific_token (`boolean`): if True, a specific token can be present only once per term
            s_and_d_merged (`boolean`): default - True,
                flag, that the base values of the token are used as the structural values (generally, in other cases the normalized values are used as structural)
            requires_grid (`boolean`): default - False, 
                flag, that a grid is required to evaluate the token
        """
        self.status = {}
        self.status['demands_equation'] = demands_equation
        self.status['meaningful'] = meaningful
        self.status['structural_and_defalut_merged'] = s_and_d_merged
        self.status['unique_specific_token'] = unique_specific_token
        self.status['unique_token_type'] = unique_token_type
        self.status['requires_grid'] = requires_grid

    def set_params(self, tokens, token_params, equality_ranges, derivs_solver_orders=None):
        """
        Define the token family with list of tokens and their parameters

        Args:
            tokens (`list of strings`): List of function names, describing all of the functions, belonging to the family. E.g. for 'trigonometric' token type, 
                this list will be ['sin', 'cos']
            token_params (`OrderedDict`): Available range for token parameters. Ordered dictionary with key - token parameter name, and value - tuple with 2 elements:
                (lower boundary, higher boundary), while type of boundaries describes the avalable token params: 
                if int - the parameters will be integer, if float - float.
            equality_ranges (`dict`): error for equality of token parameters, key is name of parameter
            derivs_solver_orders (`list`): keys for derivatides on `int` format for `solver`

        Example:
        ----------
            >>> token_names_trig = ['sin', 'cos']        
            >>> trig_token_params = OrderedDict([('power', (1, 1)), ('freq', (0.9, 1.1)), ('dim', (0, u_initial.ndim))])
            >>> trigonometric_tokens.set_params(token_names_trig, trig_token_params)
        """

        assert bool(derivs_solver_orders is not None) == bool(
            self.family_of_derivs), 'Solver form must be set for derivatives, and only for them.'

        self.tokens = tokens
        self.token_params = token_params
        if self.family_of_derivs:
            self.derivs_ords = {token: derivs_solver_orders[idx] for idx, token in enumerate(tokens)}
        self.params_set = True
        self.equality_ranges = equality_ranges

        if self.family_of_derivs:
            print(f'self.tokens is {self.tokens}')
            print(f'Here, derivs order is {self.derivs_ords}')
        if self.evaluator_set:
            self.test_evaluator()

    # , ): , **eval_params   #Test, if the evaluator works properly
    def set_evaluator(self, eval_function, eval_kwargs_keys=[], suppress_eval_test=True):
        """
        Define the evaluator for the token family and its parameters

        Args:
            eval_function (`function or EvaluatorContained object`): Function, used in the evaluator, or the evaluator
            eval_params (`keyword arguments`): The parameters for evaluator; must contain params_names (names of the token parameters) &
                param_equality (for each of the token parameters, range in which it considered as the same),
            suppress_eval_test (`boolean`): if True, run `test_evaluator` for testing of method for evaluating token

        Example:
            >>> def trigonometric_evaluator(token, token_params, eval_params):
            >>>     
            >>>     '''
            >>>     
            >>>     Example of the evaluator of token values, appropriate for case of trigonometric functions to be calculated on grid, with results in forms of tensors
            >>>     
            >>>     Parameters
            >>>     ----------
            >>>     token: {'sin', 'cos'}
            >>>         symbolic form of the function to be evaluated: 
            >>>     token_params: dictionary: key - symbolic form of the parameter, value - parameter value
            >>>         names and values of the parameters for trigonometric functions: amplitude, frequency & dimension
            >>>     eval_params : dict
            >>>         Dictionary, containing parameters of the evaluator: in this example, it contains coordinates np.meshgrid with coordinates for points, 
            >>>         names of the token parameters (frequency, axis and power). Additionally, the names of the token parameters must be included with specific key 'params_names', 
            >>>         and parameters range, for which each of the tokens if consedered as "equal" to another, like sin(1.0001 x) can be assumed as equal to (0.9999 x)
            >>>     
            >>>     Returns
            >>>     ----------
            >>>     value : numpy.ndarray
            >>>         Vector of the evaluation of the token values, that shall be used as target, or feature during the LASSO regression.
            >>>         
            >>>     '''
            >>>     
            >>>     assert 'grid' in eval_params
            >>>     trig_functions = {'sin' : np.sin, 'cos' : np.cos}
            >>>     function = trig_functions[token]
            >>>     grid_function = np.vectorize(lambda *args: function(token_params['freq']*args[token_params['dim']])**token_params['power'])
            >>>     value = grid_function(*eval_params['grid'])
            >>>     return value
            >>> 
            >>> der_eval_params = {'token_matrices':simple_functions, 'params_names':['power'], 'params_equality':{'power' : 0}}
            >>> trig_eval_params = {'grid':grid, 'params_names':['power',  'freq', 'dim'], 'params_equality':{'power': 0, 'freq':0.05, 'dim':0}}
            >>> trigonometric_tokens.set_evaluator(trigonometric_evaluator, **trig_eval_params)

        """
        if isinstance(eval_function, EvaluatorContained):
            self._evaluator = eval_function
        else:
            self._evaluator = EvaluatorContained(eval_function, eval_kwargs_keys)
        self.evaluator_set = True
        if self.params_set and not suppress_eval_test:
            self.test_evaluator()

    def set_deriv_evaluator(self, eval_functions, eval_kwargs_keys=[], suppress_eval_test=True):
        """
        Define the evaluator for the derivatives of the token family and its parameters

        Args:
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
                print('Setting evaluator kwargs:', eval_kwargs_keys)
                _deriv_evaluator = EvaluatorContained(eval_function, eval_kwargs_keys)
            self._deriv_evaluators[param_key] = _deriv_evaluator
        self.opt_param_labels = list(eval_functions.keys())
        self.deriv_evaluator_set = True
        if self.params_set and not suppress_eval_test:
            self.test_evaluator(deriv=True)

    def set_latex_form_constructor(self, latex_constructor: Callable = None):
        self.latex_constructor = latex_constructor

    def test_evaluator(self, deriv=False):
        """
        Method to test, if the evaluator and tokens are set properly

        Raises Exception, if the evaluator does not work properly.
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
            print('Test in the evaluator:', self._evaluator.eval_kwargs_keys)
            self.test_evaluation = self._evaluator.apply(self.test_token)
        print('Test evaluation performed correctly')

    def chech_constancy(self, test_function, **tfkwargs):
        '''
        Method to check, if any single simple token in the studied domain is constant, or close to it. The constant token is to be displayed and deleted from tokens and cache.

        Args:
            test_function (`callable`): the method used to evaluate

        Returns:
            None
        '''
        assert self.params_set
        constant_tokens_labels = []
        for label in self.tokens:
            print(type(global_var.tensor_cache.memory[label + ' power 1']))
            constancy = test_function(global_var.tensor_cache.memory[label + ' power 1'], **tfkwargs)
            if constancy:
                constant_tokens_labels.append(label)

        for label in constant_tokens_labels:
            print('Function ', label,
                  'is assumed to be constant in the studied domain. Removed from the equaton search')
            self.tokens.remove(label)
            global_var.tensor_cache.delete_entry(label + ' power 1')

    def evaluate(self, token):    # Return tensor of values of applied evaluator
        """
        Applying evaluator in token
        """
        raise NotImplementedError('Method has been moved to the Factor class')
        if self.evaluator_set:
            return self._evaluator.apply(token)
        else:
            raise TypeError(
                'Evaluator function or its parameters not set brfore evaluator application.')

    def create(self, label=None, token_status: dict = None,
               create_derivs: bool = False, **factor_params):
        """
        Method for creating element of the token family

        Args:
            label (`str`): one name of them, if label is None - random selection occurs from possible tokns for that family
            token_status (`dict`): information about usage of all tokens that belong to this family, 
                if `label` is not None, this argument will not be considered. Example: (number of used, max number for using, flag about permission to use)
            create_derivs (`boolean`): default - False
                flag about the presence of derivatives in the token structure

        Returns:
            occupied_by_factor (`dict`): information about blocked elements after cteated the factor
            new_factor (`Factor`): resulting factor for that token family
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
                print(
                    f'An error while creating factor of {self.ftype} token family')
                print('Status description:', token_status, ' all:', self.tokens)
                raise ValueError("'a' cannot be empty unless no samples are taken")

        if self.family_of_derivs:
            factor_deriv_code = self.derivs_ords[label]
        else:
            factor_deriv_code = None
        new_factor = Factor(token_name=label, deriv_code=factor_deriv_code,
                            status=self.status, family_type=self.ftype, 
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
        new_factor.set_evaluator(self._evaluator)
        return occupied_by_factor, new_factor

    def cardinality(self, token_status: Union[dict, None] = None):
        """
        Method for getting number of free place for creating new factors for that token family

        Args:
           token_status (`dict`):  information about usage of all tokens that belong to this family, 
                Example: (number of used, max number for using, flag about permission to use)

        Returns:
            number of place (`int`)
        """
        if token_status is None or token_status == {}:
            token_status = {label: (0, self.token_params['power'][1], False)
                            for label in self.tokens}
        return len([token for token in self.tokens if token_status[token][0] < token_status[token][1]])

    def evaluate_all(self):
        """
        Apply method of evaluation for all tokens in token family
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

                _, generated_token = self.create(token_label, **params_sets_labeled)
                generated_token.use_cache()
                if self.status['requires_grid']:
                    generated_token.use_grids_cache()
                generated_token.scaled = False
                # _ = self._evaluator.apply(generated_token)
                _ = generated_token.evaluate()
                print(generated_token.cache_label)
                if generated_token.cache_label not in global_var.tensor_cache.memory_default.keys():
                    raise KeyError('Generated token somehow was not stored in cache.')


class TFPool(object):
    """
     Class stored pool for token families

     Args:
        families (`list`): toen families that using in that run
    """
    def __init__(self, families: list, stored_pool=None):
        if stored_pool is not None:
            self = pickle.load(stored_pool)
        self.families = families

    @property
    def families_meaningful(self):
        """
        Getting token families, that are meaningful
        """
        return [family for family in self.families if family.status['meaningful']]

    @property
    def families_demand_equation(self):
        """
        Getting token families, that must have an individual equation
        """
        return [family for family in self.families if family.status['demands_equation']]

    @property
    def families_supplementary(self):
        """
        Getting token families, that are not meaningful
        """
        return [family for family in self.families if not family.status['meaningful']]

    @property
    def families_equationless(self):
        """
        Getting token families, whose presence in the equation is optional
        """
        return [family for family in self.families if not family.status['demands_equation']]

    @property
    def labels_overview(self):
        """
        Getting pairs from each token familly by next form: (name of token for this family, max number of token for using) 
        """
        overview = []
        for family in self.families:
            overview.append((family.tokens, family.token_params['power'][1]))
        return overview

    def families_cardinality(self, meaningful_only: bool = False,
                             token_status: Union[dict, None] = None):
        """
        Getting number of free place for creating new factors for each token family

        Args:
            meaningful_only (`boolean`): using only meaningful families
            token_status (`dict`): information about usage of all tokens that belong to all families of the class

        Returns:
            numpy.array with integer values (number of free place in each `self.families`)
        """
        if meaningful_only:
            return np.array([family.cardinality(token_status) for family in self.families_meaningful])
        else:
            return np.array([family.cardinality(token_status) for family in self.families])

    def create(self, label=None, create_meaningful: bool = False, token_status=None,
               create_derivs: bool = False, **kwargs) -> Union[str, Factor]:
        """
        Create token from family for current running

        Args:
            label (`str`): if noe None, that create token with name == label
            create_meaningful (`boolean`): the choice of the token to create is selected from meaningful family
            token_status (`dict`): information about status of all families
            create_derivs (`boolean`): flag about the presence of derivatives in the token structure
        
        Returns:
            created `Factor` 
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
                                                                **kwargs)
            else:
                probabilities = (self.families_cardinality(False, token_status) /
                                 np.sum(self.families_cardinality(False, token_status)))
                return np.random.choice(a=self.families,
                                        p=probabilities).create(label=None,
                                                                token_status=token_status,
                                                                create_derivs=create_derivs,
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
                return token_families[0].create(label=label,
                                                token_status=token_status,
                                                **kwargs)

    def create_from_family(self, family_label: str, token_status=None, **kwargs):
        """
        Create token from choosing family

        Args:
            family_label (`str`): the name of the family from which the token will be created
            token_status (`dict`): information about status of all families
        
        Returns:
            created `Factor`
        """
        # print('family_label', family_label, 'self.families', self.families)
        family = [f for f in self.families if family_label == f.ftype][0]
        return family.create(label=None, token_status=token_status, **kwargs)

    def __add__(self, other):
        return TFPool(families=self.families + other.families)

    def __len__(self):
        return len(self.families)

    def get_families_by_label(self, label):
        """
        Getting family by input name of token

        Args:
            label (`str`): the name of the token that will be used to search for his family

        Returns:
            `TokenFamily` for token by input name
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

    def save(self, filename: str):
        """
        Saving information about all families

        Args:
            filename (`str`): path to file, which will be save data
        """
        file_to_store = open(filename, "wb")
        pickle.dump(self, file_to_store)
        file_to_store.close()
