#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:39:18 2020

@author: mike_ubuntu
"""

import numpy as np
import math
import itertools
from typing import Union, Callable, List, Tuple
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import epde.globals as global_var
from epde.structure.factor import Factor, EvaluatorContained


def constancy_hard_equality(tensor, epsilon=1e-7):
    return np.abs(np.max(tensor) - np.min(tensor)) < epsilon


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

        set_evaluator(eval_function)
            Method to set the evaluator for the token family & its parameters;

        test_evaluator()
            Method to test, if the evaluator and tokens are set properly

        evaluate(token, token_params)
            Method, which uses the specific token evaluator to evaluate the passed token with its parameters
    """

    def __init__(self, token_type: str, variable:str = None, family_of_derivs: bool = False):
        """
        Initialize the token family;

        Args:
            token_type (`string`): The name of the token family; must be unique among other families.
            family_of_derivs (`bool`): tha flag about the existence of a derivative in the fanily of token
        """

        self.ftype = token_type
        self.variable = variable
        
        self.family_of_derivs = family_of_derivs
        self.evaluator_set = False
        self.params_set = False
        self.cache_set = False
        self.deriv_evaluator_set = True
        # Parameters whose values are known only from the fit data (a frequency
        # grid found in the spectrum); ``create`` refuses to draw until
        # ``install_param_grid`` supplies them.
        self.deferred_params = set()
        self.set_latex_form_constructor()

    def __len__(self) -> int:
        assert self.params_set, 'Familiy is not fully initialized.'
        return len(self.tokens)

    def set_status(self, demands_equation=False, meaningful=False,
                   s_and_d_merged=True, unique_specific_token=False,
                   unique_token_type=False, requires_grid=False, non_default_power=False) -> None:
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
            non_default_power (`boolean`): default - False,
                flag, that the behavior of power parameter is different. If `True`, values, other than 1 can be created during init as separate tokens.
        """
        self.status = {}
        self.status['demands_equation'] = demands_equation
        self.status['meaningful'] = meaningful
        self.status['structural_and_defalut_merged'] = s_and_d_merged
        self.status['unique_specific_token'] = unique_specific_token
        self.status['unique_token_type'] = unique_token_type
        self.status['requires_grid'] = requires_grid
        self.status['non_default_power'] = non_default_power

    def set_params(self, tokens, token_params, equality_ranges, derivs_solver_orders=None,
                   param_grids: dict = None) -> None:
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
            param_grids (`dict`): optional, parameter name -> its grid values, e.g. np.linspace(lo, hi, num).
                A gridded parameter takes only those values (drawn, crossed over, mutated and labelled by
                value on the grid; explicitly given values are moved to the nearest grid value), and its
                equality range becomes half the smallest gap between grid values. The values must lie within
                the parameter's (lower, higher) range. Not for 'power' or 'dim'. Parameters without a grid keep
                the continuous behaviour. A grid known only from the data is installed later with
                ``install_param_grid``.

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
        grids = self._validated_grids(token_params, param_grids)
        if grids:
            self.equality_ranges = dict(equality_ranges)
            self.equality_ranges.update({name: self._grid_tolerance(grid) for name, grid in grids.items()})
        self.param_grids = grids

        # if self.family_of_derivs:
        #     print(f'self.tokens is {self.tokens}')
        #     print(f'Here, derivs order is {self.derivs_ords}')
        if self.evaluator_set:
            self.test_evaluator()

    @staticmethod
    def _validated_grids(token_params, param_grids) -> dict:
        """``{name: sorted tuple of unique floats}`` for the gridded parameters,
        checked against the family's parameters and their (lower, higher) ranges."""
        grids = {}
        if not param_grids:
            return grids
        unknown = set(param_grids) - set(token_params)
        if unknown:
            raise ValueError(f'param_grids names parameters the family does not have: {sorted(unknown)}')
        # power is rewritten by power merging and RPS (filter_powers, the
        # root / min-order steps) without a snap, and dim is an index.
        fixed = {'power', 'dim'} & {name for name, grid in param_grids.items() if grid is not None}
        if fixed:
            raise ValueError(f'{sorted(fixed)} cannot take a grid; only continuous parameters can.')
        for name, values in param_grids.items():
            if values is None:
                continue
            grid = tuple(sorted({float(v) for v in np.asarray(values, dtype=float).ravel()}))
            if not grid or not np.all(np.isfinite(grid)):
                raise ValueError(f'The grid of parameter {name!r} must hold finite values, got {values!r}.')
            lo, hi = token_params[name]
            if grid[0] < lo or grid[-1] > hi:
                raise ValueError(f'The grid of parameter {name!r} leaves its range ({lo}, {hi}): {grid}.')
            grids[name] = grid
        return grids

    @staticmethod
    def _grid_tolerance(grid) -> float:
        # Grid values are stored floats, so two of them are equal or at least the
        # smallest gap apart: half that gap makes Factor.__eq__ agree with the
        # grid-value labels (0 for a single value).
        return float(np.min(np.diff(grid))) / 2. if len(grid) > 1 else 0.

    def install_param_grid(self, name: str, values, bounds: tuple = None):
        """Put parameter ``name`` on the grid ``values`` after the family was set
        up -- the frequencies found in the data, which exist only once the pool is
        built. ``bounds`` is the parameter's new (lower, higher) range (default: the
        grid's ends). New ``token_params`` / ``equality_ranges`` / ``param_grids``
        dicts are built rather than edited: factors from an earlier fit hold the old
        ``equality_ranges`` by reference. Unlike ``set_params`` this draws no test
        factor. Clears ``name`` from ``deferred_params``. Returns the grid.

        ``values`` may map axes -- values of the family's ``dim`` parameter -- to
        each axis's own grid (the frequencies of each axis's spectrum): a factor
        then draws its axis among these and its value from that axis's grid
        (``Factor.set_parameters``)."""
        if isinstance(values, dict):
            if 'dim' not in self.token_params:
                raise ValueError(f'A grid of {name!r} per axis needs a dim parameter; {self.ftype} has none.')
            lo, hi = self.token_params['dim']
            per_axis = {int(axis): tuple(sorted({float(v) for v in np.asarray(axis_values, dtype=float).ravel()}))
                        for axis, axis_values in values.items()}
            if not per_axis or not all(per_axis.values()):
                raise ValueError(f'The grid of parameter {name!r} is empty on some axis: {per_axis}.')
            outside = sorted(axis for axis in per_axis if not lo <= axis <= hi)
            if outside:
                raise ValueError(f'Axes {outside} of the grid of {name!r} are outside dim ({lo}, {hi}).')
            grid = tuple(sorted(set().union(*per_axis.values())))
        else:
            grid = tuple(sorted({float(v) for v in np.asarray(values, dtype=float).ravel()}))
            if not grid:
                raise ValueError(f'The grid of parameter {name!r} is empty.')
        token_params = type(self.token_params)(self.token_params)
        token_params[name] = ((float(bounds[0]), float(bounds[1])) if bounds is not None
                              else (grid[0], grid[-1]))
        grids = self._validated_grids(token_params, {name: grid})
        equality_ranges = dict(self.equality_ranges)
        equality_ranges[name] = self._grid_tolerance(grids[name])
        if isinstance(values, dict):
            grids = {name: per_axis}
        self.token_params = token_params
        self.equality_ranges = equality_ranges
        self.param_grids = {**getattr(self, 'param_grids', {}), **grids}
        self.deferred_params = set(getattr(self, 'deferred_params', set())) - {name}
        return grids[name]

    def fix_token_params(self, fixed: dict) -> None:
        """Give some tokens fixed parameter values: ``{label: {param: value}}``. A factor of such a
        label drawn by :meth:`create` takes these values instead of random ones (the draw itself is
        unchanged, so the random stream is too); parameters passed explicitly to ``create`` are
        kept as given. ``GridTokens`` ties each of several labels to its own axis this way: with a
        free ``dim``, two labels evaluate the same axis and make identical columns."""
        for label, params in fixed.items():
            if label not in self.tokens:
                raise ValueError(f'{self.ftype} has no token {label!r}.')
            unknown = set(params) - set(self.token_params)
            if unknown:
                raise ValueError(f'{self.ftype} has no parameter(s) {sorted(unknown)}.')
        self.token_fixed_params = {label: dict(params) for label, params in fixed.items()}

    def install_param_range(self, name: str, bounds: tuple) -> tuple:
        """Set parameter ``name``'s (lower, higher) range after the family was set up -- the axis
        index ``dim`` of a family sized to the data (``dimensionality='auto'``), known once the
        trajectories are. A new ``token_params`` dict is built, in the same key order (factors from
        an earlier fit hold the old one by reference); no test factor is drawn. Clears ``name`` from
        ``deferred_params``. Returns the range."""
        if name not in self.token_params:
            raise ValueError(f'The family has no parameter {name!r}.')
        token_params = type(self.token_params)(self.token_params)
        token_params[name] = tuple(bounds)
        self.token_params = token_params
        self.deferred_params = set(getattr(self, 'deferred_params', set())) - {name}
        return token_params[name]

    def set_evaluator(self, eval_function, suppress_eval_test=True) -> None:
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
            # >>> trig_eval_params = {'grid':grid, 'params_names':['power',  'freq', 'dim'], 'params_equality':{'power': 0, 'freq':0.05, 'dim':0}}
            >>> trigonometric_tokens.set_evaluator(trigonometric_evaluator,)

        """
        if isinstance(eval_function, EvaluatorContained):
            self._evaluator = eval_function
        else:
            self._evaluator = EvaluatorContained(eval_function)
        self.evaluator_set = True
        if self.params_set and not suppress_eval_test:
            self.test_evaluator()

    def set_deriv_evaluator(self, eval_functions, suppress_eval_test=True) -> None: # eval_kwargs_keys=[], 
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
                # print('Setting evaluator kwargs:', eval_kwargs_keys)
                _deriv_evaluator = EvaluatorContained(eval_function) # , eval_kwargs_keys
            self._deriv_evaluators[param_key] = _deriv_evaluator
        self.opt_param_labels = list(eval_functions.keys())
        self.deriv_evaluator_set = True
        if self.params_set and not suppress_eval_test:
            self.test_evaluator(deriv=True)

    def set_latex_form_constructor(self, latex_constructor: Callable = None) -> None:
        self.latex_constructor = latex_constructor

    def test_evaluator(self, deriv=False) -> None:
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
            # print('Test in the evaluator:', self._evaluator.eval_kwargs_keys)
            self.test_evaluation = self._evaluator.apply(self.test_token)
        print('Test evaluation performed correctly')

    def chech_constancy(self, **tfkwargs) -> None:
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
            data_label = (label, (1.0,))
            domain_constancies = []
            for domain_dict in global_var.tensor_cache.memory_default.values():
                data = domain_dict.get(data_label)

                try:
                    constancy = math.isclose(np.min(data), np.max(data))
                    domain_constancies.append(constancy)

                except TypeError:
                    print(f"No {label} data in cache for constancy check. Functionality of eval-d token check TBD.")
                    domain_constancies.append(False)
                    # continue

            # print(label, domain_constancies)
            if all(domain_constancies):
                constant_tokens_labels.append(label)

        for label in constant_tokens_labels:
            print(f'Function {label} is assumed to be constant in the studied domain. \
                    Removed from the equaton search.')
            data_label = (label, (1.0,))
            self.tokens.remove(label)
            # the live cache is cache_refactored.Cache (globals.py), whose method is deleteEntry
            global_var.tensor_cache.deleteEntry(data_label)

    def evaluate(self, token) -> None:
        """
        Applying evaluator in token
        """
        raise NotImplementedError('Method has been moved to the Factor class.')
        if self.evaluator_set:
            return self._evaluator.apply(token)
        else:
            raise TypeError(
                'Evaluator function or its parameters not set before evaluator application.')

    def create(self, label=None, token_status: dict = None, all_vars: List[str] = None,
               create_derivs: bool = False, **factor_params) -> Tuple[dict, Factor]:
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
        deferred = getattr(self, 'deferred_params', None)     # absent on families pickled before it existed
        if deferred:
            raise RuntimeError(f'{self.ftype}: {sorted(deferred)} come from the fit data and are resolved when '
                               'the pool is built (EpdeSearch.create_pool / fit); no factor can be drawn before.')
        if token_status is None or token_status == {}:
            token_status = {label: (0, self.token_params['power'][1], False)
                            for label in self.tokens}
        if label is None:
            try:
                if create_derivs:
                    label = np.random.choice([token for token in self.tokens
                                              if (not token_status[token][0] + 1 > token_status[token][1]
                                                  and self.is_derivative_token(token))])
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
        # A family pickled before param_grids existed has no attribute.
        param_grids = getattr(self, 'param_grids', None)
        if len(factor_params) == 0:
            new_factor.set_parameters(params_description=self.token_params,
                                      equality_ranges=self.equality_ranges,
                                      random=True, param_grids=param_grids)
            # Absent on families pickled before it existed.
            for name, value in getattr(self, 'token_fixed_params', {}).get(label, {}).items():
                new_factor.set_param(value, name=name)
        else:
            new_factor.set_parameters(params_description=self.token_params,
                                      equality_ranges=self.equality_ranges,
                                      random=False, param_grids=param_grids,
                                      **factor_params)
        new_factor.evaluator = self._evaluator

        return occupied_by_factor, new_factor

    def is_derivative_token(self, token: str) -> bool:
        """True iff ``token`` is a derivative: the family is a family of
        derivatives and the token's derivative code is not ``[None,]`` (the
        variable itself). Families of grids, trigonometric functions etc. have
        no ``derivs_ords`` at all, so they offer no derivative token."""
        return bool(self.family_of_derivs) and self.derivs_ords[token][0] is not None

    def cardinality(self, token_status: Union[dict, None] = None, derivs_only: bool = False) -> int:
        """
        Method for getting number of free place for creating new factors for that token family

        Args:
           token_status (`dict`):  information about usage of all tokens that belong to this family,
                Example: (number of used, max number for using, flag about permission to use)
           derivs_only (`boolean`): count only the derivative tokens (:meth:`is_derivative_token`),
                the ones ``create(create_derivs=True)`` can draw

        Returns:
            number of place (`int`)
        """
        if token_status is None or token_status == {}:
            token_status = {label: (0, self.token_params['power'][1], False)
                            for label in self.tokens}
        return len([token for token in self.tokens if token_status[token][0] < token_status[token][1]
                    and (not derivs_only or self.is_derivative_token(token))])

    def evaluate_all(self, all_vars: List[str]) -> None:
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

                _, generated_token = self.create(token_label, all_vars=all_vars, **params_sets_labeled)
                generated_token.use_cache()
                if self.status['requires_grid']:
                    generated_token.use_grids_cache()
                generated_token.scaled = False
                _ = generated_token.evaluate()
                print(generated_token.structural_label)
                if generated_token.structural_label not in global_var.tensor_cache.memory_default['numpy'].keys():
                    raise KeyError('Generated token somehow was not stored in cache.')


class TFPool(object):
    """
     Class stored pool for token families

     Args:
        families (`list`): toen families that using in that run
    """    
    def __init__(self, families: list):
        self.families = families
    
    def manual_reconst(self, attribute:str, value, except_attrs:dict) -> None:
        from epde.loader import obj_to_pickle, attrs_from_dict        
        supported_attrs = []
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')
    
    @property
    def families_meaningful(self) -> List[TokenFamily]:
        """
        Getting token families, that are meaningful
        """
        return [family for family in self.families if family.status['meaningful']]

    @property
    def families_demand_equation(self) -> List[TokenFamily]:
        """
        Getting token families, that must have an individual equation
        """
        return [family for family in self.families if family.status['demands_equation']]

    @property
    def families_supplementary(self) -> List[TokenFamily]:
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
                             token_status: Union[dict, None] = None, derivs_only: bool = False):
        """
        Getting number of free place for creating new factors for each token family

        Args:
            meaningful_only (`boolean`): using only meaningful families
            token_status (`dict`): information about usage of all tokens that belong to all families of the class
            derivs_only (`boolean`): count only derivative tokens (see ``TokenFamily.cardinality``)

        Returns:
            numpy.array with integer values (number of free place in each `self.families`)
        """
        if meaningful_only:
            return np.array([family.cardinality(token_status, derivs_only=derivs_only)
                             for family in self.families_meaningful])
        else:
            return np.array([family.cardinality(token_status, derivs_only=derivs_only)
                             for family in self.families])

    def create(self, label=None, create_meaningful: bool = False, token_status=None,
               create_derivs: bool = False, **kwargs) -> Union[str, Factor]:
        """
        Create token from family for current running

        Args:
            label (`str`): if noe None, that create token with name == label
            create_meaningful (`boolean`): the choice of the token to create is selected from meaningful family
            token_status (`dict`): information about status of all families
            create_derivs (`boolean`): flag about the presence of derivatives in the token structure.
                The family is then drawn among those that can still offer a derivative token, in
                proportion to how many they can offer: a family without one (grids, trigonometric
                functions, the variable itself) cannot serve the draw.

        Returns:
            created `Factor`
        """
        if label is None:
            if create_meaningful:
                cardinality = self.families_cardinality(True, token_status, derivs_only=create_derivs)
                if np.sum(cardinality) == 0:
                    raise ValueError(
                        'Tring to create a term from an empty pool')

                probabilities = cardinality / np.sum(cardinality)
                return np.random.choice(a=self.families_meaningful,
                                        p=probabilities).create(label=None,
                                                                token_status=token_status,
                                                                create_derivs=create_derivs,
                                                                all_vars = [family.variable for family in
                                                                            self.families_demand_equation],
                                                                **kwargs)
            else:
                cardinality = self.families_cardinality(False, token_status, derivs_only=create_derivs)
                if create_derivs and np.sum(cardinality) == 0:
                    raise ValueError('No family can offer a derivative token for the term')
                probabilities = cardinality / np.sum(cardinality)
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
        Create token from choosing family

        Args:
            family_label (`str`): the name of the family from which the token will be created
            token_status (`dict`): information about status of all families
        
        Returns:
            created `Factor`
        """
        # print([f.ftype for f in self.families], family_label)
        family = [f for f in self.families if family_label == f.ftype][0]
        return family.create(label=None, token_status=token_status, 
                             all_vars = [family.variable for family in self.families_demand_equation], 
                             **kwargs)

    def create_with_var(self, variable: str, token_status=None, **kwargs):
        """
        Create token from choosing family

        Args:
            family_label (`str`): the name of the family from which the token will be created
            token_status (`dict`): information about status of all families
        
        Returns:
            created `Factor`
        """
        # print([f.ftype for f in self.families], family_label)
        assert variable is not None, 'Can not create token with a specific variable for '
        families = [f for f in self.families if variable == f.variable]

        max_iter = len(families) + 1
        family = None
        for _ in range(max_iter):
            if not families:
                raise RuntimeError(
                    f"TFPool.create_with_var: no family can produce a token for variable={variable!r}"
                )
            try:
                probabilities = np.array([len(f.tokens) for f in families])
                family = np.random.choice(families, p=probabilities/probabilities.sum())
                return family.create(label=None, token_status=token_status,
                                     all_vars=[fam.variable for fam in self.families_demand_equation],
                                     **kwargs)
            except ValueError:
                if family is not None and family in families:
                    families.remove(family)
                    family = None
        raise RuntimeError(
            f"TFPool.create_with_var: exhausted {max_iter} attempts for variable={variable!r}"
        )
                
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
