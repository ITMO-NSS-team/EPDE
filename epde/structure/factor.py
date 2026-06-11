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
from epde.structure.structure_template import _deepcopy_slots
from epde.evaluators import simple_function_evaluator

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

    def __init__(self, eval_function): # , eval_kwargs_keys={}
        self._evaluator = eval_function
        # self.eval_kwargs_keys = eval_kwargs_keys

    def apply(self, token, structural=False, func_args=None, torch_mode=False): # , **kwargs
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
        # assert list(kwargs.keys()) == self.eval_kwargs_keys, f'Kwargs {kwargs.keys()} != {self.eval_kwargs_keys}'
        return self._evaluator(token, structural, func_args, torch_mode = torch_mode)


class Factor(TerminalToken):
    __slots__ = ['_params', '_params_description', '_hash_val', '_latex_constructor', 'label',
                 'ftype', '_variable', '_all_vars', 'grid_set', 'grid_idx', 'is_deriv', 'deriv_code',
                 'cache_linked', '_status', 'equality_ranges', '_evaluator', 'saved',
                 '_cache_label', '_structural_label', '_structural_label_without_power']

    def __init__(self, token_name: str, status: dict, family_type: str, latex_constructor: Callable,
                 variable: str = None, all_vars: list = None, randomize: bool = False,
                 params_description=None, deriv_code=None, equality_ranges = None):
        # Label memoization slots: initialize BEFORE anything that
        # could trigger ``params.setter`` (e.g. ``set_parameters`` ->
        # ``TerminalToken.__init__`` -> ``self.params = ...``). The
        # overridden setter calls ``_invalidate_label_cache``, which
        # touches these slots.
        self._cache_label = None
        self._structural_label = None
        self._structural_label_without_power = None
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
        if self._variable is None:
            return self.ftype
        else:
            return self._variable
        
    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        from epde.loader import obj_to_pickle, attrs_from_dict        
        supported_attrs = []
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')

    @property
    def ann_representation(self) -> torch.nn.modules.container.Sequential:
        try:
            return self._ann_repr
        except AttributeError:
            _, grids = global_var.grid_cache.get_all()
            self._ann_repr = train_ann(grids = grids, data=self.evaluate())
            return self._ann_repr

    def predict_with_ann(self, grids: list):
        return use_ann_to_predict(self.ann_representation, grids)

    def reset_saved_state(self):
        self.saved = {'base': False, 'structural': False}

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status_dict):
        '''
        Parameters
        ----------
        status_dict : dict
            Description of token behaviour during the equation construction and processsing.
            Keys:
                'mandatory' - if True, a token from the family must be present in every term; 

                'unique_token_type' - if True, only one token of the family can be present in the term; 

                'unique_specific_token' - if True, a specific token can be present only once per term;            

                'requires_grid' - if True, the token requires grid for evaluation, if False, the tokens will be
                loaded from cache.
        '''
        self._status = status_dict

    def set_parameters(self, params_description: dict, equality_ranges: dict,
                       random=True, **kwargs):
        '''

        Avoid periodic parameters (e.g. phase shift) 

        '''
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
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator):
        if isinstance(evaluator, EvaluatorContained):
            self._evaluator = evaluator
        else:
            factor_family = [family for family in evaluator.families if family.ftype == self.ftype][0]
            self._evaluator = factor_family._evaluator # TODO: fix calling private attribute
            
    def evaluate(self, structural=False, grids=None, torch_mode: bool = False):
        assert self.cache_linked, 'Missing linked cache.'
        if self.is_deriv and grids is not None:
            raise Exception(
                'Derivatives have to evaluated on the initial grid')

        # Key the tensor cache on ``structural_label`` rather than
        # ``cache_label``: continuous-tolerance params (e.g. trig
        # ``freq`` with ``equality_ranges['freq'] > 0``) collapse into
        # bucket indices, so two trig factors with freq=1.99999999 and
        # freq=2.00000001 share one cache entry instead of evaluating
        # separately. For factors with only exact-tolerance params
        # (derivatives, grid, const, ...) the two labels are equal so
        # behaviour is unchanged.
        tcache_key = self.structural_label
        key = 'structural' if structural else 'base'
        if (tcache_key, structural) in global_var.tensor_cache and grids is None:
            return global_var.tensor_cache.get(tcache_key,
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
                    self.saved[key] = global_var.tensor_cache.add(tcache_key, value, structural=False,
                                                                  deriv_code=full_deriv_code)
                    global_var.tensor_cache.use_structural(use_base_data=True,
                                                           label=tcache_key)
                elif key == 'structural' and not self.status['structural_and_defalut_merged']:
                    global_var.tensor_cache.use_structural(use_base_data=False,
                                                           label=tcache_key,
                                                           replacing_data=value)
                else:
                    self.saved[key] = global_var.tensor_cache.add(tcache_key, value, structural=False,
                                                                  deriv_code=full_deriv_code)
            return value

    def _invalidate_label_cache(self):
        """Drop memoized ``cache_label`` / ``structural_label`` /
        ``structural_label_without_power``.

        Called automatically by the overridden ``params`` setter and
        ``set_param``. External code that mutates ``self.params`` via
        numpy in-place assignment (``factor.params[i] = X``) bypasses
        the setter and MUST call this method directly -- otherwise the
        memoized label remains stale and dedup / cache-key checks
        return wrong answers. See [[feedback_label_format_coupling]].
        """
        self._cache_label = None
        self._structural_label = None
        self._structural_label_without_power = None

    @TerminalToken.params.setter
    def params(self, params):
        # Reuse the base validation + storage + _fix_val reset, then
        # drop the memoized labels. Routing through the setter is the
        # canonical mutation path for whole-array reassignment; in-place
        # numpy index assignment bypasses this and must invalidate
        # explicitly (see ``_invalidate_label_cache`` docstring).
        TerminalToken.params.fset(self, params)
        self._invalidate_label_cache()

    def set_param(self, param, name=None, idx=None):
        # Single-parameter mutation path. ``TerminalToken.set_param``
        # writes to ``self._params[idx]`` in place and clears
        # ``_fix_val``; the label cache must also drop because the
        # quantization buckets / cache key depend on the param value.
        super().set_param(param, name=name, idx=idx)
        self._invalidate_label_cache()

    @property
    def cache_label(self):
        if self._cache_label is None:
            self._cache_label = factor_params_to_str(self)
        return self._cache_label

    def _quantized_params(self, drop_power: bool = False) -> tuple:
        """Return params with continuous-tolerance ones quantized into bucket
        indices and exact-equality ones passed through. Continuous params
        (those with ``equality_ranges[name] > 0``, e.g. trig ``freq``) get
        ``int((v - bounds[0]) / equality_ranges[name])``; exact-equality
        params (``power``, ``dim``) stay numeric. When ``drop_power=True``
        the param named ``'power'`` is omitted from the result tuple.
        """
        parts = []
        for i in range(len(self.params)):
            name = self.params_description[i]['name']
            if drop_power and name == 'power':
                continue
            v = self.params[i]
            tol = self.equality_ranges.get(name, 0)
            if tol > 0:
                origin = self.params_description[i]['bounds'][0]
                parts.append(int((v - origin) / tol))
            else:
                parts.append(v)
        return tuple(parts)

    @property
    def structural_label(self):
        """Hashable canonical identity for structural dedup.

        Sits next to ``cache_label`` (which keys the tensor cache and
        must stay exact). Continuous params are quantized into bucket
        indices so set-based dedup and ``Factor.__eq__``'s tolerance
        comparison agree.
        """
        if self._structural_label is None:
            self._structural_label = (self.cache_label[0],
                                      self._quantized_params(drop_power=False))
        return self._structural_label

    @property
    def structural_label_without_power(self):
        """``structural_label`` with the ``power`` param dropped.

        Used by ``simplify_equation`` to find shared factors across
        terms regardless of their individual powers.
        """
        if self._structural_label_without_power is None:
            self._structural_label_without_power = (
                self.cache_label[0],
                self._quantized_params(drop_power=True),
            )
        return self._structural_label_without_power

    @property
    def name(self):
        form = self.label + '{'
        for param_idx, param_info in self.params_description.items():
            form += param_info['name'] + ': ' + str(self.params[param_idx])
            if param_idx < len(self.params_description.items()) - 1:
                form += ', '
        form += '}'
        return form

    @property
    def latex_name(self):
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
        return self._hash_val

    @property
    def grids(self):
        _, grids = global_var.grid_cache.get_all()
        return grids

    def use_grids_cache(self):
        dim_param_idx = np.inf
        dim_set = False
        for param_idx, param_descr in self.params_description.items():
            if param_descr['name'] == 'dim':
                dim_param_idx = param_idx
                dim_set = True
        self.grid_idx = int(self.params[dim_param_idx]) if dim_set else 0
        self.grid_set = True

    def __deepcopy__(self, memo=None):
        # ``_evaluator``, ``equality_ranges``, ``_latex_constructor``,
        # ``_all_vars`` and ``deriv_code`` are family-owned objects set
        # once at family construction and never mutated per-factor;
        # share by reference saves ~5-10 % of deepcopy work in the
        # mutation hot path. ``_status`` IS mutated by ``Factor.status``
        # setter and stays deep-copied.
        new_struct = _deepcopy_slots(
            self, memo,
            attrs_to_share_by_ref=(
                '_evaluator', 'equality_ranges', '_latex_constructor',
                '_all_vars', 'deriv_code',
            ),
        )
        # ``Token`` / ``TerminalToken`` parents don't declare ``__slots__``,
        # so a Factor instance also carries a ``__dict__`` populated by
        # ``TerminalToken.__init__`` (val, cache_val, _fix_val, etc.).
        # Shallow update mirrors the original contract -- ``val`` is
        # reassigned on every ``Factor.value()`` call rather than mutated
        # in place, so aliasing the reference is safe.
        new_struct.__dict__.update(self.__dict__)
        return new_struct

    def copy_for_power_update(self):
        """Cheap clone for ``supplementary.filter_powers``.

        ``filter_powers`` deepcopies a factor only to mutate its
        ``power`` param via ``set_param``. Everything else can be aliased:
        only ``_params`` (a small numpy array written in place by
        ``set_param``) needs its own storage. The label-cache slots are
        zeroed because ``set_param`` would invalidate them anyway, and
        the dict-based parent attrs ride along by reference.

        This path was called 237k times per lv_new rep (47s cumtime via
        ``copy.deepcopy``); the per-call cost drops from ~200μs to a
        handful of attribute reads + one numpy array copy.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        for slot in cls.__slots__:
            try:
                setattr(new, slot, getattr(self, slot))
            except AttributeError:
                pass
        new._params = self._params.copy()
        new._cache_label = None
        new._structural_label = None
        new._structural_label_without_power = None
        new.__dict__.update(self.__dict__)
        return new

    def use_cache(self):
        self.cache_linked = True
