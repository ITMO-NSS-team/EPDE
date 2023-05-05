import numpy as np
from collections import OrderedDict
from functools import reduce, singledispatchmethod

from epde.structure.factor import Factor
from epde.structure.main_structures import Term
from epde.supplementary import factor_params_to_str

from copy import deepcopy
from pprint import pprint
import epde.globals as global_var



class ParametricFactor(Factor):
    __slot__ = ['_params', '_params_description', '_hash_val',
                'label', 'type', 'grid_set', 'grid_idx', 'is_deriv', 'deriv_code',
                'cache_linked', '_status', 'equality_ranges', '_evaluator', 'saved',
                'params_defined', 'params_to_optimize']

    def __init__(self, token_name: str, status: dict, family_type: str, params_description=None,
                 params_to_optimize=None, deriv_code=None, equality_ranges=None):
        self.params_defined = False
        self.params_to_optimize = params_to_optimize
        super().__init__(token_name, status, family_type, False,
                         params_description, deriv_code, equality_ranges)

        _params_description = {}
        for param_idx, param_info in enumerate(params_description.items()):
            _params_description[param_idx] = {'name': param_info[0],
                                              'bounds': param_info[1]}
        self.params_description = _params_description
        # Костыль, разобраться с лишними объектами
        self.params_description_odict = params_description
        self.equality_ranges = equality_ranges
        self.defined_params_passed = False
        self.params_predefined = {}

        self.reset_saved_state()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        for k in self.__slots__:
            try:
                if not isinstance(k, list):
                    setattr(result, k, deepcopy(getattr(self, k), memo))
                else:
                    temp = []
                    for elem in getattr(self, k):
                        temp.append(deepcopy(elem, memo))
                    setattr(result, k, temp)
            except AttributeError:
                pass

        return result

    def set_grad_evaluator(self, evaluator):
        self._grad_evaluator = evaluator

    @property
    def grad_cache_label(self):
        grad_cache_label = list(factor_params_to_str(self))
        grad_cache_label[0] += '_grad'
        return tuple(grad_cache_label)

    @property
    def required_params(self):
        return self.hash_descr, self.params_to_optimize

    def __contains__(self, element):
        return element in self.params_to_optimize

    def use_params(self, params):
        self.reset_saved_state()
        assert len(params) == len(self.params_to_optimize), 'The number of the passed parameters does not match declared problem'
        _params = np.ones(shape=len(self.params_description))
        for param_idx, param_info in self.params_description.items():
            if param_info['name'] not in self.params_to_optimize and param_info['name'] != 'power':
                if not self.defined_params_passed:
                    _params[param_idx] = (np.random.randint(param_info['bounds'][0], param_info['bounds'][1] + 1) if isinstance(param_info['bounds'][0], int)
                                          else np.random.uniform(param_info['bounds'][0], param_info['bounds'][1])) if param_info['bounds'][1] > param_info['bounds'][0] else param_info['bounds'][0]
                else:
                    _params[param_idx] = self.params_predefined[param_info['name']]

            elif param_info['name'] in self.params_to_optimize:
                opt_param_idx = self.params_to_optimize.index(param_info['name'])
                _params[param_idx] = params[opt_param_idx][1]
        _kw_params = {param_info['name']: _params[idx] for idx, param_info in enumerate(list(self.params_description.values()))}

        super().set_parameters(self.params_description_odict,
                               self.equality_ranges, random=False, **_kw_params)

    def set_defined_params(self, defined_params: dict):
        for param_label, val in defined_params.items():
            if val is None:
                raise ValueError('Trying to set the parameter with None value')
            self.params_predefined[param_label] = val
        self.defined_params_passed = True

    def reset_saved_state(self):
        deriv_eval_dict = {label: False for label in self.params_to_optimize}
        self.saved = {'base': False,
                      'deriv': deriv_eval_dict, 'structural': False}

    def eval_grad(self, param_label: str):
        # if self.saved['deriv'][param_label]:
        #     return global_var.tensor_cache.get(self.grad_cache_label,
        #                                        structural = False)
        # else:
        value = self._grad_evaluator[param_label].apply(self)
        # self.saved['deriv'][param_label] = global_var.tensor_cache.add(self.grad_cache_label, value, structural = False)
        return value


class ParametricTerm(Term):
    __slots__ = ['_history', 'structure', 'interelement_operator', 'saved', 'saved_as',
                 'pool', 'max_factors_in_term', 'cache_linked', 'occupied_tokens_labels',
                 'parametric_factors', 'defined_factors', 'params_to_optimize', 'all_params']

    def __init__(self, pool, parametric_factors: dict, defined_factors: dict, interelement_operator=np.multiply):
        self.parametric_factors = parametric_factors
        self.defined_factors = defined_factors
        # print('parametric factors:', self.parametric_factors)
        self.all_params = reduce(lambda x, y: x+y, [factor.params_to_optimize for factor in self.parametric_factors.values()], [])
        # print('Params in term:', self.all_params, len(self.all_params))
        self.pool = pool
        self.operator = interelement_operator

    def __deepcopy__(self, memo):
        # print('while copying factor:')
        # print('properties of self')
        # pprint(vars(self))

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        # clss = self.__class__
        # new_struct = clss.__new__(clss)
        # memo[id(self)] = new_struct

        # new_struct.__dict__.update(self.__dict__)

        # attrs_to_avoid_copy = []

        for k in self.__slots__:
            try:
                if not isinstance(k, list):
                    setattr(result, k, deepcopy(getattr(self, k), memo))
                else:
                    temp = []
                    for elem in getattr(self, k):
                        temp.append(deepcopy(elem, memo))
                    setattr(result, k, temp)
            except AttributeError:
                pass
        # print('properties of copy')
        # pprint(vars(result))
        return result

    @property
    def term_id(self) -> int:
        _term_id = sum([factor.hash_descr for factor in self.parametric_factors.values()])
        # print('_term_id', _term_id)
        return _term_id

    def parse_opt_params(self, params: np.ndarray):
        params_dict = OrderedDict()
        init_idx = 0
        for factor in self.parametric_factors.values():
            hash_descr, factor_params = factor.required_params
            params_dict[hash_descr] = list(zip(factor_params, params[init_idx: init_idx + len(factor_params)]))
            init_idx += len(factor_params)
        return params_dict

    def opt_params_num(self):
        return len(self.all_params)
    # sum([len(params) for params in self.parse_opt_params().values()])

    def use_params(self, params: dict):
        for factor_label, factor_params in params.items():
            # print('setting params:', params)
            self.parametric_factors[factor_label].use_params(factor_params)

    def evaluate(self):
        value = np.multiply.reduce([factor.evaluate() for factor in self.parametric_factors.values()] +
                                   [factor.evaluate() for factor in self.defined_factors.values()])  # , initial =
        return value.reshape(-1)

    def evaluate_grad(self, parameter):
        # print(self.parametric_factors, parameter)
        param_factor_idxs = [idx for idx, factor in enumerate(self.parametric_factors.values()) if parameter in factor]
        # print(param_factor_idxs)
        assert len(param_factor_idxs) == 1, 'More than one factor in a term contains the same parameter'

        return np.multiply.reduce([factor.evaluate() for idx, factor in enumerate(self.parametric_factors.values()) if idx != param_factor_idxs[0]] +
                                  [factor.evaluate() for factor in self.defined_factors.values()]) * list(self.parametric_factors.values())[param_factor_idxs[0]].eval_grad(parameter)

    def equivalent_common_term(self):
        factors_to_convert = []
        const_set = False
        for factor in self.parametric_factors.values():
            if factor.label != 'const':
                factors_to_convert.append(factor)
            else:
                const_set = True
                const_val = factor.param(name='value')

        for factor in self.defined_factors.values():
            factors_to_convert.append(factor)

        if not const_set:
            const_val = 1
        equilvalent_term = Term(pool=self.pool, passed_term=factors_to_convert,
                                max_factors_in_term=len(factors_to_convert))
        return const_val, equilvalent_term

    @singledispatchmethod
    def __contains__(self, element):
        raise NotImplementedError(
            'Incorrect type of the requested item for the __contains__ method')

    @__contains__.register
    def _(self, element: str):
        return element in self.params_to_optimize

    @__contains__.register
    def _(self, element: ParametricFactor):
        return element in self.parametric_factors.values()

    @__contains__.register
    def _(self, element: Factor):
        return element in self.defined_factors.values()
