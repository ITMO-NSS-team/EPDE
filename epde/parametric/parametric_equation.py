import numpy as np
import scipy.optimize as optimize
import time

from pprint import pprint
from copy import deepcopy
from collections import OrderedDict
from typing import Union
from functools import reduce, singledispatchmethod

from epde.structure.main_structures import Equation



class ParametricEquation(object):
    def __init__(self, pool, terms: Union[list, tuple], right_part_index: int = -1):
        self.pool = pool
        self.terms = terms

        self.total_params_count = [term.opt_params_num()
                                   for term in self.terms]
        total_count = 0
        self.param_term_beloning = {}
        for term_idx, term_params_num in enumerate(self.total_params_count):
            local_count = 0
            for _ in range(term_params_num):
                self.param_term_beloning[total_count] = (term_idx, local_count)
                total_count += 1
                local_count += 1

        self.rpi = right_part_index if right_part_index >= 0 else len(
            terms) + right_part_index
        self._optimization_held = False

    def optimize_equations(self, initial_params=None, method='L-BFGS-B'):
        def opt_func(params, *variables):
            '''

            Into the params the parametric tokens (or better their parameters) shall be passed,
            the variables: variables[0] - the object, containing parametric equation.  

            '''
            # print('params in opt_func', params)
            err = np.linalg.norm(variables[0].evaluate_with_params(params))
            print('error:', err)
            return err

        def opt_fun_grad(params, *variables):
            # print('evaluating gradient')
            grad = np.zeros_like(params)
            for param_idx, param_in_term_props in variables[0].param_term_beloning.items():
                grad[param_idx] = np.sum(
                    variables[0].evaluate_grad(params, param_in_term_props))
            print('gradient:', grad)
            return grad

        if initial_params is None:
            initial_params = np.zeros(np.sum(self.total_params_count))

        # print('Reaching copy moment')
        optimizational_copy = deepcopy(self)

        # print("----------------------------------------")
        # pprint(vars(optimizational_copy))
        # print("----------------------------------------")

        # print(initial_params)
        # print('Optimized function test run', opt_func(initial_params, self))
        # print('Grad function test run', opt_fun_grad(initial_params, self))

        def opt_lbd(params): return opt_func(params, optimizational_copy)
        def opt_grd(params): return opt_fun_grad(params, optimizational_copy)

        if method == 'L-BFGS-B':
            optimal_params = optimize.minimize(opt_lbd, x0=initial_params)
        elif method == 'GD':
            optimal_params = optimize.fmin_cg(opt_lbd, x0=initial_params, fprime=opt_grd)
        else:
            raise NotImplementedError(
                'Implemented methods of parameter optimization are limited to "L-BFGS-B" and gradient descent as "GD"')
        print(type(optimal_params))

        if type(optimal_params) == np.ndarray:
            self.set_term_params(optimal_params)
        elif type(optimal_params) == optimize.optimize.OptimizeResult:
            self.set_term_params(optimal_params.x)
        self._optimization_held = True
        self.equation_set = False

    def parse_eq_terms(self):
        weights = []
        equation_term = []
        for idx, term in enumerate(self.terms):
            weight, equivalent_term = term.equivalent_common_term()
            equation_term.append(equivalent_term)

            if idx != self.rpi:
                weights.append(weight)
        weights.append(0)
        return weights, equation_term

    def parse_opt_params(self, params):
        params_parsed = OrderedDict()
        cur_idx = 0
        for term in self.terms:
            # print('params', params)
            params_parsed[term.term_id] = term.parse_opt_params(params[cur_idx: cur_idx + term.opt_params_num()])
            cur_idx += term.opt_params_num()
        return params_parsed

    def set_term_params(self, params):
        params_parsed = self.parse_opt_params(params)
        for term in self.terms:
            term.use_params(params_parsed[term.term_id])

    def evaluate_with_params(self, params):
        self.set_term_params(params)
        if self.rpi < 0:
            val1 = np.add.reduce([term.evaluate() for term_idx, term in enumerate(self.terms) 
                                 if term_idx != len(self.terms) + self.rpi])
        else:
            val1 = np.add.reduce([term.evaluate() for term_idx, term in enumerate(self.terms)
                                 if term_idx != self.rpi])
        val2 = self.terms[self.rpi].evaluate()
        return val1 - val2

    def evaluate_grad(self, params, param_in_term_props):
        param_label = self.terms[param_in_term_props[0]].all_params[param_in_term_props[1]]
        param_grad_vals = 2 * self.evaluate_with_params(params) * \
            self.terms[param_in_term_props[0]].evaluate_grad(param_label).reshape(-1)
        print('------------------------------------------------------------------')
        print(
            f'grad for param {param_label} is {np.linalg.norm(param_grad_vals)}')
        print(
            f'while eq value is {np.linalg.norm(self.evaluate_with_params(params))} and grad of factor is {np.linalg.norm(self.terms[param_in_term_props[0]].evaluate_grad(param_label))}')

        return param_grad_vals

    @property
    def equation(self):
        if self._optimization_held:
            if not self.equation_set:
                self.set_equation()
            return self._equation
        else:
            raise AttributeError(
                'Equation terms have not been initialized before calling.')

    def set_equation(self, rpi: int = -1):
        weights, terms = self.parse_eq_terms()
        self._equation = Equation(pool=self.pool, basic_structure=terms, terms_number=len(terms),
                                  max_factors_in_term=max([len(term.structure) for term in terms]))
        self._equation.target_idx = rpi if rpi >= 0 else len(self._equation.structure) - 1
        self._equation.weights_internal = weights
        self._equation.weights_final = weights
        self.equation_set = True

    @singledispatchmethod
    def get_term_for_param(self, param):
        raise NotImplementedError(
            'The term must be called by parameter index or label')

    @get_term_for_param.register
    def _(self, param: str):
        term_index = [idx for idx, term in enumerate(self.terms) if param in term][0]
        return self.terms[term_index]

    @get_term_for_param.register
    def _(self, param: int):
        return self.terms[self.param_term_beloning[param]]
