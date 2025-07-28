#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 17:05:58 2021

@author: mike_ubuntu
"""
from typing import Union, List
import numpy as np
from sklearn.linear_model import LinearRegression

from epde.structure.encoding import Chromosome
from epde.structure.main_structures import Term, Equation, SoEq

from functools import singledispatch

def float_convertable(obj):
    try:
        float(obj)
        return True
    except (ValueError, TypeError) as e:
        return False

@singledispatch
def translate_equation(text_form, pool, all_vars):
    raise NotImplementedError(f'Equation shall be translated from {type(text_form)}')

@translate_equation.register
def _(text_form : str, pool, all_vars: List[str], use_pic: bool = False):
    parsed_text_form = parse_equation_str(text_form)
    term_list = []
    weights = np.empty(len(parsed_text_form) - 1)
    max_factors = 0
    for idx, term in enumerate(parsed_text_form):
        if (any([not float_convertable(elem) for elem in term]) and
            any([float_convertable(elem) for elem in term])):
            factors = [parse_factor(factor, pool, all_vars) for factor in term[1:]]
            if len(factors) > max_factors:
                max_factors = len(factors)
            term_list.append(Term(pool, passed_term=factors, collapse_powers=False))
            weights[idx] = float(term[0])
        elif float_convertable(term[0]) and len(term) == 1:
            weights[idx] = float(term[0])
        elif all([not float_convertable(elem) for elem in term]):
            factors = [parse_factor(factor, pool, all_vars) for factor in term]
            if len(factors) > max_factors:
                max_factors = len(factors)
            term_list.append(Term(pool, passed_term=factors, collapse_powers=False))

    metaparameters={'terms_number': {'optimizable': False, 'value': len(term_list)},
                    'max_factors_in_term': {'optimizable': False, 'value': max_factors}}
    for var_key in all_vars:
        metaparameters[('sparsity', var_key)] = {'optimizable': True, 'value': 1.}


    equation = Equation(pool=pool, basic_structure=term_list, var_to_explain = all_vars[0],
                        metaparameters=metaparameters)
    equation.target_idx = len(term_list) - 1
    equation.weights_internal = weights
    equation.weights_final = weights    
    
    system = SoEq(pool = pool, metaparameters=metaparameters)
    system.use_default_multiobjective_function(use_pic = use_pic)
    system.create(passed_equations = [equation,])
    # structure = {'u' : equation}
    # system.vals = Chromosome(structure, params={key: val for key, val in system.metaparameters.items()
    #                                             if val['optimizable']})
    
    return system

@translate_equation.register
def _(text_form : dict, pool, all_vars: List[str], use_pic: bool = False):
    equations = []
    for var_key, eq_text_form in text_form.items(): 
        parsed_text_form = parse_equation_str(eq_text_form)
        term_list = []
        weights = np.empty(len(parsed_text_form) - 1)
        max_factors = 0
        for idx, term in enumerate(parsed_text_form):
            if (any([not float_convertable(elem) for elem in term]) and
                any([float_convertable(elem) for elem in term])):
                factors = [parse_factor(factor, pool, all_vars) for factor in term[1:]]
                if len(factors) > max_factors:
                    max_factors = len(factors)
                term_list.append(Term(pool, passed_term=factors, collapse_powers=False))
                weights[idx] = float(term[0])
            elif float_convertable(term[0]) and len(term) == 1:
                weights[idx] = float(term[0])
            elif all([not float_convertable(elem) for elem in term]):
                factors = [parse_factor(factor, pool, all_vars) for factor in term]
                if len(factors) > max_factors:
                    max_factors = len(factors)
                term_list.append(Term(pool, passed_term=factors, collapse_powers=False))
    
        metaparameters={'terms_number': {'optimizable': False, 'value': len(term_list)},
                        'max_factors_in_term': {'optimizable': False, 'value': max_factors}}
        for var_key in all_vars:
            metaparameters[('sparsity', var_key)] = {'optimizable': True, 'value': 1.}

        equation = Equation(pool = pool, basic_structure = term_list, var_to_explain = var_key,
                            metaparameters = metaparameters)
        equation.target_idx = len(term_list) - 1
        equation.weights_internal = weights
        equation.weights_final = weights
        equations.append(equation)
    
    # structure = {'u' : equation}

    system = SoEq(pool = pool, metaparameters=metaparameters)
    system.use_default_multiobjective_function(use_pic = use_pic)
    system.create(passed_equations = equations)
    # system.vals = Chromosome(structure, params={key: val for key, val in system.metaparameters.items()
    #                                             if val['optimizable']})
    
    return system


def parse_equation_str(text_form):
    '''

    Example input: '0.0 * d^3u/dx2^3{power: 1} * du/dx2{power: 1} + 0.0 * d^3u/dx1^3{power: 1} +
    0.015167810810763344 * d^2u/dx1^2{power: 1} + 0.0 * d^3u/dx2^3{power: 1} + 0.0 * du/dx2{power: 1} +
    4.261009307104081e-07 = d^2u/dx1^2{power: 1} * du/dx1{power: 1}'

    '''
    left, right = text_form.split(' = ')
    left = left.split(' + ')
    for idx in range(len(left)):
        left[idx] = left[idx].split(' * ')
    right = right.split(' * ')
    return left + [right,]


def parse_term_str(term_form):
    pass


def parse_factor(factor_form, pool, all_vars):   # В проект: работы по обрезке сетки, на которых нулевые значения производных
    # print(factor_form)
    label_str, params_str = tuple(factor_form.split('{'))
    if '}' not in params_str:
        raise ValueError('Missing brackets, denoting parameters part of factor text form. Possible explanation: passing wrong argument')
    params_str = parse_params_str(params_str.replace('}', ''))
    # print(label_str, params_str)
    factor_family = [family for family in pool.families if label_str in family.tokens][0]
    _, factor = factor_family.create(label=label_str, all_vars = all_vars, **params_str)
    factor.set_param(param = params_str['power'], name = 'power')
    return factor


def parse_params_str(param_str):
    assert isinstance(param_str, str), 'Passed parameters are not in string format'
    params_split = param_str.split(',')
    params_parsed = dict()
    for param in params_split:
        temp = param.split(':')
        temp[0] = temp[0].replace(' ', '')
        params_parsed[temp[0]] = float(temp[1]) if '.' in temp[1] else int(temp[1])
    return params_parsed

class CoeffLessEquation():
    def __init__(self, lp_terms : Union[list, tuple, dict], rp_term : Union[list, tuple, dict], 
                 pool, all_vars, use_pic: bool = False):
        '''
        ``lp_terms''
        '''
        if isinstance(lp_terms, dict):
            if not len(lp_terms.keys()) == len(rp_term.keys()):
                raise KeyError(f'Number of left parts {lp_terms.keys()} mismatches right parts {rp_term.keys()}.')
            equations = []
            for variable in rp_term.keys():
                lp_terms_translated = [Term(pool, passed_term = [parse_factor(factor, pool, all_vars) for factor in term],
                                            collapse_powers=False) for term in lp_terms[variable]]
                rp_translated = Term(pool, passed_term = [parse_factor(factor, pool, all_vars) for factor in rp_term[variable]], 
                                     collapse_powers=False)
                
                lp_values = np.vstack(list(map(lambda x: x.evaluate(False).reshape(-1), lp_terms_translated)))
                rp_value = rp_translated.evaluate(False).reshape(-1)
                lr = LinearRegression()
                lr.fit(lp_values.T, rp_value)
                # print(lr.coef_, lr.intercept_, type(lr.coef_))
                terms_aggregated = lp_terms_translated + [rp_translated,]
                max_factors = max([len(term.structure) for term in terms_aggregated])

                metaparameters={'terms_number': {'optimizable': False, 'value': len(term_list)},
                                'max_factors_in_term': {'optimizable': False, 'value': max_factors}}
                for var_key in all_vars:
                    metaparameters[('sparsity', var_key)] = {'optimizable': True, 'value': 1.}

                equation = Equation(pool=pool, basic_structure=terms_aggregated,
                                    metaparameters=metaparameters)
                                    # terms_number=len(lp_terms) + 1, max_factors_in_term=max_factors)
                equation.target_idx = len(terms_aggregated) - 1
                equation.weights_internal = np.append(lr.coef_, lr.intercept_)
                equation.weights_final = np.append(lr.coef_, lr.intercept_)
                equations.append(equation)

            self.system = SoEq(pool = pool, metaparameters=metaparameters)
            self.system.use_default_multiobjective_function(use_pic = use_pic)
            self.system.create(equations)

        else:
            self.lp_terms_translated = [Term(pool, passed_term = [parse_factor(factor, pool, all_vars) for factor in term],
                                            collapse_powers=False) for term in lp_terms]
            self.rp_translated = Term(pool, passed_term = [parse_factor(factor, pool, all_vars) for factor in rp_term], 
                                      collapse_powers=False)
            
            self.lp_values = np.vstack(list(map(lambda x: x.evaluate(False).reshape(-1), self.lp_terms_translated)))
            self.rp_value = self.rp_translated.evaluate(False).reshape(-1)
            lr = LinearRegression()
            lr.fit(self.lp_values.T, self.rp_value)
            # print(lr.coef_, lr.intercept_, type(lr.coef_))
            terms_aggregated = self.lp_terms_translated + [self.rp_translated,]
            max_factors = max([len(term.structure) for term in terms_aggregated])

            metaparameters={'terms_number': {'optimizable': False, 'value': len(term_list)},
                            'max_factors_in_term': {'optimizable': False, 'value': max_factors}}
            for var_key in all_vars:
                metaparameters[('sparsity', var_key)] = {'optimizable': True, 'value': 1.}


            self.equation = Equation(pool=pool, basic_structure=terms_aggregated,
                                     metaparameters=metaparameters)
                                    # terms_number=len(lp_terms) + 1, max_factors_in_term=max_factors)
            self.equation.target_idx = len(terms_aggregated) - 1
            self.equation.weights_internal = np.append(lr.coef_, lr.intercept_)
            self.equation.weights_final = np.append(lr.coef_, lr.intercept_)
            self.system = SoEq(pool = pool, metaparameters=metaparameters)
            self.system.use_default_multiobjective_function(use_pic = use_pic)
            self.system.create(equations)
