#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 17:05:58 2021

@author: mike_ubuntu
"""

import numpy as np
from epde.structure import Term, Equation

def float_convertable(obj):
    try:
        float(obj)
        return True
    except (ValueError, TypeError) as e:
        return False

def translate_equation(text_form, pool):
    parsed_text_form = parse_equation_str(text_form)
    term_list = []; weights = np.empty(len(parsed_text_form) - 1)
    max_factors = 0
    for idx, term in enumerate(parsed_text_form):
        if (any([not float_convertable(elem) for elem in term]) and 
            any([float_convertable(elem) for elem in term])):

            factors = [parse_factor(factor, pool) for factor in term[1:]]
            if len(factors) > max_factors:
                max_factors = len(factors)
            term_list.append(Term(pool, passed_term=factors))
            weights[idx] = float(term[0])
        elif float_convertable(term[0]) and len(term) == 1:
            weights[idx] = float(term[0])
        elif all([not float_convertable(elem) for elem in term]):
            factors = [parse_factor(factor, pool) for factor in term]
            if len(factors) > max_factors:
                max_factors = len(factors)
            term_list.append(Term(pool, passed_term=factors))            

    equation = Equation(pool = pool, basic_structure = term_list, terms_number = len(term_list), 
                        max_factors_in_term = max_factors)
    equation.target_idx = len(term_list) - 1
    equation.weights_internal = weights    
    equation.weights_final = weights
    return equation

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

def parse_factor(factor_form, pool):   # В проект: работы по обрезке сетки, на которых нулевые значения производных
    print(factor_form)
    label_str, params_str = tuple(factor_form.split('{'))
    if not '}' in params_str:
        raise ValueError('Missing brackets, denoting parameters part of factor text form. Possible explanation: passing wrong argument')
    params_str = parse_params_str(params_str.replace('}', ''))
    print(label_str, params_str)
    factor_family = [family for family in pool.families if label_str in family.tokens][0]    
    _, factor = factor_family.create(label = label_str, **params_str)
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
    
