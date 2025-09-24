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
    """
    Checks if a given object can be interpreted as a floating-point number.
    
    This function is crucial for ensuring that data used in equation discovery is of the correct type, preventing errors during the evaluation of equation fitness. By verifying the convertibility of data points to floats, the system avoids attempting mathematical operations on incompatible data types, which would lead to inaccurate or invalid results in the equation search process.
    
    Args:
        obj: The object to check for float convertibility.
    
    Returns:
        bool: True if the object can be converted to a float without error, False otherwise.
    """
    try:
        float(obj)
        return True
    except (ValueError, TypeError) as e:
        return False

@singledispatch
def translate_equation(text_form, pool, all_vars):
    """
    Translates an equation from a text form into an executable representation.
    
        This is the base implementation that raises a `NotImplementedError`.
        Specific text formats (e.g., SymPy, LaTeX) should have their own
        specialized implementations registered using `singledispatch`.
    
        Args:
            text_form: The equation in a text-based format.
            pool: A pool of resources or variables available for translation.
            all_vars: A collection of all variables used in the equation.
    
        Returns:
            NotImplementedError: Always raises an exception indicating that
              the translation is not implemented for the given text form.
    
        WHY: This base case ensures that all equation types must be explicitly
        handled by a dedicated translation function, preventing unexpected
        behavior when encountering an unsupported format during the equation
        discovery process.
    """
    raise NotImplementedError(f'Equation shall be translated from {type(text_form)}')

@translate_equation.register
def _(text_form : str, pool, all_vars: List[str], use_pic: bool = False):
    """
    Parses a text-based equation into a system of equations suitable for symbolic regression.
    
        This method takes a string representation of an equation, parses it into individual terms,
        and constructs an `Equation` object representing the mathematical relationship. This `Equation`
        is then encapsulated within a `SoEq` (System of Equations) object, which provides a structure
        for further analysis and optimization within the EPDE framework. The parsing process involves
        identifying factors, handling numerical coefficients, and extracting relevant metadata.
    
        Args:
            text_form: The string representation of the equation to be parsed.
            pool: A pool object for parallel processing, enabling efficient computation of equation terms.
            all_vars: A list of all variable names present in the equation, used for creating symbolic
                representations of the terms.
            use_pic: A boolean flag indicating whether to use a default multiobjective function.
    
        Returns:
            SoEq: The created system of equations object, ready for symbolic regression and analysis.
    
        WHY: This method transforms a human-readable equation string into a structured, symbolic representation
        that the EPDE framework can then use to perform symbolic regression, identify governing equations,
        and build predictive models.
    """
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
        metaparameters[('sparsity', var_key)] = {'optimizable': True, 'value': 0.}


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
    """
    Generates a system of equations (SoEq) from a text-based equation form.
    
        This method parses a dictionary of equation strings, converts them into Equation objects,
        and combines them into a SoEq object to represent a complete equation system. It prepares
        the equations for the evolutionary search process by creating metaparameters and setting up
        the multiobjective function, which guides the search towards optimal equation structures.
        This is a crucial step in translating human-readable equations into a format suitable for
        automated discovery and refinement.
    
        Args:
            text_form: A dictionary where keys are variable names and values are equation strings.
            pool: A pool object (likely for parallel processing or resource management).
            all_vars: A list of all variable names used in the equations.
            use_pic: A boolean flag indicating whether to use a default multiobjective function.
    
        Returns:
            SoEq: A system of equations object representing the parsed equations.
    """
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
            metaparameters[('sparsity', var_key)] = {'optimizable': True, 'value': 0.}

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
    Parses a string representation of an equation into a structured format.
    
    This function is essential for transforming symbolic equations into a processable format.
    It splits the equation into left-hand side and right-hand side components,
    further breaking down each side into individual terms. This structured representation
    facilitates subsequent analysis and manipulation of the equation's components.
    
    Args:
        text_form (str): A string representing the equation, e.g.,
            '0.0 * d^3u/dx2^3{power: 1} * du/dx2{power: 1} + 0.0 * d^3u/dx1^3{power: 1} +
            0.015167810810763344 * d^2u/dx1^2{power: 1} + 0.0 * d^3u/dx2^3{power: 1} + 0.0 * du/dx2{power: 1} +
            4.261009307104081e-07 = d^2u/dx1^2{power: 1} * du/dx1{power: 1}'
    
    Returns:
        list: A list containing the terms from the left-hand side and right-hand side of the equation.
              Each side is further split into a list of terms.
    '''
    left, right = text_form.split(' = ')
    left = left.split(' + ')
    for idx in range(len(left)):
        left[idx] = left[idx].split(' * ')
    right = right.split(' * ')
    return left + [right,]


def parse_term_str(term_form):
    """
    Parses a term string into a structured representation suitable for equation discovery.
    
    This function takes a string representing a term in a differential equation and 
    transforms it into a format that the EPDE framework can use for symbolic manipulation 
    and evaluation. The parsed representation facilitates the automated search for 
    equation structures that best fit the observed data.
    
    Args:
        term_form (str): The term string to parse.
    
    Returns:
        None: The function's return type is currently None, but it would ideally return
              a structured representation of the parsed term (e.g., a tree or a list of tokens).
    """
    pass


def parse_factor(factor_form, pool, all_vars):   # В проект: работы по обрезке сетки, на которых нулевые значения производных
    # print(factor_form)
    label_str, params_str = tuple(factor_form.split('{'))
    if '}' not in params_str:
    """
    Parses a factor from its string representation to construct a component of the overall equation.
    
    This method dissects a factor's string representation, extracting its label and parameters.
    It then instantiates a corresponding factor object from a predefined pool of factor families
    and configures its 'power' parameter, effectively defining a term within a larger equation.
    This process is crucial for building equation structures that can be optimized to fit observed data.
    
    Args:
        factor_form (str): The string representation of the factor, e.g., "factor_label{param1=value1,param2=value2}".
        pool (FactorPool): The pool of factor families to search for the appropriate family.
        all_vars (list): A list of all variables.
    
    Returns:
        Factor: The created factor object, representing a component of the equation.
    """
        raise ValueError('Missing brackets, denoting parameters part of factor text form. Possible explanation: passing wrong argument')
    params_str = parse_params_str(params_str.replace('}', ''))
    # print(label_str, params_str)
    factor_family = [family for family in pool.families if label_str in family.tokens][0]
    _, factor = factor_family.create(label=label_str, all_vars = all_vars, **params_str)
    factor.set_param(param = params_str['power'], name = 'power')
    return factor


def parse_params_str(param_str):
    """
    Parses a comma-separated string of parameters into a dictionary to configure equation discovery process.
    
        This function is essential for setting up the parameters of the evolutionary search,
        allowing users to specify configurations like population size, mutation rates, and
        other optimization settings via a string format. The input string should follow
        the format "key1:value1,key2:value2,...", where values are automatically converted
        to integers or floats based on their format.
    
        Args:
            param_str (str): The string containing the parameters.
    
        Returns:
            dict: A dictionary where keys are the parameter names (strings) and values are
                  the corresponding parameter values (integers or floats). These parameters
                  influence the behavior of the equation discovery algorithms.
    """
    assert isinstance(param_str, str), 'Passed parameters are not in string format'
    params_split = param_str.split(',')
    params_parsed = dict()
    for param in params_split:
        temp = param.split(':')
        temp[0] = temp[0].replace(' ', '')
        params_parsed[temp[0]] = float(temp[1]) if '.' in temp[1] else int(temp[1])
    return params_parsed

class CoeffLessEquation():
    """
    Represents a coefficient-less equation in a linear program.
    
        This class is used to store equations where the coefficients are implicitly 1.
    """

    def __init__(self, lp_terms : Union[list, tuple, dict], rp_term : Union[list, tuple, dict], 
                 pool, all_vars, use_pic: bool = False):
        """
        Initializes a `CoeffLessEquation` object, translating symbolic terms into a numerical representation suitable for equation discovery. It prepares the equation for coefficient optimization by structuring the terms and setting up the optimization environment.
        
                Args:
                    lp_terms (Union[list, tuple, dict]): Left-hand side terms of the equation. Can be a list/tuple or a dictionary where keys are variables and values are lists/tuples of terms.
                    rp_term (Union[list, tuple, dict]): Right-hand side term(s) of the equation.  Must match the structure of `lp_terms`.
                    pool: A pool of variables and constants used in the equation.
                    all_vars: A list of all variable names used in the equation.
                    use_pic (bool, optional): A flag to use PIC (physics-informed calibration). Defaults to False.
        
                Returns:
                    None
        
                Why:
                    This initialization process is crucial for converting symbolic representations of equations into a numerical format that can be processed by optimization algorithms. By translating the terms and setting up the optimization environment, the method prepares the equation for coefficient fitting, which is a key step in discovering the underlying differential equation from data.
        """
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
                    metaparameters[('sparsity', var_key)] = {'optimizable': True, 'value': 0.}

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
                metaparameters[('sparsity', var_key)] = {'optimizable': True, 'value': 0.}


            self.equation = Equation(pool=pool, basic_structure=terms_aggregated,
                                     metaparameters=metaparameters)
                                    # terms_number=len(lp_terms) + 1, max_factors_in_term=max_factors)
            self.equation.target_idx = len(terms_aggregated) - 1
            self.equation.weights_internal = np.append(lr.coef_, lr.intercept_)
            self.equation.weights_final = np.append(lr.coef_, lr.intercept_)
            self.system = SoEq(pool = pool, metaparameters=metaparameters)
            self.system.use_default_multiobjective_function(use_pic = use_pic)
            self.system.create(equations)
