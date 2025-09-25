#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 18:48:23 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import partial
from sklearn.linear_model import LinearRegression
import epde.globals as global_var


def generate_partial(obj_function, equation_key):
    """
    Generates a partially applied function, fixing the equation key.
    
        This is useful for creating specialized versions of a function
        that operate on a specific equation derived within the EPDE framework.
        By pre-filling the equation key, subsequent calls to the function
        can focus on other parameters, simplifying the overall workflow
        when dealing with a set of predefined equation structures.
    
        Args:
            obj_function: The function to partially apply.
            equation_key: The equation key to pre-fill.
    
        Returns:
            A partial function with the equation_key argument pre-filled.
    """
    return partial(obj_function, equation_key=equation_key)


def equation_fitness(system, equation_key):
    """
    Evaluate the fitness of a specific equation within the PDE system.
    
        This method retrieves the pre-calculated fitness value for a given equation.
        It's used to assess how well each individual equation in the system contributes
        to the overall model accuracy. By examining individual equation fitness, the
        algorithm can identify and refine the most relevant equations for describing
        the underlying dynamics.
    
        Args:
            system (``epde.structure.main_structures.SoEq``): The system of equations being evaluated.
            equation_key (str): The identifier (key) of the equation within the system to evaluate.
    
        Returns:
            float: The fitness value of the specified equation.
    """
    assert system.vals[equation_key].fitness_calculated, 'Trying to call fitness before its evaluation.'
    res = system.vals[equation_key].fitness_value
    return res


def equation_complexity_by_terms(system, equation_key):
    """
    Evaluate the complexity of a specific equation within the PDE system by counting the number of non-zero terms. This provides a measure of the equation's complexity, which is useful for guiding the equation discovery process by favoring simpler, more parsimonious models.
    
        Args:
            system (``epde.structure.main_structures.SoEq``): The system of equations being analyzed.
            equation_key (int): Index of equation in system
    
        Returns:
            int: The number of terms with non-zero weights in the specified equation.
    """
    return np.count_nonzero(system.vals[equation_key].weights_internal)


def equation_complexity_by_factors(system, equation_key):
    """
    Evaluate the complexity of a single equation within the system based on its constituent factors.
        This function quantifies the equation's complexity by summing the complexities of individual terms,
        considering their structure and the significance of their weights. The complexity is evaluated
        by considering the number of derivatives within each term.
    
        Args:
            system (``epde.structure.main_structures.SoEq``): The system of equations to be evaluated.
            equation_key (int): The index of the equation within the system to evaluate.
    
        Returns:
            int: The complexity score for the specified equation.
    
        Why:
        This method helps to assess the complexity of candidate equation structures, guiding the search
        towards simpler and more interpretable models that still accurately represent the underlying dynamics.
        By penalizing overly complex equations, the method promotes parsimony and reduces the risk of overfitting
        the data.
    """
    # eq_compl = 0

    # for idx, term in enumerate(system.vals[equation_key].structure):
    #     if idx < system.vals[equation_key].target_idx:
    #         if not system.vals[equation_key].weights_final[idx] == 0:
    #             eq_compl += len(term.structure)
    #     elif idx > system.vals[equation_key].target_idx:
    #         if not system.vals[equation_key].weights_final[idx-1] == 0:
    #             eq_compl += len(term.structure)
    #     else:
    #         eq_compl += len(term.structure)
    # return eq_compl
    eq_compl = 0

    for idx, term in enumerate(system.vals[equation_key].structure):
        if idx < system.vals[equation_key].target_idx:
            if not system.vals[equation_key].weights_final[idx] == 0:
                eq_compl += complexity_deriv(term.structure)
        elif idx > system.vals[equation_key].target_idx:
            if not system.vals[equation_key].weights_final[idx-1] == 0:
                eq_compl += complexity_deriv(term.structure)
        else:
            eq_compl += complexity_deriv(term.structure)
    return eq_compl


def equation_terms_stability(system, equation_key):
    """
    Calculates the stability of the coefficients within a given equation.
    
        This function retrieves the pre-calculated stability of the coefficients
        for a specified equation within the system. It relies on the system
        having already computed the stability during the equation discovery
        process. This information is crucial for assessing the reliability and
        robustness of the identified equation structure.
    
        Args:
            system: The system object containing equation values and their
                associated stability metrics.
            equation_key: The key identifying the equation for which to retrieve
                the coefficient stability.
    
        Returns:
            The stability of the coefficients for the specified equation.
    """
    assert system.vals[equation_key].stability_calculated
    res = system.vals[equation_key].coefficients_stability
    return res

def equation_aic(system, equation_key):
    """
    Retrieves the pre-calculated Akaike Information Criterion (AIC) for a specified equation.
    
    This function accesses the stored AIC value, which quantifies the trade-off between the goodness of fit and the complexity of the equation.
    A lower AIC indicates a better balance, aiding in the selection of the most suitable equation form to represent the underlying dynamics.
    
    Args:
        system: The system object holding the equation and its associated results, including the pre-calculated AIC.
        equation_key: The identifier for the equation within the system.
    
    Returns:
        float: The pre-calculated AIC value for the specified equation.
    """
    assert system.vals[equation_key].aic_calculated
    res = system.vals[equation_key].aic
    return res

def complexity_deriv(term_list: list):
    """
    Calculates a complexity score reflecting the number of operations in a symbolic expression.
    
    This function iterates through a list of terms, summing up the complexity
    contributed by each term based on its `deriv_code` attribute, which
    represents the operations involved in the term. The final score is then
    scaled by a 'power' parameter, likely representing the overall influence
    or weight of the expression. This complexity score is used to guide the
    search for simpler and more parsimonious equation structures during the
    equation discovery process.
    
    Args:
        term_list: A list of terms, where each term is expected to have
            a `deriv_code` attribute (representing operations) and a `param`
            method to access parameters like 'power'.
    
    Returns:
        float: The calculated complexity score, scaled by the 'power'
            parameter of the last factor. This score reflects the overall
            complexity of the symbolic expression.
    """
    total = 0
    for factor in term_list:
        if factor.deriv_code == [None]:
            total += 0.5
        elif factor.deriv_code is None:
            total += 0.5
        else:
            total += len(factor.deriv_code)
    return total*factor.param('power')
