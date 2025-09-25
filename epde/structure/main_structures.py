#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:46:45 2022

@author: maslyaev
"""

import gc
import warnings
import copy
import os
import pickle
from typing import Union, Callable, Tuple
from functools import singledispatchmethod, reduce
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


import numpy as np
import torch

import epde.globals as global_var
import epde.optimizers.moeadd.solution_template as moeadd

from epde.decorators import HistoryExtender, BoundaryExclusion
from epde.evaluators import simple_function_evaluator
from epde.interface.token_family import TFPool
from epde.preprocessing.domain_pruning import DomainPruner

from epde.structure.encoding import Chromosome
from epde.structure.factor import Factor
from epde.structure.structure_template import ComplexStructure, check_uniqueness
from epde.supplementary import filter_powers, normalize_ts, population_sort, flatten, rts, exp_form, minmax_normalize


class Term(ComplexStructure):
    """
    Represents a term within a differential equation, encapsulating its structure and properties.
    
    
        Attributes:
            _descr_variable_marker
    
            pool
            max_factors_in_term:
            cache_linked:
            structure:
            occupied_tokens_labels:
            descr_variable_marker:
            prev_normalized
    """

    __slots__ = ['_history', 'structure', 'interelement_operator', 'saved', 'saved_as',
                 'pool', 'max_factors_in_term', 'cache_linked', 'occupied_tokens_labels',
                 '_descr_variable_marker']

    def __init__(self, pool, passed_term=None, mandatory_family=None, max_factors_in_term=1,
                 create_derivs: bool = False, interelement_operator=np.multiply, collapse_powers = True):
        """
        Initializes a Term object, representing a candidate term in the equation search space.
        
                This method constructs a Term object, either by randomly selecting factors
                from a pool or by using a provided term as a starting point. The initialization
                process prepares the term for evaluation and optimization within the
                equation discovery process.
        
                Args:
                    pool: The pool of factors to use for term creation.
                    passed_term: An existing term to use as a basis (optional).
                    mandatory_family: A family that must be present in the term (optional).
                    max_factors_in_term: The maximum number of factors allowed in the term.
                    create_derivs: A boolean indicating whether to create derivatives.
                    interelement_operator: The operator to use between elements (default: np.multiply).
                    collapse_powers: A boolean indicating whether to collapse powers (default: True).
        
                Returns:
                    None
        
                Class Fields:
                    pool: The pool of factors used for term creation.
                    max_factors_in_term: The maximum number of factors allowed in the term.
                    term: The underlying term representation (list of factors).
                    tensor: The tensor representation of the term.
                    is_normalized: A boolean indicating whether the term is normalized.
                    saved_state: A dictionary tracking the saved state of normalization.
        
                Why:
                    The Term object represents a potential building block of a differential equation.
                    This initialization is a crucial step in creating and manipulating these terms
                    during the equation discovery process. The choice between random initialization
                    and using a passed term allows for both exploration of the search space and
                    refinement of existing candidate solutions.
        """
        super().__init__(interelement_operator)
        self.pool = pool
        self.max_factors_in_term = max_factors_in_term

        if passed_term is None:
            self.randomize(mandatory_family=mandatory_family,
                           create_derivs=create_derivs)
        else:
            self.defined(passed_term, collapse_powers = collapse_powers)

        if global_var.tensor_cache is not None:
            self.use_cache()
        # key - state of normalization, value - if the variable is saved in cache
        self.reset_saved_state()

    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        """
        Manually reconstructs a specific attribute of the `Term` object, primarily to define its structure.
        
                This method is used to bypass the standard term creation process, allowing for direct specification of the `Term`'s internal components. This is particularly useful when loading a previously discovered equation or when fine-tuning the equation's structure.
        
                Args:
                    attribute (str): The name of the attribute to reconstruct. Currently, only 'structure' is supported.
                    value (list): The new value to assign to the 'structure' attribute. This should be a list of dictionaries, where each dictionary represents a `Factor` object.
                    except_attrs (dict): A dictionary of attributes to exclude during the reconstruction of `Factor` objects. This allows for selective overriding of attributes during reconstruction.
        
                Returns:
                    None. The method modifies the object's state directly.
        
                Fields:
                    structure (list): A list of `Factor` objects representing the structure of the equation term.
        """
        from epde.loader import attrs_from_dict, get_typespec_attrs
        supported_attrs = ['structure']
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')

        if attribute == supported_attrs[0]:
            # Validate correctness of a term definition
            self.structure = []
            for factor_elem in value:
                factor = Factor.__new__(Factor)

                attrs_from_dict(factor, factor_elem, except_attrs)
                factor.evaluator = self.pool
                self.structure.append(factor)

    @property
    def cache_label(self):
        """
        Caches and returns a label representing the sorted structure of the `Term`.
        
                This property ensures that the order of elements within the `Term`'s structure
                is consistently represented in the cache label. This is crucial for
                equivalence checks, as it guarantees that terms with the same elements,
                regardless of their original order, are treated as identical.
        
                Args:
                    self: The instance of the `Term` class.
        
                Returns:
                    tuple or any: A tuple of cache labels if the structure contains multiple
                    elements (sorted by their cache labels), or the cache label of the
                    single element if the structure has only one element.
        """
        if len(self.structure) > 1:
            structure_sorted = sorted(self.structure, key=lambda x: x.cache_label)
            cache_label = tuple([elem.cache_label for elem in structure_sorted])
        else:
            cache_label = self.structure[0].cache_label
        return cache_label

    def use_cache(self):
        """
        Enables caching for the current term and propagates the setting to its children.
        
                This method sets the `cache_linked` flag to True for the current term,
                indicating that caching should be used. It then iterates through the
                term's structure (children) and recursively calls the `use_cache` method
                on any child term that does not already have caching enabled. This optimization
                is crucial for improving the efficiency of the equation discovery process,
                as it avoids redundant computations of term values during the evolutionary search.
        
                Args:
                    self: The instance of the Term class.
        
                Returns:
                    None.
        
                Initializes:
                    cache_linked (bool): A boolean flag indicating whether caching is enabled for this term.
        """
        self.cache_linked = True
        for idx, _ in enumerate(self.structure):
            if not self.structure[idx].cache_linked:
                self.structure[idx].use_cache()

    # TODO: non-urgent, make self.descr_variable_marker setting for defined parameter

    @singledispatchmethod
    def defined(self, passed_term):
        """
        Abstract method to handle term definitions. It serves as a base for specialized implementations that process different types of terms within the equation discovery process.
        
        Args:
            passed_term: The term to be defined. This term will be processed to construct a symbolic representation of a component within a differential equation.
        
        Raises:
            NotImplementedError: Always raised, as this is the base implementation. It indicates that a specific implementation for the given term type is required.
        
        Returns:
            None: This method always raises an error, signaling the need for a concrete implementation in a subclass.
        """
        raise NotImplementedError(
            f'passed term should have string or list/dict types, not {type(passed_term)}')

    @defined.register
    def _(self, passed_term: list, collapse_powers = True):
        """
        Initializes a Term object, building its structure from a list of factors.
        
                This method processes a list of factors, which can be strings (factor labels) or existing Factor objects,
                to define the term's structure. It also simplifies the term by collapsing powers of identical factors if specified.
                This ensures that the term is represented in its most compact form, which is crucial for efficient equation discovery.
        
                Args:
                    passed_term (list): A list of strings (factor labels) or Factor objects representing the term's structure.
                    collapse_powers (bool, optional): A boolean indicating whether to combine identical factors into powers (e.g., x*x becomes x^2). Defaults to True.
        
                Returns:
                    None
        
                Class Fields:
                    structure (list): A list of Factor objects representing the term's structure after processing the input passed_term.
        """
        self.structure = []
        for _, factor in enumerate(passed_term):
            if isinstance(factor, str):
                _, temp_f = self.pool.create(label=factor)
                self.structure.append(temp_f)
            elif isinstance(factor, Factor):
                self.structure.append(factor)
            else:
                raise ValueError('The structure of a term should be declared with str or factor.Factor obj, instead got', type(factor))
        if collapse_powers:
            self.structure = filter_powers(self.structure)

    @defined.register
    def _(self, passed_term: str, collapse_powers = True):
        """
        Initializes the term structure by parsing a string representation or directly using a `Factor` object. This prepares the term for symbolic manipulation within the equation discovery process.
        
                Args:
                    passed_term (str | Factor): The term to be added, either as a string to be parsed into a `Factor`, or a `Factor` object directly.
                    collapse_powers (bool): A boolean indicating whether to collapse powers (default is True).
        
                Returns:
                    None
        
                Raises:
                    ValueError: If `passed_term` is not a string or a `Factor` object.
        
                Why:
                    This method ensures that each term within an equation is represented as a structured list of `Factor` objects, facilitating subsequent operations such as simplification, differentiation, and evaluation.
        """
        self.structure = []
        if isinstance(passed_term, str):
            _, temp_f = self.pool.create(label=passed_term)
            self.structure.append(temp_f)
        elif isinstance(passed_term, Factor):
            self.structure.append(passed_term)
        else:
            raise ValueError('The structure of a term should be declared with str or factor.Factor obj, instead got', type(passed_term))

    def randomize(self, mandatory_family=None, forbidden_factors=None,
                  create_derivs=False, **kwargs):
        """
        Generates a randomized structure (a list of factors) representing a potential equation term.
        
                This method constructs a term by iteratively creating factors, ensuring
                that the generated structure adheres to token availability and constraints
                defined within the pool. The process can be seeded with a mandatory token
                family to guide the search or initialized randomly. The method ensures
                that the generated term is valid by tracking token usage and filtering
                the final structure. This approach allows exploring diverse equation
                structures while respecting predefined constraints.
        
                Args:
                    mandatory_family: A specific token family that must be included in the
                        initial factor. If None, the initial factor is created randomly.
                    forbidden_factors: A dictionary specifying tokens that are forbidden
                        and their current status. If None, it is initialized based on
                        `self.max_factors_in_term`.
                    create_derivs: A boolean indicating whether derivatives should be
                        created during factor creation.
                    **kwargs: Additional keyword arguments passed to the `pool.create`
                        method.
        
                Returns:
                    list: The randomized structure, which is a list of factors.
        
                Initializes:
                    self.occupied_tokens_labels (dict): A dictionary tracking the status of occupied tokens,
                        initialized based on `forbidden_factors`. Each key is a token label, and the value
                        is a list containing the current count, maximum count, and a boolean indicating
                        whether the token is fully occupied.
                    self.descr_variable_marker: Stores the value of `mandatory_family` if it's provided, otherwise False.
                        Used to mark if the initial factor was created with a mandatory family.
                    self.structure (list): A list of factors representing the generated structure.
                        Initialized with the first factor and subsequently appended to in the loop.
        """
        if np.sum(self.pool.families_cardinality(meaningful_only=True)) == 0:
            raise ValueError('No token families are declared as meaningful for the process of the system search')

        def update_token_status(token_status, changes):
            for key, value in changes.items():
                token_status[key][0] += value
                if token_status[key][0] >= token_status[key][1]:
                    token_status[key][2] = True
                else:
                    token_status[key][2] = False
            return token_status

        if forbidden_factors is None:
            forbidden_factors = {}
            for family in self.pool.labels_overview:
                for token_label in family[0]:
                    if isinstance(self.max_factors_in_term, int):
                        forbidden_factors[token_label] = [0, min(self.max_factors_in_term, family[1]), False]
                    elif isinstance(self.max_factors_in_term, dict) and 'probas' in self.max_factors_in_term.keys():
                        forbidden_factors[token_label] = [0, min(self.max_factors_in_term['factors_num'][-1], family[1]),
                                                          False]

        if isinstance(self.max_factors_in_term, int):
            factors_num = np.random.randint(1, self.max_factors_in_term + 1)
        elif isinstance(self.max_factors_in_term, dict) and 'probas' in self.max_factors_in_term.keys():
            factors_num = np.random.choice(a=self.max_factors_in_term['factors_num'],
                                           p=self.max_factors_in_term['probas'])
        else:
            raise ValueError('Incorrect value of max_factors_in_term metaparameters')

        self.occupied_tokens_labels = copy.copy(forbidden_factors)

        self.descr_variable_marker = mandatory_family if mandatory_family is not None else False

        if not mandatory_family:
            occupied_by_factor, factor = self.pool.create(label=None, create_meaningful=True,
                                                          token_status=self.occupied_tokens_labels,
                                                          create_derivs=create_derivs, **kwargs)
        else:
            occupied_by_factor, factor = self.pool.create_with_var(variable=mandatory_family,
                                                                   token_status=self.occupied_tokens_labels,
                                                                   create_derivs=create_derivs,
                                                                   **kwargs)
        self.structure = [factor,]
        update_token_status(self.occupied_tokens_labels, occupied_by_factor)

        for i in np.arange(1, factors_num):
            occupied_by_factor, factor = self.pool.create(label=None, create_meaningful=False,
                                                          token_status=self.occupied_tokens_labels,
                                                          **kwargs)

            update_token_status(self.occupied_tokens_labels, occupied_by_factor)
            self.structure.append(factor)
        self.structure = filter_powers(self.structure)

    @property
    def descr_variable_marker(self):
        """
        Gets the description variable marker. This marker is used to identify the description variable within the equation's symbolic representation, enabling the framework to correctly interpret and process the equation during the equation discovery process.
        
                Returns:
                    str: The description variable marker.
        """
        return self._descr_variable_marker

    @descr_variable_marker.setter
    def descr_variable_marker(self, marker: False):
        """
        Sets the marker used to identify described variables within a symbolic expression.
        
                This marker is used to distinguish variables that have been explicitly described or defined,
                allowing the system to handle them differently during the equation discovery process.
                It ensures that the framework can correctly interpret and manipulate these variables
                when constructing and evaluating candidate equations.
        
                Args:
                    marker: The marker to set. Must be `False` or a string representing a family label (e.g., "u").
        
                Returns:
                    None.
        
                Raises:
                    ValueError: If the provided marker is not `False` or a string.
        
                Class Fields:
                    _descr_variable_marker (str or False): The marker used for described variables.
        """
        if not marker or isinstance(marker, str):
            self._descr_variable_marker = marker
        else:
            raise ValueError('Described variable marker shall be a family label (i.e. "u") of "False"')

    def evaluate(self, structural, grids=None):
        """
        Evaluates the term, potentially normalizing its structure, and leverages a tensor cache for efficiency.
        
                This method checks if the term's value is already computed and stored in a global tensor cache.
                If so, it retrieves the value directly. Otherwise, it computes the value, normalizes it based on the `structural` flag,
                and stores it in the cache for future use. Normalization ensures consistent scaling of terms,
                which is crucial when combining them to form more complex equation structures.
        
                Args:
                    structural (bool): A flag indicating whether to normalize the structure. Normalization helps in maintaining consistent scales across different terms.
                    grids (object, optional): Optional grids data (currently unused).
        
                Returns:
                    np.ndarray: The evaluated value as a flattened NumPy array. The shape is reshaped to 1-D array
        """
        assert global_var.tensor_cache is not None, 'Currently working only with connected cache'
        normalize = structural
        if self.saved[structural] or (self.cache_label, normalize) in global_var.tensor_cache:
            value = global_var.tensor_cache.get(self.cache_label, normalized=normalize,
                                                saved_as=self.saved_as[normalize])
            value = value.reshape(-1)
            return value
        else:
            self.prev_normalized = normalize
            value = super().evaluate(structural)
            if normalize:
                value = np.ones_like(value)
                if np.ndim(value) != 1:
                    for factor in self.structure:
                        temp = factor.evaluate()
                        # value *= normalize_ts(temp)
                        value *= minmax_normalize(temp)
                        # value *= factor.evaluate(structural)
                    # else:
                    #     # value = normalize_ts(value)
                    #     value = minmax_normalize(value)
                else:
                    # if np.std(value) != 0:
                    #     value = (value - np.mean(value)) / np.std(value)
                    # else:
                    #     value = (value - np.mean(value))
                    for factor in self.structure:
                        temp = factor.evaluate()
                        # value *= normalize_ts(temp)
                        value *= (temp - np.mean(temp) - np.min(temp)) / (np.max(temp) - np.min(temp))
            if np.all([len(factor.params) == 1 for factor in self.structure]) and grids is None:
                # Место возможных проблем: сохранение/загрузка нормализованных данных
                self.saved[normalize] = global_var.tensor_cache.add(self.cache_label, value, normalized=normalize)
                if self.saved[normalize]:
                    self.saved_as[normalize] = self.cache_label
            value = value.reshape(-1)
            return value

    def filter_tokens_by_right_part(self, reference_target, equation, equation_position):
        """
        Filters the tokens of a term to ensure its uniqueness within the equation.
        
                This method refines the term's structure by iteratively modifying tokens that might cause conflicts with other parts of the equation. It aims to generate a unique term by replacing problematic tokens until a suitable solution is found or a maximum number of attempts is reached. This process is crucial for maintaining the integrity and identifiability of terms within the equation discovery process.
        
                Args:
                    reference_target: The reference target object providing context for token uniqueness.
                    equation: The equation object to which the term belongs.
                    equation_position: The position of the term within the equation's structure.
        
                Returns:
                    None. The method modifies the term's structure in place to ensure uniqueness.
        """
        warnings.warn(message='Tokens can no longer be set as right-part-unique',
                      category=DeprecationWarning)
        taken_tokens = [factor.label for factor in reference_target.structure
			 if factor.status['unique_for_right_part']]
        meaningful_taken = any([factor.status['meaningful'] for factor in reference_target.structure
                                if factor.status['unique_for_right_part']])

        accept_term_try = 0
        while True:
            accept_term_try += 1
            new_term = copy.deepcopy(self)
            for factor_idx, factor in enumerate(new_term.structure):
                if factor.label in taken_tokens:
                    new_term.reset_occupied_tokens()
                    _, new_term.structure[factor_idx] = self.pool.create(create_meaningful=meaningful_taken,
                                                                         occupied=new_term.occupied_tokens_labels + taken_tokens)
            if check_uniqueness(new_term, equation.structure[:equation_position] +
                                equation.structure[equation_position + 1:]):
                self.structure = new_term.structure
                self.structure = filter_powers(self.structure)
                self.reset_saved_state()
                break
            if accept_term_try == 10 and global_var.verbose.show_warnings:
                warnings.warn('Can not create unique term, while filtering equation tokens in regards to the right part.')
            if accept_term_try >= 10:
                self.randomize(forbidden_factors=new_term.occupied_tokens_labels + taken_tokens)
            if accept_term_try == 100:
                print('Something wrong with the random generation of term while running "filter_tokens_by_right_part"')
                print('proposed', new_term.name, 'for ', equation.text_form, 'with respect to', reference_target.name)

    def reset_occupied_tokens(self):
        """
        Resets the list of occupied token labels based on the current structure and token pool.
        
                This method iterates through the factors in the current equation structure and the available token families.
                It identifies tokens that represent unique elements within the equation or unique instances of specific tokens
                and updates the list of occupied tokens accordingly. This ensures that the evolutionary process considers
                only valid and meaningful combinations of tokens when constructing new equation structures.
        
                Args:
                    self: The instance of the Term class.
        
                Returns:
                    None. The method updates the `occupied_tokens_labels` attribute of the object, reflecting the tokens currently in use.
        """
        occupied_tokens_new = []
        for factor in self.structure:
            for token_family in self.pool.families:
                if factor in token_family.tokens and factor.status['unique_token_type']:
                    occupied_tokens_new.extend(
                        [token for token in token_family.tokens])
                elif factor.status['unique_specific_token']:
                    occupied_tokens_new.append(factor.label)
        self.occupied_tokens_labels = occupied_tokens_new

    @property
    def available_tokens(self):
        """
        Identify and return token families that have at least one token available for use.
        
        This method iterates through the token families in the pool and checks if any of their tokens are not currently occupied.
        If a token family has available tokens, a copy of that family is created, containing only the available tokens, and added to the list of available token families.
        This ensures that only usable tokens are considered during the equation discovery process.
        
        Args:
            None
        
        Returns:
            list: A list of token families with at least one available token.
        
        Class Fields:
            pool (Pool): The pool of token families.
            occupied_tokens_labels (set): A set of labels that are currently occupied.
        """
        available_tokens = []
        for token in self.pool.families:
            if not all([label in self.occupied_tokens_labels for label in token.tokens]):
                token_new = copy.deepcopy(token)
                token_new.tokens = [
                    label for label in token.tokens if label not in self.occupied_tokens_labels]
                available_tokens.append(token_new)
        return available_tokens

    @property
    def total_params(self):
        """
        Calculates the total number of trainable parameters within the term's structure.
        
        This property computes the sum of parameters across all elements in the term's structure,
        representing the complexity of the term. It ensures that the returned value is at least 1,
        preventing issues in subsequent calculations or interpretations where a non-positive number of parameters might be problematic.
        
        Args:
            None
        
        Returns:
            int: The total number of trainable parameters in the term's structure, with a minimum value of 1.
        """
        return max(sum([len(element.params) - 1 for element in self.structure]), 1)

    @property
    def name(self):
        """
        Returns the formatted name of the term.
        
        The name is constructed by concatenating the names of the tokens
        in the term's structure, separated by ' * '. This provides a human-readable representation of the term,
        making it easier to interpret and analyze the discovered equation.
        
        Args:
            self: The term instance.
        
        Returns:
            str: The formatted name of the term.
        """
        form = ''
        for token_idx in range(len(self.structure)):
            form += self.structure[token_idx].name
            if token_idx < len(self.structure) - 1:
                form += ' * '
        return form

    @property
    def latex_form(self):
        """
        Returns the latex form of the term's structure.
        
                This representation is crucial for displaying and interpreting the discovered equation.
                It ensures that the equation structure is easily understandable in mathematical notation.
        
                Returns:
                    str: The latex form of the term.
        """
        form = reduce(lambda x, y: x + r' \cdot ' + y, [factor.latex_name for
                                                        factor in self.structure])
        return form

    def contains_deriv(self, variable=None):
        """
        Checks if the term contains a derivative factor suitable for symbolic manipulation.
        
                This check is crucial for determining if the term can be further simplified or processed within the equation discovery workflow. It specifically looks for derivative factors that are compatible with the symbolic evaluation methods used in the project.
        
                Args:
                    variable: The variable to check for derivatives with respect to. If None, checks for any derivative factor.
        
                Returns:
                    bool: True if the term contains a derivative factor (with respect to the specified variable, if provided) and is compatible with symbolic evaluation, False otherwise.
        """
        if variable is None:
            return any([factor.is_deriv and factor.deriv_code != [None,] and
                        factor.evaluator._evaluator == simple_function_evaluator
                        for factor in self.structure])
        else:
            return any([factor.variable == variable and factor.deriv_code != [None,] and
                        factor.evaluator._evaluator == simple_function_evaluator
                        for factor in self.structure])

    def contains_variable(self, variable):
        """
        Checks if any factor within the term's structure involves a specific variable.
        
        This is crucial for determining the term's dependence on that variable, 
        which is essential for constructing the overall differential equation.
        
        Args:
            variable: The variable to check for within the term's factors.
        
        Returns:
            bool: True if the variable is found in any of the term's factors, False otherwise.
        """
        return any([factor.variable == variable for factor in self.structure])

    def __eq__(self, other):
        """
        Compares this term with another object to determine structural equality.
        
                This method verifies if the other object represents the same mathematical term,
                irrespective of the order of its components. This is crucial for identifying
                equivalent expressions during the equation discovery process.
        
                Args:
                    other: The object to compare with.
        
                Returns:
                    bool: True if the objects have the same structure and length, False otherwise.
        """
        return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure])
                and all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure])
                and len(other.structure) == len(self.structure))

    @HistoryExtender('\n -> was copied by deepcopy(self)', 'n')
    def __deepcopy__(self, memo=None):
        """
        Creates a deep copy of the Term object.
        
                This method ensures that when a Term object is copied, all its attributes
                are also copied independently, preventing modifications to the copy from
                affecting the original. This is crucial for maintaining the integrity of
                equation structures during evolutionary operations.
        
                Args:
                    memo (dict, optional): A dictionary used by `copy.deepcopy` to keep track
                        of already copied objects, preventing infinite recursion in case of
                        circular references. Defaults to None.
        
                Returns:
                    Term: A new Term object that is a deep copy of the original.
        """
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct

        attrs_to_avoid_copy = []
        for k in self.__slots__:
            try:
                if k not in attrs_to_avoid_copy:
                    if not isinstance(k, list):
                        setattr(new_struct, k, copy.deepcopy(
                            getattr(self, k), memo))
                    else:
                        temp = []
                        for elem in getattr(self, k):
                            temp.append(copy.deepcopy(elem, memo))
                        setattr(new_struct, k, temp)
                else:
                    setattr(new_struct, k, None)
            except AttributeError:
                pass

        return new_struct


class Equation(ComplexStructure):
    """
    Class for the single equation for the dynamic system.
    
                Attributes:
                    structure : list of Term objects
                        List, containing all terms of the equation; first 2 terms are reserved for constant value and the input function;
    
                    target_idx : int
                        Index of the target term, selected in the Split phase;
    
                    target : 1-d array of float
                        values of the Term object, reshaped into 1-d array, designated as target for application in sparse regression;
    
                    features : matrix of float
                        matrix, composed of terms, not included in target, value columns, designated as features for application in sparse regression;
    
                    fitness_value : float
                        Inverse value of squared error for the selected target 2function and features and discovered weights;
    
                    estimator : sklearn estimator of selected type
    """

    __slots__ = ['_history', 'structure', 'interelement_operator', 'n_immutable', 'pool',
                  # '_target', '_features', 'saved', 'saved_as','max_factors_in_term', 'operator',
                 'target_idx', 'right_part_selected', '_weights_final', 'weights_final_evald', 'simplified',
                 '_weights_internal', 'weights_internal_evald', 'fitness_calculated', 'stability_calculated', 'aic_calculated', 'solver_form_defined',
                 '_fitness_value', '_coefficients_stability', '_aic', 'metaparameters', 'main_var_to_explain'] # , '_solver_form'


    def __init__(self, pool: TFPool, basic_structure: Union[list, tuple, set], var_to_explain: str = None,
                 metaparameters: dict = {'sparsity': {'optimizable': True, 'value': 1.},
                                         'terms_number': {'optimizable': False, 'value': 5.},
                                         'max_factors_in_term': {'optimizable': False, 'value': 1.}},
                 interelement_operator: Callable = np.add):
        """
        Initializes an Equation object, representing a single equation within a dynamic system.
        
        This class encapsulates the structure and parameters of an equation, facilitating its evolution and optimization within the EPDE framework.
        The initialization process involves setting up the basic terms of the equation and generating additional terms to explore the solution space.
        The goal is to create a flexible representation that can be refined through evolutionary algorithms to accurately model the underlying dynamics of the system.
        
        Args:
            pool (TFPool): A pool of symbolic tokens (functions, derivatives) used to construct the equation terms.
            basic_structure (Union[list, tuple, set]): A collection of initial terms (either Term objects or symbolic strings) that form the basis of the equation. These terms are considered immutable during the evolutionary process.
            var_to_explain (str, optional): A variable that the equation should primarily explain. Defaults to None.
            metaparameters (dict, optional): A dictionary containing metaparameters that control the equation's complexity and optimization. Includes 'sparsity', 'terms_number', and 'max_factors_in_term'. Defaults to a predefined dictionary.
            interelement_operator (Callable, optional): The operator used to combine the terms of the equation (e.g., np.add). Defaults to np.add.
        
        Returns:
            None
        """
        super().__init__(interelement_operator)
        self.reset_state()

        self.n_immutable = len(basic_structure)
        self.pool = pool
        self.structure = []
        self.metaparameters = metaparameters
        if (self.metaparameters['terms_number']['value'] < self.n_immutable):
            raise ValueError(
                'Maximum number of terms parameter is lower, than number of passed basic terms.')

        for passed_term in basic_structure:
            if isinstance(passed_term, Term):
                self.structure.append(passed_term)
            elif isinstance(passed_term, str):
                self.structure.append(Term(self.pool, passed_term=passed_term,
                                           max_factors_in_term=self.metaparameters['max_factors_in_term']['value']))

        self.main_var_to_explain = var_to_explain

        force_var_to_explain = True   # False
        for i in range(len(basic_structure), self.metaparameters['terms_number']['value']):
            check_test = 0
            while True:
                check_test += 1
                mf = var_to_explain if force_var_to_explain else None
                new_term = Term(self.pool, max_factors_in_term=self.metaparameters['max_factors_in_term']['value'],
                                mandatory_family=mf, passed_term=None)

                if check_uniqueness(new_term, self.structure):
                    force_var_to_explain = False
                    break

            self.structure.append(new_term)

        for idx, _ in enumerate(self.structure):
            self.structure[idx].use_cache()
#        self.coefficients_stability = np.inf

    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        """
        Manually reconstructs a specific attribute of the equation.
        
                This method allows for direct assignment of a new value to the specified attribute,
                bypassing the typical attribute setting process. It is used to rebuild equation's structure
                from the scratch, for example, after crossover or mutation operations during evolutionary search.
                
                Args:
                    attribute: The name of the attribute to reconstruct. Currently, only 'structure' is supported.
                    value: The new value to assign to the specified attribute. For 'structure', this should be a list of dictionaries,
                        where each dictionary represents a term.
                    except_attrs: A dictionary of attributes to exclude during the reconstruction process.
                
                Returns:
                    None.
        """
        from epde.loader import attrs_from_dict, get_typespec_attrs
        supported_attrs = ['structure']
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')

        if attribute == supported_attrs[0]:
            # Validate correctness of a term definition
            self.structure = []
            for term_elem in value:
                term = Term.__new__(Term)
                # except_attr, _ = get_typespec_attrs(term)

                attrs_from_dict(term, term_elem, except_attrs)
                self.structure.append(term)

    def reset_explaining_term(self, term_idx=0):
        """
        Resets the term designated to explain the target variable within the equation.
        
                This method iterates through the equation's terms, assigning the target variable as the descriptor
                to the specified term while ensuring that only terms containing the target variable can be assigned.
                All other terms are marked as not contributing to the explanation of the target variable.
        
                Args:
                    term_idx (int): The index of the term to designate as the primary descriptor of the target variable. Defaults to 0.
        
                Returns:
                    None
        
                Raises:
                    AssertionError: If the selected term does not contain the target variable,
                                    preventing it from being a valid descriptor.
        
                Why:
                    This ensures that the equation structure correctly reflects the relationships
                    between terms and the variable being explained, which is crucial for the
                    equation discovery process.
        """
        for idx, term in enumerate(self.structure):
            if idx == term_idx:
                assert term.contains_variable(
                    self.main_var_to_explain), f'Trying explain a variable {self.main_var_to_explain} \
                                                 with term without right family.'
                term.descr_variable_marker = self.main_var_to_explain
            else:
                term.descr_variable_marker = False

    def __eq__(self, other):
        """
        Compares two `Equation` objects for equality.
        
                This method checks if two `Equation` objects are structurally equivalent and,
                if both have their final weights evaluated, compares their final weights as well.
                Structural equivalence is determined by ensuring that both equations contain
                the same terms, regardless of their order. Comparing final weights ensures
                that the equations not only have the same structure but also represent the
                same quantitative relationship. This is crucial for determining if two independently
                evolved equations represent the same underlying model.
        
                Args:
                    other: The `Equation` object to compare with.
        
                Returns:
                    bool: True if the `Equation` objects are equal, False otherwise.
                        Equality is determined by comparing the structure and, if
                        both objects have final weights evaluated, by comparing the
                        final weights as well.
        """
        if self.weights_final_evald and other.weights_final_evald:
            return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure])
                    and all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure])
                    and len(other.structure) == len(self.structure)
                    and np.all(np.isclose(self.weights_final, other.weights_final)))
        else:
            return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure])
                    and all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure])
                    and len(other.structure) == len(self.structure))

    def contains_deriv(self, variable=None):
        """
        Checks if the equation contains any derivative terms.
        
        This is crucial for determining the equation's complexity and
        suitability for fitting observed data, as equations with derivatives
        require specific numerical methods for evaluation and solution.
        
        Args:
            variable: The variable to check for in the derivative.
                      If None, checks for any derivative.
        
        Returns:
            bool: True if any term contains a derivative (optionally of the
                  specified variable), False otherwise.
        """
        return any([term.contains_deriv(variable) for term in self.structure])

    def contains_variable(self, variable):
        """
        Checks if any term within the equation's structure involves a specific variable.
        
        This is crucial for determining the equation's dependence on particular variables,
        which is a key step in the equation discovery process. By identifying which variables
        are present, the search space for potential equation structures can be effectively narrowed.
        
        Args:
            variable: The variable to check for within the equation's terms.
        
        Returns:
            bool: True if the variable is found in any of the equation's terms, False otherwise.
        """
        return any([term.contains_variable(variable) for term in self.structure])

    @property
    def forbidden_token_labels(self):
        """
        Return the set of token labels that cannot be used in the equation's target expression.
        
                This method identifies and returns a set of token labels that are
                considered "forbidden" based on the structure and status of token
                families within the pool. It issues a deprecation warning as tokens
                can no longer be set as right-part-unique. This is done to prevent
                the evolutionary search from generating equations that are structurally
                invalid or lead to poor model performance.
        
                Args:
                    self: The instance of the Equation class.
        
                Returns:
                    set: A set containing the forbidden token labels.
        """
        warnings.warn(message='Tokens can no longer be set as right-part-unique',
                      category=DeprecationWarning)
        target_symbolic = [
            factor.label for factor in self.structure[self.target_idx].structure]
        forbidden_tokens = set()

        for token_family in self.pool.families:
            for token in token_family.tokens:
                if token in target_symbolic and token_family.status['unique_for_right_part']:
                    forbidden_tokens.add(token)
        return forbidden_tokens

    def restore_property(self, deriv: bool = False, mandatory_family: bool = False):
        """
        Restores a property within the equation's structure to maintain validity.
        
                This method attempts to restore either a derivative or a mandatory family
                property within the equation's structure. It replaces a randomly selected
                element in the structure with a new term that satisfies the specified
                property requirements. This is crucial for maintaining the equation's
                integrity during operations like mutation or crossover, ensuring that
                the generated equations remain physically meaningful and well-defined.
        
                Args:
                  deriv: A boolean indicating whether to restore a derivative property.
                  mandatory_family: A boolean indicating whether to restore a mandatory
                    family property.
        
                Returns:
                  None. The method modifies the object's structure in place.
        """
        # TODO: non-urgent, rewrite for an arbitrary equation property check
        if not (deriv or mandatory_family):
            raise ValueError('No property passed for restoration.')
        while True:
            print(
                f'Restoring containment of {mandatory_family} in {self.text_form}.')
            replacement_idx = np.random.randint(low=0, high=len(self.structure))
            mf_marker = mandatory_family if mandatory_family else None
            temp = Term(self.pool, mandatory_family=mf_marker,
                        max_factors_in_term=self.metaparameters['max_factors_in_term']['value'])
            if deriv and mandatory_family and temp.contains_deriv() and temp.contains_variable(self.main_var_to_explain):
                self.structure[replacement_idx] = temp
                break
            elif deriv and temp.contains_deriv() and not mandatory_family:
                self.structure[replacement_idx] = temp
                break
            elif mandatory_family and temp.contains_variable(self.main_var_to_explain) and not deriv:
                self.structure[replacement_idx] = temp
                break
            else:
                print('temp', temp.name, 'self.main_var_to_explain',
                      self.main_var_to_explain)

    def reconstruct_by_right_part(self, right_part_idx):
        """
        Reconstructs the equation to emphasize a specific term, enhancing equation discovery.
        
                This method creates a modified copy of the equation, focusing on the term
                specified by `right_part_idx`. It filters tokens in other terms based on
                the factors present in the focused term. This process helps to isolate
                and refine the relationships captured within that specific term,
                facilitating the identification of relevant equation structures. This is done to simplify equation discovery by focusing search on particular equation terms.
        
                Args:
                    right_part_idx (int): The index of the term to reconstruct around.
        
                Returns:
                    Equation: A new equation object reconstructed to emphasize the specified term.
        """
        warnings.warn(message='Tokens can no longer be set as right-part-unique',
                      category=DeprecationWarning)
        new_eq = copy.deepcopy(self)
        self.copy_properties_to(new_eq)
        new_eq.target_idx = right_part_idx
        if any([factor.status['unique_for_right_part'] for factor in new_eq.structure[right_part_idx].structure]):
            for term_idx, term in enumerate(new_eq.structure):
                if term_idx != right_part_idx:
                    term.filter_tokens_by_right_part(
                        new_eq.structure[right_part_idx], self, term_idx)

        new_eq.reset_saved_state()
        return new_eq

    def evaluate(self, normalize=True, return_val=False, grids=None):
        """
        Evaluates the equation model against the data and returns the predicted values, target, and features.
        
                This method is central to assessing how well the discovered equation represents the underlying relationships in the data. It calculates the model's output based on the learned structure and weights, allowing for comparison with the actual target values.
        
                Args:
                  normalize: A boolean indicating whether to normalize the features before evaluation. Normalization can improve the stability and accuracy of the evaluation, especially when dealing with features on different scales.
                  return_val: A boolean indicating whether to return the predicted values. If True, the method returns the model's predictions; otherwise, it returns None for the predicted values.
                  grids: Grids used for evaluation.
        
                Returns:
                  A tuple containing:
                    - value: The predicted values if `return_val` is True, otherwise None. These values represent the model's output based on the input features and learned equation structure.
                    - target: The target values from the data, representing the actual values the model is trying to predict.
                    - features: The feature values used as input to the model. These are the independent variables used to predict the target variable.
        """
        target = self.structure[self.target_idx].evaluate(normalize, grids=grids)

        # Place for improvent: introduce shifted_idx where necessary
        def shifted_idx(idx):
            if idx < self.target_idx:
                return idx
            elif idx > self.target_idx:
                return idx - 1
            else:
                return -1

        if normalize:
            feature_indexes = list(range(len(self.structure)))
            feature_indexes.remove(self.target_idx)
        else:
            feature_indexes = [idx for idx in range(len(self.structure))
                               if self.weights_internal[shifted_idx(idx)] != 0 and idx != self.target_idx]
        if len(feature_indexes) > 0:
            for feat_idx in range(len(feature_indexes)):
                if feat_idx == 0:
                    features = self.structure[feature_indexes[feat_idx]].evaluate(normalize, grids=grids)
                else:
                    temp = self.structure[feature_indexes[feat_idx]].evaluate(normalize, grids=grids)
                    features = np.vstack([features, temp])

            if features.ndim == 1:
                features = np.expand_dims(features, 1).T
            temp_feats = np.vstack([features, np.ones(features.shape[1])])
            features = np.transpose(features)
            temp_feats = np.transpose(temp_feats)
        else:
            features = None

        if return_val:
            self.prev_normalized = normalize
            if normalize:
                elem1 = np.expand_dims(target, axis=1)
                value = np.add(elem1, - reduce(lambda x, y: np.add(x, y), [np.multiply(self.weights_internal[idx_full], temp_feats[:, idx_sparse])
                                                                           for idx_sparse, idx_full in enumerate(feature_indexes)]))
                                                                           # for feature_idx, weight in np.ndenumerate(self.weights_internal)]))
            else:
                elem1 = np.expand_dims(target, axis=1)
                if features is not None:
                    features_val = reduce(lambda x, y: np.add(x, y), [np.multiply(self.weights_final[idx_full], temp_feats[:, idx_sparse])
                                                                      for idx_sparse, idx_full in enumerate(feature_indexes)]) # Possible mistake here
                    features_val = np.expand_dims(features_val, axis=1)
                else:
                    features_val = np.zeros_like(target)
                value = np.add(elem1, - features_val)
                # print(value.shape)
            return value, target, features
        else:
            return None, target, features

    def reset_state(self, reset_right_part: bool = True):
        """
        Resets the internal state of the equation.
        
                This method is crucial for ensuring a clean slate when re-evaluating or
                re-fitting the equation, particularly after modifications or during
                the evolutionary search process. It resets flags associated with
                different stages of the equation's processing, such as weight evaluation,
                fitness calculation, and simplification.
        
                Args:
                    reset_right_part: Whether to also reset the flag indicating if the
                        right-hand side of the equation has been selected. Defaults to True.
        
                Returns:
                    None
        
                Class fields (object properties) that are initialized:
                    right_part_selected (bool): Indicates whether the right part is selected.
                    weights_internal_evald (bool): Indicates whether internal weights have been evaluated.
                    weights_final_evald (bool): Indicates whether final weights have been evaluated.
                    fitness_calculated (bool): Indicates whether fitness has been calculated.
                    stability_calculated (bool): Indicates whether stability has been calculated.
                    aic_calculated (bool): Indicates whether AIC has been calculated.
                    simplified (bool): Indicates whether the object has been simplified.
                    solver_form_defined (bool): Indicates whether the solver form has been defined.
        """
        if reset_right_part:
            self.right_part_selected = False
        self.weights_internal_evald = False
        self.weights_final_evald = False
        self.fitness_calculated = False
        self.stability_calculated = False
        self.aic_calculated = False
        self.simplified = False
        self.solver_form_defined = False

    @HistoryExtender('\n -> was copied by deepcopy(self)', 'n')
    def __deepcopy__(self, memo=None):
        """
        Creates a deep copy of the `Equation` object.
        
                This method ensures that when an equation is duplicated, all its components,
                including tokens and parameters, are independently copied. This prevents
                unintended modifications to the original equation during the evolutionary
                process of equation discovery. By creating a new instance of the class and
                copying all attributes from the original object to the new object using
                `copy.deepcopy`, the method guarantees that the copied equation is a
                distinct entity.
        
                Args:
                    memo (dict, optional): A dictionary used by `copy.deepcopy` to keep track
                        of objects that have already been copied during the deep copy process.
                        This prevents infinite recursion when copying objects that contain
                        circular references. Defaults to None.
        
                Returns:
                    Equation: A deep copy of the `Equation` object.
        """
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct

        attrs_to_avoid_copy = []
        for k in self.__slots__:
            try:
                if k not in attrs_to_avoid_copy:
                    if not isinstance(k, list):
                        setattr(new_struct, k, copy.deepcopy(getattr(self, k), memo))
                    else:
                        temp = []
                        for elem in getattr(self, k):
                            temp.append(copy.deepcopy(elem, memo))
                        setattr(new_struct, k, temp)
                else:
                    setattr(new_struct, k, None)
            except AttributeError:
                pass

        return new_struct

    def copy_properties_to(self, new_equation):
        """
        Copies essential equation attributes to a new equation instance.
        
                This method transfers flags related to equation evaluation status,
                selection, simplification and calculated values, ensuring the new equation
                inherits the relevant state of the original. This is crucial when
                generating populations of equations and evolving them, as it preserves
                the progress made in evaluating and refining individual equations.
                The solver form is reset to `False` to ensure that the new equation
                will be properly processed during evolution.
                It handles potential AttributeError exceptions when copying
                specific properties that might not exist in either equation.
        
                Args:
                    new_equation: The equation object to copy properties to.
        
                Returns:
                    None
        """
        new_equation.weights_internal_evald = self.weights_internal_evald
        new_equation.weights_final_evald = self.weights_final_evald
        new_equation.right_part_selected = self.right_part_selected
        new_equation.fitness_calculated = self.fitness_calculated
        new_equation.stability_calculated = self.stability_calculated
        new_equation.aic_calculated = self.aic_calculated
        new_equation.simplified = self.simplified
        new_equation.solver_form_defined = False

        try:
            new_equation._fitness_value = self._fitness_value
        except AttributeError:
            pass

        try:
            new_equation._coefficients_stability = self._coefficients_stability
        except AttributeError:
            pass

        try:
            new_equation._aic = self._aic
        except AttributeError:
            pass

    def add_history(self, add):
        """
        Adds a step to the equation's derivation history.
        
        This function records each transformation applied to the equation,
        allowing for later inspection and reconstruction of the solution path.
        
        Args:
            add (str): The string representing the transformation step to add to the history.
        
        Returns:
            None
        
        Why:
            Tracking the history of transformations is crucial for understanding
            how the equation was solved and for verifying the correctness of the solution.
            It enables debugging and allows users to retrace the steps taken during the derivation process.
        """
        # print(add)
        self._history += add

    @property
    def history(self):
        """
        Returns the history of the equation's evolution.
        
                This history tracks the transformations and refinements
                applied to the equation during the discovery process,
                providing insights into how the final form was achieved.
        
                Returns:
                    list: A list representing the history of the equation.
        """
        return self._history

    @property
    def fitness_value(self):
        """
        Gets the fitness value of the equation.
                This value represents how well the equation fits the observed data,
                guiding the search for the best equation structure.
        
                Returns:
                    float: The fitness value.
        """
        return self._fitness_value

    @fitness_value.setter
    def fitness_value(self, val):
        """
        Sets the fitness value of the equation.
        
                This value reflects how well the equation fits the observed data,
                guiding the search for the best equation structure.
        
                Args:
                    val: The fitness value to set.
        
                Returns:
                    None.
        """
        self._fitness_value = val

    def penalize_fitness(self, coeff=1.):
        """
        Penalizes the fitness value, influencing equation selection during the evolutionary process.
        
        This adjustment, controlled by the coefficient, guides the search towards simpler and more accurate equation representations.
        
        Args:
            coeff (float): The coefficient to penalize the fitness value.  A value less than 1.0 will improve the fitness, while a value greater than 1.0 will worsen it. Defaults to 1.0 (no penalty).
        
        Returns:
            None. Modifies the `_fitness_value` attribute of the Equation object directly.
        
        Why:
            Penalizing fitness allows the evolutionary algorithm to prioritize certain equation characteristics, such as simplicity or specific structural properties, during the equation discovery process.
        
        Class Fields:
            _fitness_value (float): The fitness value of the individual.
        """
        self._fitness_value = self._fitness_value*coeff

    @property
    def coefficients_stability(self):
        """
        Return the stability of the equation's coefficients.
        
                This property provides a measure of how much the coefficients of the equation vary during the evolutionary process.
                It helps assess the robustness and reliability of the identified equation, as stable coefficients indicate a more consistent and trustworthy model.
        
                Returns:
                    np.ndarray: The coefficients stability.
        """
        return self._coefficients_stability

    @coefficients_stability.setter
    def coefficients_stability(self, val):
        """
        Sets the stability threshold for equation coefficients.
        
                This value influences the evolutionary search process by guiding the algorithm 
                towards solutions with more robust and reliable coefficients. Setting an appropriate
                stability threshold can help prevent overfitting and improve the generalization 
                performance of the discovered equations.
        
                Args:
                    val (Any): The desired stability value for the coefficients.
        
                Returns:
                    None
        """
        self._coefficients_stability = val

    @property
    def aic(self):
        """
        Calculates the Akaike Information Criterion (AIC).
        
        The AIC is used to compare different equation structures and select the one that best balances model fit and complexity.
        It estimates the quality of each equation, relative to each of the other candidates, penalizing models with more free parameters.
        
        Args:
            self: The instance of the Equation class.
        
        Returns:
            float: The AIC value, used for model selection.
        """
        return self._aic

    @aic.setter
    def aic(self, val):
        """
        Sets the Akaike Information Criterion (AIC) value.
        
        This value is used to evaluate the trade-off between the goodness of fit and the complexity of the equation.
        A lower AIC indicates a better model.
        
        Args:
            val (float): The AIC value to set.
        
        Returns:
            None
        """
        self._aic = val

    @property
    def weights_internal(self):
        """
        Return the weights used internally within the equation.
        
                These weights are crucial for evaluating the equation's terms and determining its overall behavior. Accessing them allows for inspection of the learned equation structure.
        
                Raises:
                    AttributeError: If internal weights are accessed before the equation has been initialized and its weights have been determined.
        
                Returns:
                    The internal weights assigned to each term in the equation.
        """
        if self.weights_internal_evald:
            return self._weights_internal
        else:
            raise AttributeError(
                'Internal weights called before initialization')

    @weights_internal.setter
    def weights_internal(self, weights):
        """
        Sets the internal equation weights and resets evaluation flags.
        
                This ensures that the equation is re-evaluated with the new weights,
                reflecting their impact on the equation's fitness.
        
                Args:
                    weights (object): The weights to be set internally.
        
                Returns:
                    None
        """
        self._weights_internal = weights
        self.weights_internal_evald = True
        self.weights_final_evald = False

    @property
    def weights_final(self):
        """
        Return the final optimized weights of the equation. These weights represent the coefficients in the discovered differential equation that minimize the error between the equation's predictions and the observed data.
        
        Args:
            None
        
        Raises:
            AttributeError: If final weights are accessed before the equation has been optimized (i.e., before the evolutionary search has been completed).
        
        Returns:
            np.ndarray: A NumPy array containing the final optimized weights for each term in the equation.
        """
        if self.weights_final_evald:
            return self._weights_final
        else:
            print(self.text_form)
            raise AttributeError('Final weights called before initialization')

    @weights_final.setter
    def weights_final(self, weights):
        """
        Assigns the optimized weights to the equation and flags it as fully evaluated.
        
        This ensures that the equation is ready for use in subsequent calculations or analysis, 
        such as prediction or sensitivity analysis, by marking the weights as finalized.
        
        Args:
            weights (np.ndarray): The optimized weights to be assigned to the equation's terms.
        
        Returns:
            None
        """
        self._weights_final = weights
        self.weights_final_evald = True

    @property
    def text_form(self):
        """
        Generates a textual representation of the equation.
        
                This representation is crucial for interpreting the discovered equation.
                It allows users to view the equation in a human-readable format,
                either with the optimized weights or with symbolic coefficients,
                facilitating understanding of the relationships between terms.
        
                Returns:
                    str: A string representing the equation, either with evaluated weights or symbolic coefficients.
        """
        form = ''
        if self.weights_final_evald:
            for term_idx in range(len(self.structure)):
                if term_idx != self.target_idx:
                    form += str(self.weights_final[term_idx]) if term_idx < self.target_idx else str(
                        self.weights_final[term_idx-1])
                    form += ' * ' + self.structure[term_idx].name + ' + '
            form += str(self.weights_final[-1]) + ' = ' + \
                self.structure[self.target_idx].name
        else:
            for term_idx in range(len(self.structure)):
                form += 'k_' + str(term_idx) + ' ' + \
                    self.structure[term_idx].name + ' + '
            form += 'k_' + str(len(self.structure)) + ' = 0'
        return form

    @property
    def latex_form(self):
        """
        Returns the equation in LaTeX format, combining the identified structure and optimized coefficients.
        
                The equation is constructed by summing the LaTeX representations of each term in the identified structure,
                weighted by their corresponding optimized coefficients. This provides a human-readable representation
                of the discovered equation, facilitating analysis and interpretation of the identified relationships
                within the data.
        
                Returns:
                    str: A LaTeX formatted string representing the equation.
        """
        form = self.structure[self.target_idx].latex_form + r' = '
        digits_rounding_max = 3
        for idx, term in enumerate(self.structure):
            idx_corrected = idx if idx <= self.target_idx else idx - 1
            if idx == self.target_idx or self.weights_final[idx_corrected] == 0:
                continue

            mnt, exp = exp_form(self.weights_final[idx_corrected], digits_rounding_max)
            exp_str = r'\cdot 10^{{{0}}} '.format(str(exp)) if exp != 0 else ''
            form += str(mnt) + exp_str + term.latex_form + r' + '

        mnt, exp = exp_form(self.weights_final[-1], digits_rounding_max)
        exp_str = r'\cdot 10^{{{0}}} '.format(str(exp)) if exp != 0 else ''

        form += str(mnt) + exp_str
        return form

    @property
    def state(self):
        """
        Returns the equation in a human-readable format.
        
                This representation is used to display the equation and can be 
                useful for debugging or understanding the discovered model.
        
                Returns:
                    str: The equation represented as a string.
        """
        return self.text_form

    @property
    def described_variables(self):
        """
        Identifies variable types effectively captured by the equation.
        
                This method determines which variable types are well-represented
                within the equation, considering both the equation's structure
                and the learned coefficients. It assesses each term in the
                equation to see if it contributes meaningfully to describing
                variables. The target term invariably contributes. Other terms
                contribute if their corresponding coefficient has a significant
                magnitude. A term's contribution involves adding the family
                types of its factors to the set of described variables, but
                only if the factor is a derivative and its derivative code is
                not None.
        
                Args:
                    self: The Equation instance.
        
                Returns:
                    frozenset: A frozen set containing the variable types
                        (family types of factors) that are considered described
                        by the equation. The equation uses this information to
                        understand which variables are most relevant to its
                        solution.
        """
        eps = 1e-7
        described = set()
        for term_idx, term in enumerate(self.structure):
            if term_idx == self.target_idx:
                described.update({factor.family_type for factor in term.structure
                                  if factor.is_deriv and factor.deriv_code != [None]})
            else:
                weight_idx = term_idx if term_idx < term_idx else term_idx - 1
                if np.abs(self.weights_final[weight_idx]) > eps:
                    described.update({factor.family_type for factor in term.structure
                                      if factor.is_deriv and factor.deriv_code != [None]})
        described = frozenset(described)
        return described

    def max_deriv_orders(self):
        """
        Computes the maximum derivative orders for each axis in the solver form.
        
        This method analyzes the solver form of the equation to determine the
        highest derivative order present for each spatial dimension. It iterates
        through the terms in the solver form, counts the derivative order for each
        axis, and updates the maximum orders accordingly. This is crucial for
        determining the complexity of the equation and ensuring compatibility
        with the numerical solver.
        
        Args:
            self: The object instance.
        
        Returns:
            np.ndarray: An array containing the maximum derivative order for each axis.
        """
        solver_form = self.solver_form()
        max_orders = np.zeros(global_var.grid_cache.get('0').ndim)

        def count_order(obj, deriv_ax):
            if obj is None:
                return 0
            else:
                return obj.count(deriv_ax)

        for term in solver_form:
            if isinstance(term[2], list):
                for deriv_factor in term[1]:
                    orders = np.array([count_order(deriv_factor, ax) for ax
                                       in np.arange(max_orders.size)])
                    max_orders = np.maximum(max_orders, orders)
            else:
                orders = np.array([count_order(term[1], ax) for ax
                                   in np.arange(max_orders.size)])
                max_orders = np.maximum(max_orders, orders)
        if np.max(max_orders) > 4:
            raise NotImplementedError('The current implementation allows does not allow higher orders of equation, than 2.')
        return max_orders

    def boundary_conditions(self, max_deriv_orders=(1,), main_var_key=('u', (1.0,)), full_domain: bool = False,
                                grids : list = None):
        """
        Generates boundary conditions for the problem domain.
        
                This method constructs boundary conditions necessary for solving
                differential equations. It determines the locations and values
                where the solution is known or constrained, using grid and tensor
                data to define these conditions. These constraints are essential
                for obtaining a unique and physically relevant solution to the
                equation.
        
                Args:
                    max_deriv_orders: A tuple indicating the maximum derivative orders
                        for each axis. This determines how many boundary conditions
                        are needed along each dimension.
                    main_var_key: A tuple containing the key for the main variable
                        and its associated scaling factor. Specifies the variable
                        for which boundary conditions are being defined.
                    full_domain: A boolean flag indicating whether to use the full
                        domain for boundary conditions. If True, uses initial data
                        instead of the current grid state.
                    grids: A list of grids.
        
                Returns:
                    list: A list of boundary conditions, where each boundary condition
                    is a list containing coordinates, values, and the type of boundary
                    condition (currently only 'dirichlet'). These conditions constrain
                    the solution space of the differential equation.
        """
            required_bc_ord = max_deriv_orders   # We assume, that the maximum order of the equation here is 2
            if global_var.grid_cache is None:
                raise NameError('Grid cache has not been initialized yet.')

            bconds = []
            hardcoded_bc_relative_locations = {0: (), 1: (0,), 2: (0, 1),
                                               3: (0., 0.5, 1.), 4: (0., 1/3., 2/3., 1.)}

            if full_domain:
                grid_cache = global_var.initial_data_cache
                tensor_cache = global_var.initial_data_cache
            else:
                grid_cache = global_var.grid_cache
                tensor_cache = global_var.tensor_cache

            tensor_shape = grid_cache.get('0').shape

            def get_boundary_ind(tensor_shape, axis, rel_loc):
                return tuple(np.meshgrid(*[np.arange(shape) if dim_idx != axis else min(int(rel_loc * shape), shape-1)
                                           for dim_idx, shape in enumerate(tensor_shape)], indexing='ij'))
            for ax_idx, ax_ord in enumerate(required_bc_ord):
                for loc_fraction in hardcoded_bc_relative_locations[ax_ord]:
                    indexes = get_boundary_ind(tensor_shape, axis=ax_idx, rel_loc=loc_fraction)
                    coords_raw = np.array([grid_cache.get(str(idx))[indexes] for idx
                                           in np.arange(len(tensor_shape))])
                    coords = coords_raw.T
                    if coords.ndim > 2:
                        coords = coords.squeeze()
                    vals = np.expand_dims(tensor_cache.get(main_var_key)[indexes], axis=0).T

                    coords = torch.from_numpy(coords).type(torch.FloatTensor)

                    vals = torch.from_numpy(vals).type(torch.FloatTensor)
                    bconds.append([coords, vals, 'dirichlet'])

            return bconds

    def clear_after_solver(self):
        """
        Clears the solver and model from memory after solving.
        
        After a solution is found, this method releases the memory occupied by the model and solver.
        This ensures efficient resource management, especially when exploring multiple equation candidates.
        It also resets the flag indicating solver form definition.
        
        Args:
            self: The Equation object instance.
        
        Returns:
            None.
        """
        del self.model
        del self._solver_form
        self.solver_form_defined = False
        gc.collect()

    def __iter__(self):
        """
        Returns an iterator for the equation.
        
        This allows to traverse the equation's components sequentially, which is essential for 
        various operations like evaluation, simplification, or transformation during the equation 
        discovery process.
        
        Args:
            None
        
        Returns:
            EquationIterator: An iterator object for traversing the equation.
        """
        return EquationIterator(self)        

class EquationIterator(object):
    """
    An iterator class for traversing an equation's terms.
    
        Class Methods:
        - __init__: Initializes the EquationIterator.
        - __next__: Returns the next coefficient and term in the equation structure.
    
        Class Fields:
            _internal_idx (int): An internal index used to track the current position in the equation's terms.
            _equation (Equation): The equation being iterated over.
    """

    def __init__(self, equation: Equation):
        """
        Initializes the EquationIterator for traversing terms within an equation.
        
                The iterator prepares to step through the equation's terms, facilitating the extraction and manipulation of individual components during the equation discovery process. This is a crucial step for algorithms that analyze and modify equations to fit observed data.
        
                Args:
                    equation (Equation): The equation to iterate over.
        
                Returns:
                    None
        """
        self._internal_idx = 0
        self._equation = equation

    def __next__(self) -> Tuple[Union[None, float], Term]:
        """
        Returns the next coefficient and term in the equation.
        
                This iterator is used to traverse the equation's structure, yielding coefficient-term pairs.
                If the equation has been evaluated (weights are available), it returns the corresponding weight as the coefficient for each term.
                Terms with zero weight are skipped to optimize the equation's representation, except for the target term, which always has a coefficient of -1.
                If the equation hasn't been evaluated, it returns `None` as the coefficient, indicating that the weights are not yet determined.
                This is done to provide an efficient way to access the equation's components during the evolutionary process,
                allowing for manipulation and evaluation of the equation's structure.
        
                Args:
                    self: The instance of the EquationIterator class.
        
                Returns:
                    Tuple[Union[None, float], Term]: A tuple containing the coefficient (a float if weights are evaluated, None otherwise, or -1.0 for the target term) and the Term object.
        
                Raises:
                    StopIteration: If the end of the equation structure is reached.
        """
        if self._internal_idx < len(self._equation.structure):
            if self._equation.weights_final_evald:
                while True:
                    idx_in_weights = self._internal_idx if self._internal_idx <= self._equation.target_idx \
                        else self._internal_idx - 1

                    if self._internal_idx == self._equation.target_idx:
                        coeff = -1.
                        break
                    elif self._equation.weights_final[idx_in_weights] == 0:
                        self._internal_idx += 1
                        if self._internal_idx >= len(self._equation.structure):
                            raise StopIteration
                    else:
                        coeff = self._equation.weights_final[idx_in_weights]
                        break
            else:                    
                coeff = None
            
            term = self._equation.structure[self._internal_idx]
            self._internal_idx += 1
            return (coeff, term)
        else:
            raise StopIteration

def solver_formed_grid(training_grid=None):
    """
    Solves a formed grid (deprecated).
    
    This method is intended to be deprecated as the approach of forming a grid
    before equation discovery is no longer optimal within the EPDE framework.
    It processes a training grid, potentially retrieving it from a global cache
    if not provided. It then reshapes the grid and converts it into a PyTorch
    FloatTensor. This was initially used to prepare data for the equation
    discovery process, but newer methods directly operate on raw data or
    use more flexible data structures.
    
    Args:
      training_grid: The training grid to process. If None, the grid is
        retrieved from the global cache.
    
    Returns:
      torch.Tensor: A PyTorch FloatTensor representing the processed training grid.
    
    Raises:
      NotImplementedError: Always raised, as this function is deprecated.
    """
    raise NotImplementedError('solver_formed_grid function is to be depricated')
    if training_grid is None:
        keys, training_grid = global_var.grid_cache.get_all()
    else:
        keys, _ = global_var.grid_cache.get_all()

    assert len(keys) == training_grid[0].ndim, 'Mismatching dimensionalities'

    training_grid = np.array(training_grid).reshape((len(training_grid), -1))
    return torch.from_numpy(training_grid).T.type(torch.FloatTensor)

def check_metaparameters(metaparameters: dict):
    """
    Validates essential metaparameters required for equation discovery.
    
    This function ensures that the necessary configuration parameters
    for the equation search process are present and structurally sound,
    allowing the system to proceed with the equation discovery.
    It checks for the existence of key labels within the provided
    metaparameters dictionary.
    
    Args:
        metaparameters (dict): A dictionary containing metaparameters,
            expected to include keys like 'terms_number',
            'max_factors_in_term', and 'sparsity'.
    
    Returns:
        bool: True, indicating that the metaparameters dictionary
            contains the expected keys. Always returns True, but serves
            as a placeholder for potential future validation logic.
    """
    metaparam_labels = ['terms_number', 'max_factors_in_term', 'sparsity']
    return True


class SoEq(moeadd.MOEADDSolution):
    """
    Represents a system of equations (SoEq) for symbolic regression.
    
                The class encapsulates a collection of equations and provides
                functionality for their manipulation, evaluation, and optimization
                within an evolutionary algorithm.
    
            Class Methods:
                - __init__
                - manual_reconst
                - use_default_multiobjective_function
                - use_legacy_multiobjective_function
                - use_pic_multiobjective_function
                - use_default_singleobjective_function
                - set_objective_functions
                - matches_complexitiy
                - create
                - equation_opt_iteration
                - obj_fun
                - __call__
                - text_form
                - __eq__
                - latex_form
                - __hash__
                - __deepcopy__
                - reset_state
                - copy_properties_to
                - solver_params
                - __iter__
                - fitness_calculated
    """

    def __init__(self, pool: TFPool, metaparameters: dict):
        '''
        Initializes the equation search object with token families and metaparameters.
        
        This setup prepares the search space by separating tokens intended for equation construction 
        from those used for supporting roles, and initializes the metaparameters that guide the search process.
        It ensures that the provided metaparameters are valid and prepares the token pools for equation discovery.
        
        Args:
            pool (epde.interface.token_familiy.TFPool): Pool containing token families for the equation search.
            metaparameters (dict): Dictionary of metaparameters, where each key is a parameter label (e.g., 'sparsity')
                and each value is a tuple containing a flag for metaoptimization and an initial value.
        
        Returns:
            None
        '''
        check_metaparameters(metaparameters)

        self.obj_funs = None

        self.metaparameters = metaparameters
        self.tokens_for_eq = TFPool(pool.families_demand_equation)
        self.tokens_supp = TFPool(pool.families_equationless)
        self.moeadd_set = False

        self.vars_to_describe = [token_family.variable for token_family in self.tokens_for_eq.families]

    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        """
        Manually reconstructs a chromosome attribute, ensuring the solution space remains valid after direct manipulation.
                
                This method allows for the targeted modification of a chromosome's attribute,
                specifically the 'vals' attribute which represents the equation terms. It
                validates the provided term definition and updates the chromosome's
                internal state accordingly, maintaining the integrity of the equation discovery process.
                
                Args:
                    attribute: The attribute to reconstruct. Currently, only 'vals' is supported.
                    value: The new value for the specified attribute. For 'vals', this should be
                        a list of equation elements, each defining a term in the equation.
                    except_attrs: A dictionary of attributes to exclude during the reconstruction
                        process. This allows for selective updates while preserving other aspects
                        of the equation definition.
                
                Returns:
                    None.
                
                Raises:
                    ValueError: If the specified attribute is not supported.
                
                Class Fields:
                    vals (Chromosome): The chromosome representing the solution, containing
                        equations and optimizable metaparameters. Initialized with a new
                        Chromosome object based on the provided value and metaparameters.
                
                Why:
                    This method is crucial for directly influencing the equation search process.
                    It enables the user to inject prior knowledge or correct specific parts of
                    the equation structure, guiding the evolutionary algorithm towards more
                    promising solutions.
        """
        from epde.loader import attrs_from_dict, get_typespec_attrs
        supported_attrs = ['vals']
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')

        if attribute == supported_attrs[0]:
            # Validate correctness of a term definition
            equations = {}
            for idx, eq_elem in enumerate(value):
                eq = Equation.__new__(Equation)
                attrs_from_dict(eq, eq_elem, except_attrs)
                equations[self.vars_to_describe[idx]] = eq
            self.vals = Chromosome(equations, {key: val for key, val in self.metaparameters.items()
                                               if val['optimizable']})

    def use_default_multiobjective_function(self, use_pic: bool = False):
        """
        Selects and applies a default multi-objective function to guide the evolutionary search.
        
                This method determines which multi-objective function to use during the equation discovery process.
                The choice between the PIC-based and legacy functions is made based on the `use_pic` flag,
                allowing the system to adapt its optimization strategy. This choice influences how the algorithm
                balances different objectives when searching for the best equation structure.
        
                Args:
                    use_pic: A boolean flag indicating whether to use the PIC-based
                        multi-objective function.
        
                Returns:
                    None.
        """
        if use_pic:
            self.use_pic_multiobjective_function()
        else:
            self.use_legacy_multiobjective_function()

    def use_legacy_multiobjective_function(self):
        """
        Configures the evolutionary process to use a legacy multi-objective function.
        
                This method sets the objective functions for the evolutionary process
                using a combination of equation fitness and equation complexity,
                calculated for each variable to be described. It leverages legacy
                functions for calculating fitness and complexity. This is done to
                evaluate candidate equations based on both their accuracy in fitting
                the data and their simplicity, guiding the search towards parsimonious
                models.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        from epde.eq_mo_objectives import generate_partial, equation_fitness, equation_complexity_by_factors
        complexity_objectives = [generate_partial(equation_complexity_by_factors, eq_key)
                                 for eq_key in self.vars_to_describe]
        quality_objectives = [generate_partial(
            equation_fitness, eq_key) for eq_key in self.vars_to_describe]
        self.set_objective_functions(
            quality_objectives + complexity_objectives)

    def use_pic_multiobjective_function(self):
        """
        Sets the objective functions for multi-objective optimization.
        
                This method configures the objective functions used in the multi-objective optimization process.
                It generates partial functions for equation fitness and equation term stability for each variable to be described.
                Finally, it sets the objective functions to be a combination of quality (fitness) and stability objectives.
                This configuration is crucial for guiding the search towards solutions that not only accurately describe the data but also exhibit stable and reliable behavior.
        
                Args:
                    self: The instance of the class.
        
                Returns:
                    None
        """
        from epde.eq_mo_objectives import generate_partial, equation_fitness, equation_complexity_by_factors, equation_terms_stability, equation_aic
        complexity_objectives = [generate_partial(equation_complexity_by_factors, eq_key)
                                 for eq_key in self.vars_to_describe]
        quality_objectives = [generate_partial(
            equation_fitness, eq_key) for eq_key in self.vars_to_describe]
        stability_objectives = [generate_partial(
            equation_terms_stability, eq_key) for eq_key in self.vars_to_describe]
        aic_objectives = [generate_partial(
            equation_aic, eq_key) for eq_key in self.vars_to_describe]
        self.set_objective_functions(
            # quality_objectives + stability_objectives + complexity_objectives)
            # quality_objectives + stability_objectives + aic_objectives)
            quality_objectives + stability_objectives)

    def use_default_singleobjective_function(self):
        """
        Sets the objective functions to default single-objective functions.
        
                It generates partial functions based on equation fitness for each variable
                to describe and sets them as the objective functions. This is done to
                evaluate the quality of candidate equations by focusing on individual
                variables, enabling a more granular assessment of equation fitness
                during the equation discovery process.
        
                Args:
                  self: The object instance.
        
                Returns:
                  None.
        
                Initializes:
                    objective_functions (list): A list of objective functions, where each function
                      evaluates the fitness of an equation with respect to a specific variable
                      being described.
        """
        from epde.eq_mo_objectives import generate_partial, equation_fitness
        quality_objectives = [generate_partial(equation_fitness, eq_key) for eq_key in self.vars_to_describe]#range(len(self.tokens_for_eq))]
        self.set_objective_functions(quality_objectives)

    def set_objective_functions(self, obj_funs):
        """
        Method to define the functions used to assess the suitability of a generated system of equations. These functions quantify how well the equations capture the underlying dynamics of the system.
        
                Args:
                    obj_funs (callable or list of callables): A function or list of functions. Each function should evaluate the system of equations and return a numerical score (or a list of scores) representing its quality.  For example, a function might measure the error between the model's predictions and the observed data. The results from all functions are combined to provide a comprehensive evaluation.
        
                Returns:
                    None
        
                Why:
                    Objective functions are set to evaluate generated equations and guide the search towards those that best represent the underlying system dynamics.
        """
        assert callable(obj_funs) or all([callable(fun) for fun in obj_funs])
        self.obj_funs = obj_funs

    def matches_complexitiy(self, complexity : Union[int, list]):
        """
        Checks if the provided complexity values align with the tail end of the objective function. This ensures that the evolutionary search is guided towards solutions that respect pre-defined complexity constraints or default to the objective function's inherent complexity.
        
                Args:
                    complexity (Union[int, list]): The complexity to check. If an integer or float, it's treated as a uniform complexity for all variables. If a list, it specifies the complexity for each variable being described.
        
                Returns:
                    bool: True if the last elements of the objective function match the (possibly adjusted) complexity, indicating a match; False otherwise.
        """
        if isinstance(complexity, (int, float)):
            complexity = [complexity,]

        if not isinstance(complexity, list) or len(self.vars_to_describe) != len(complexity):
            raise ValueError('Incorrect list of complexities passed.')
        adj_complexity = copy.copy(complexity)
        for idx, compl in enumerate(adj_complexity):
            if compl is None:
                adj_complexity[idx] = self.obj_fun[-len(complexity) + idx]

        return list(self.obj_fun[-len(adj_complexity):]) == adj_complexity

    def create(self, passed_equations: list = None):
        """
        Creates the equation structure that defines the search space for potential solutions.
        
                This method sets up the equation structure, either by generating a default one
                based on available tokens and variables or by using a pre-defined list of
                equations. This structure is then used to initialize the chromosome, which
                represents a candidate solution in the evolutionary search process.
        
                Args:
                    passed_equations: A list of Equation objects to use for the structure.
                        If None, a default structure is generated based on available tokens.
        
                Returns:
                    None
        
                Class Fields Initialized:
                    vals (Chromosome): A Chromosome object representing the solution,
                        initialized with the created equation structure and optimizable
                        metaparameters.
                    moeadd_set (bool): A boolean flag set to True, indicating that the
                        MOEADDSolution has been initialized.
        
                Why:
                    This method is crucial for defining the space of possible differential
                    equations that the evolutionary algorithm will explore. By creating an
                    appropriate equation structure, the search can be guided towards
                    discovering equations that accurately model the underlying dynamics of
                    the system.
        """
        if passed_equations is None:
            structure = {}

            token_selection = self.tokens_supp
            current_tokens_pool = token_selection + self.tokens_for_eq

            for eq_idx, variable in enumerate(self.vars_to_describe):
                structure[variable] = Equation(current_tokens_pool, basic_structure=[],
                                               var_to_explain=variable,
                                               metaparameters=self.metaparameters)
        else:
            if len(passed_equations) != len(self.vars_to_describe):
                raise ValueError('Length of passed equations list does not match')
            structure = {self.vars_to_describe[idx] : eq for idx, eq in enumerate(passed_equations)}

        self.vals = Chromosome(structure, params={key: val for key, val in self.metaparameters.items()
                                                  if val['optimizable']})
        moeadd.MOEADDSolution.__init__(self, self.vals, self.obj_funs)
        self.moeadd_set = True

    @staticmethod
    def equation_opt_iteration(population, evol_operator, population_size, iter_index, unexplained_vars, strict_restrictions=True):
        """
        Performs one iteration of the equation optimization process.
        
                This method refines the population of equation candidates by penalizing those that include unexplained variables, sorting them based on their fitness, and truncating the population to maintain a manageable size.  The evolutionary operator is then applied to generate a new, potentially improved population of equation candidates. This iterative process aims to evolve a population of equations that accurately describe the underlying dynamics of the system.
        
                Args:
                    population: The current population of equations.
                    evol_operator: The evolutionary operator to apply.
                    population_size: The desired size of the population.
                    iter_index: The index of the current iteration (unused).
                    unexplained_vars: A list of variables that are currently unexplained.
                    strict_restrictions: A boolean indicating whether to apply strict restrictions (default: True). (unused)
        
                Returns:
                    The new population after applying the evolutionary operator.
        """
        for equation in population:
            if equation.described_variables in unexplained_vars:
                equation.penalize_fitness(coeff=0.)
        population = population_sort(population)
        population = population[:population_size]
        gc.collect()
        population = evol_operator.apply(population, unexplained_vars)
        return population

    @property
    def obj_fun(self):
        """
        Calculates the objective function values based on the defined objective functions.
        
                This property evaluates each objective function defined within the model
                and returns a flattened array of the resulting values. This is crucial
                for the multi-objective optimization process, where the algorithm seeks
                to minimize all objective functions simultaneously to find the best equation
                representation.
        
                Returns:
                    np.ndarray: An array containing the flattened objective function values.
        """
        return np.array(flatten([func(self) for func in self.obj_funs]))

    def __call__(self):
        """
        Call the object to retrieve the objective function.
        
        This method ensures that the equation's structure, represented by the moeadd_set, has been defined.
        This is crucial because the objective function relies on this structure to evaluate the equation's fitness
        against the data.
        
        Args:
            self: The object instance.
        
        Returns:
            The objective function.
        """
        assert self.moeadd_set, 'The structure of the equation is not defined, therefore no moeadd operations can be called'
        return self.obj_fun

    @property
    def text_form(self):
        """
        Generates a text-based representation of the equation or system of equations.
        
                This method constructs a string representation of the object,
                primarily based on the 'vals' (equations) and 'metaparameters' attributes.
                If 'vals' contains multiple equations, it formats them with separators to
                represent a system of equations. The metaparameters are also included
                in the final string representation.
        
                Args:
                    self: The instance of the class (SoEq).
        
                Returns:
                    str: A string representation of the equation or system of equations,
                         including metaparameters.
        
                The text form is essential for displaying the discovered equation or
                system of equations in a human-readable format, allowing users to
                easily inspect the discovered relationships and their associated
                metaparameters. This is crucial for understanding the model discovered
                by the EPDE framework.
        """
        form = ''
        if len(self.vals) > 1:
            for eq_idx, equation in enumerate(self.vals):
                if eq_idx == 0:
                    form += ' / ' + equation.text_form + '\n'
                elif eq_idx == len(self.vals) - 1:
                    form += ' \ ' + equation.text_form + '\n'
                else:
                    form += ' | ' + equation.text_form + '\n'
        else:
            form += [val.text_form for val in self.vals][0] + '\n'
        form += str(self.metaparameters)
        return form

    def __eq__(self, other):
        """
        Compares two equation objects for structural equality.
        
        This method determines if two equation objects are structurally equivalent by comparing their 'vals' attributes,
        ensuring that they represent the same equation structure. Structural equality is crucial for
        identifying redundant or equivalent equation forms during the evolutionary search process.
        It verifies that each element in the 'vals' of both objects has a corresponding equal element in the 'vals'
        of the other object, and that the lengths of their 'vals' are the same. It also asserts that the moeadd_set
        is not empty, ensuring that the equation structure is properly defined before comparison.
        
        Args:
            other: The equation object to compare with.
        
        Returns:
            bool: True if the equation objects are structurally equal, False otherwise.
        """
        assert self.moeadd_set, 'The structure of the equation is not defined, therefore no moeadd operations can be called'
        return (all([any([other_elem == self_elem for other_elem in other.vals]) for self_elem in self.vals]) and
                all([any([other_elem == self_elem for self_elem in self.vals]) for other_elem in other.vals]) and
                len(other.vals) == len(self.vals))  # or all(np.isclose(self.obj_fun, other.obj_fun)

    @property
    def latex_form(self):
        """
        Returns the LaTeX form of the equations.
        
                Generates a LaTeX representation of the stored equations,
                formatted within an `eqnarray*` environment. This representation
                facilitates the symbolic manipulation and interpretation of the
                discovered equations.
        
                Args:
                    self: The object instance.
        
                Returns:
                    str: A LaTeX string representing the equations.
                
                Why: To provide a human-readable and easily integrable representation of the discovered equations for further analysis or use in other symbolic computation tools.
        """
        form = r"\begin{eqnarray*} "
        for idx, equation in enumerate(self.vals):
            postfix = '' if idx == len(self.vals) - 1 else r", \\ "
            form += equation.latex_form + postfix
        form += r" \end{eqnarray*}"
        return form

    def __hash__(self):
        """
        Returns the hash value of the object.
        
                This hash value is derived from the internal 'vals' object's hash descriptor, ensuring that structurally equivalent objects are treated as identical for hashing purposes. This is crucial for efficient storage and retrieval of equations within sets and dictionaries, as it allows the framework to quickly identify and avoid redundant equation representations during the evolutionary search process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    int: The hash value of the object.
        """
        return hash(self.vals.hash_descr)

    def __deepcopy__(self, memo=None):
        """
        Creates a deep copy of the object, ensuring that all components are duplicated independently.
        
                This method is essential for preserving the integrity of equation structures
                during evolutionary processes. It creates a new instance of the class and
                recursively copies all attributes from the original object to the new object
                using `copy.deepcopy`. This prevents unintended modifications to shared
                components when exploring different equation candidates. Handles both
                `__dict__` and `__slots__` attributes to ensure a complete and independent copy.
        
                Args:
                    memo (dict, optional): A dictionary used to keep track of objects that have
                        already been copied to prevent infinite recursion. Defaults to None.
        
                Returns:
                    SoEq: The deep copy of the object.
        """
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct

        for k, v in self.__dict__.items():
            setattr(new_struct, k, copy.deepcopy(v, memo))

        for k in self.__slots__:
            try:
                if not isinstance(k, list):
                    setattr(new_struct, k, copy.deepcopy(getattr(self, k), memo))
                else:
                    temp = []
                    for elem in getattr(self, k):
                        temp.append(copy.deepcopy(elem, memo))
                    setattr(new_struct, k, temp)
            except AttributeError:
                pass
        return new_struct

    def reset_state(self, reset_right_part: bool = True):
        """
        Resets the state of all equations stored in the object.
        
                This method iterates through each equation and resets its internal state,
                preparing it for a new optimization cycle. This ensures that previous
                computations do not influence subsequent evolutionary steps in discovering
                the underlying differential equation.
        
                Args:
                  reset_right_part: Whether to reset the right-hand side of the equation.
        
                Returns:
                  None
        """
        for equation in self.vals:
            equation.reset_state(reset_right_part)

    def copy_properties_to(self, objective):
        """
        Copies the equation properties to another object.
        
        This ensures that the objective object has the same equation
        properties as the current object, maintaining consistency in
        equation structure during the evolutionary process.
        
        Args:
            objective: The objective object to copy properties to.
        
        Returns:
            None.
        """
        for eq_label in self.vals.equation_keys:  # Not the best code possible here
            self.vals[eq_label].copy_properties_to(objective.vals[eq_label])

    def solver_params(self, full_domain, grids=None):
        """
        Returns the equation transformed into a solver-compatible form, the grid on which the solution is computed, and the boundary conditions applied to the equations. This is a crucial step in preparing the problem for numerical solution, ensuring that the discovered equations can be accurately solved within the specified domain and constraints.
        
                Args:
                    full_domain: The full domain of the problem.
                    grids: Optional grid to use for the solver. If None, a default grid is used.
        
                Returns:
                    A tuple containing:
                      - equation_forms: A list of equations in a solver-compatible form.
                      - solver_formed_grid: The grid used for the solver.
                      - bconds: A list of boundary conditions for each equation.
        """
        equation_forms = []
        bconds = []

        for idx, equation in enumerate(self.vals):
            equation_forms.append(equation.solver_form(grids=grids))
            bconds.append(equation.boundary_conditions(full_domain=full_domain, grids=grids,
                                                       index=idx))

        return equation_forms, solver_formed_grid(grids), bconds

    def __iter__(self):
        """
        Returns an iterator for the equation.
        
                This allows to traverse through the equation's terms, enabling operations like evaluation and simplification.
        
                Returns:
                    SoEqIterator: An iterator object for traversing the elements of the SoEq object.
        """
        return SoEqIterator(self)

    @property
    def fitness_calculated(self):
        """
        Indicates whether all equations in the system have their fitness values computed. This is crucial for assessing the overall quality of the equation system during the evolutionary search process.
        
                Args:
                    self: The SoEq instance.
        
                Returns:
                    bool: True if all equations have calculated fitness, False otherwise.
        """
        return all([equation.fitness_calculated for equation in self.vals])


class SoEqIterator(object):
    """
    An iterator for stepping through solutions of a system of equations.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, system: SoEq):
        """
        Initializes the iterator with a system of equations.
        
                This prepares the iterator to traverse the solution space of the given equation system.
                The iterator extracts the variables to be described and sets up the internal state for exploration.
        
                Args:
                    system: The system of equations to solve.
        
                Class Fields:
                    _idx: An index, initialized to 0, used for tracking the current position within the solution space.
                    system: The system of equations to be solved.
                    keys: A list of the variables to be described, extracted from the input system.
        
                Returns:
                    None
        """
        self._idx = 0
        self.system = system
        self.keys = list(system.vars_to_describe)

    def __next__(self):
        """
        Returns the next value associated with a key in the system.
        
        This method advances the iterator to the next key and retrieves the corresponding value
        from the system's value store. It's used to sequentially access values during the
        equation discovery process.
        
        Args:
            self: The object instance.
        
        Returns:
            The next value in the iteration.
        
        Raises:
            StopIteration: If all keys have been iterated over, indicating the end of the sequence.
        """
        if self._idx < len(self.keys):
            res = self.system.vals[self.keys[self._idx]]
            self._idx += 1
            return res
        else:
            raise StopIteration
