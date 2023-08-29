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
from typing import Union, Callable
from functools import singledispatchmethod, reduce

import numpy as np
import torch

import epde.globals as global_var
import epde.optimizers.moeadd.solution_template as moeadd

from epde.structure.encoding import Chromosome
from epde.interface.token_family import TFPool
from epde.decorators import HistoryExtender, ResetEquationStatus
from epde.supplementary import filter_powers, normalize_ts, population_sort, flatten
from epde.structure.factor import Factor
from epde.structure.structure_template import ComplexStructure, check_uniqueness


class Term(ComplexStructure):
    """
    Class for describing the term of differential equation

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

    @property
    def cache_label(self):
        if len(self.structure) > 1:
            structure_sorted = sorted(self.structure, key=lambda x: x.cache_label)
            cache_label = tuple([elem.cache_label for elem in structure_sorted])
        else:
            cache_label = self.structure[0].cache_label
        return cache_label

    def use_cache(self):
        self.cache_linked = True
        for idx, _ in enumerate(self.structure):
            if not self.structure[idx].cache_linked:
                self.structure[idx].use_cache()

    # TODO: make self.descr_variable_marker setting for defined parameter

    @singledispatchmethod
    def defined(self, passed_term):
        raise NotImplementedError(
            f'passed term should have string or list/dict types, not {type(passed_term)}')

    @defined.register
    def _(self, passed_term: list, collapse_powers = True):
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
                # print('family is ', family)
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
        # print(f'self.descr_variable_marker is set as {self.descr_variable_marker}')

        if not mandatory_family:
            occupied_by_factor, factor = self.pool.create(label=None, create_meaningful=True,
                                                          token_status=self.occupied_tokens_labels,
                                                          create_derivs=create_derivs, **kwargs)
        else:
            occupied_by_factor, factor = self.pool.create_from_family(family_label=mandatory_family,
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
        return self._descr_variable_marker

    @descr_variable_marker.setter
    def descr_variable_marker(self, marker: False):
        if not marker or isinstance(marker, str):
            self._descr_variable_marker = marker
        else:
            raise ValueError('Described variable marker shall be a family label (i.e. "u") of "False"')

    def evaluate(self, structural, grids=None):
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
            if normalize and np.ndim(value) != 1:
                value = normalize_ts(value)
            elif normalize and np.ndim(value) == 1 and np.std(value) != 0:
                value = (value - np.mean(value))/np.std(value)
            elif normalize and np.ndim(value) == 1 and np.std(value) == 0:
                value = (value - np.mean(value))
            if np.all([len(factor.params) == 1 for factor in self.structure]) and grids is None:
                # Место возможных проблем: сохранение/загрузка нормализованных данных
                self.saved[normalize] = global_var.tensor_cache.add(self.cache_label, value, normalized=normalize)
                if self.saved[normalize]:
                    self.saved_as[normalize] = self.cache_label
            value = value.reshape(-1)
            return value

    def filter_tokens_by_right_part(self, reference_target, equation, equation_position):
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
        return max(sum([len(element.params) - 1 for element in self.structure]), 1)

    @property
    def name(self):
        form = ''
        for token_idx in range(len(self.structure)):
            form += self.structure[token_idx].name
            if token_idx < len(self.structure) - 1:
                form += ' * '
        return form

    def contains_deriv(self, family=None):
        if family is None:
            return any([factor.is_deriv and factor.deriv_code != [None,] for factor in self.structure])
        else:
            return any([factor.ftype == family and factor.deriv_code != [None,] for factor in self.structure])

    def contains_family(self, family):
        return any([factor.ftype == family for factor in self.structure])

    def __eq__(self, other):
        return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure])
                and all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure])
                and len(other.structure) == len(self.structure))

    @HistoryExtender('\n -> was copied by deepcopy(self)', 'n')
    def __deepcopy__(self, memo=None):
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
    __slots__ = ['_history', 'structure', 'interelement_operator', 'saved', 'saved_as',
                 'n_immutable', 'pool', 'terms_number', 'max_factors_in_term', 'operator',
                 '_target', 'target_idx', '_features', 'right_part_selected',
                 '_weights_final', 'weights_final_evald', '_weights_internal', 'weights_internal_evald',
                 'fitness_calculated', 'solver_form_defined', '_solver_form', '_fitness_value',
                 'crossover_selected_times', 'metaparameters', 'main_var_to_explain']

    def __init__(self, pool: TFPool, basic_structure: Union[list, tuple, set], var_to_explain: str = None,
                 metaparameters: dict = {'sparsity': {'optimizable': True, 'value': 1.},
                                         'terms_number': {'optimizable': False, 'value': 5.},
                                         'max_factors_in_term': {'optimizable': False, 'value': 1.}},
                 interelement_operator: Callable = np.add):
        """

        Class for the single equation for the dynamic system.

        attributes:
            structure : list of Term objects \r\n
            List, containing all terms of the equation; first 2 terms are reserved for constant value and the input function;

            target_idx : int \r\n
            Index of the target term, selected in the Split phase;

            target : 1-d array of float \r\n
            values of the Term object, reshaped into 1-d array, designated as target for application in sparse regression;

            features : matrix of float \r\n
            matrix, composed of terms, not included in target, value columns, designated as features for application in sparse regression;

            fitness_value : float \r\n
            Inverse value of squared error for the selected target 2function and features and discovered weights; 

            estimator : sklearn estimator of selected type \r\n

        parameters:

            Matrix of derivatives: first axis through various orders/coordinates in order: ['1', 'f', all derivatives by one coordinate axis
            in increasing order, ...]; second axis: time, further - spatial coordinates;

            tokens : list of strings \r\n
            Symbolic forms of functions, including derivatives;

            terms_number : int, base value of 6 \r\n
            Maximum number of terms in the discovered equation; 

            max_factors_in_term : int, base value of 2\r\n
            Maximum number of factors, that can form a term (e.g. with 2: df/dx_1 * df/dx_2)

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

    def reset_explaining_term(self, term_idx=0):
        for idx, term in enumerate(self.structure):
            if idx == term_idx:
                # print(f'Checking if {self.main_var_to_explain} is in {term.name}')
                assert term.contains_family(
                    self.main_var_to_explain), 'Trying explain a variable with term without right family.'
                term.descr_variable_marker = self.main_var_to_explain
            else:
                term.descr_variable_marker = False

    def __eq__(self, other):
        if self.weights_final_evald and other.weights_final_evald:
            return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure])
                    and all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure])
                    and len(other.structure) == len(self.structure)
                    and np.all(np.isclose(self.weights_final, other.weights_final)))
        else:
            return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure])
                    and all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure])
                    and len(other.structure) == len(self.structure))

    def contains_deriv(self, family=None):
        return any([term.contains_deriv(family) for term in self.structure])

    def contains_family(self, family):
        return any([term.contains_family(family) for term in self.structure])

    @property
    def forbidden_token_labels(self):
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
        # TODO: rewrite for an arbitrary equation property check
        if not (deriv or mandatory_family):
            raise ValueError('No property passed for restoration.')
        while True:
            print(
                f'Restoring containment of {mandatory_family} in {self.text_form}.')
            replacement_idx = np.random.randint(low=0, high=len(self.structure))
            mf_marker = mandatory_family if mandatory_family else None
            temp = Term(self.pool, mandatory_family=mf_marker,
                        max_factors_in_term=self.metaparameters['max_factors_in_term']['value'])
            if deriv and temp.contains_deriv():
                self.structure[replacement_idx] = temp
                break
            elif mandatory_family and temp.contains_family(self.main_var_to_explain):
                self.structure[replacement_idx] = temp
                break
            else:
                print('temp', temp.name, 'self.main_var_to_explain',
                      self.main_var_to_explain)

    def reconstruct_by_right_part(self, right_part_idx):
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
        self._target = self.structure[self.target_idx].evaluate(normalize, grids=grids)
        
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
                    self._features = self.structure[feature_indexes[feat_idx]].evaluate(normalize, grids=grids)
                else: #if feat_idx != 0:
                    temp = self.structure[feature_indexes[feat_idx]].evaluate(normalize, grids=grids)
                    self._features = np.vstack([self._features, temp])
                # else:
                    # continue
            if self._features.ndim == 1:
                self._features = np.expand_dims(self._features, 1).T
            temp_feats = np.vstack([self._features, np.ones(self._features.shape[1])])
            self._features = np.transpose(self._features)
            temp_feats = np.transpose(temp_feats)
        else:
            self._features = None
            
        if return_val:
            self.prev_normalized = normalize
            if normalize:
                elem1 = np.expand_dims(self._target, axis=1)
                value = np.add(elem1, - reduce(lambda x, y: np.add(x, y), [np.multiply(self.weights_internal[idx_full], temp_feats[:, idx_sparse])
                                                                           for idx_sparse, idx_full in enumerate(feature_indexes)]))
                                                                           # for feature_idx, weight in np.ndenumerate(self.weights_internal)]))
            else:
                elem1 = np.expand_dims(self._target, axis=1)
                if self._features is None:
                    feature_list = [np.multiply(self.weights_final[idx_full], temp_feats[:, idx_sparse])
                                    for idx_sparse, idx_full in enumerate(feature_indexes)]
                else:
                    feature_list = 0               
                value = np.add(elem1, feature_list)
                                               
            return value, self._target, self._features
        else:
            return None, self._target, self._features

    def reset_state(self, reset_right_part: bool = True):
        if reset_right_part:
            self.right_part_selected = False
        self.weights_internal_evald = False
        self.weights_final_evald = False
        self.fitness_calculated = False
        self.solver_form_defined = False

    # @ResetEquationStatus(reset_input = False, reset_output = True)
    @HistoryExtender('\n -> was copied by deepcopy(self)', 'n')
    def __deepcopy__(self, memo=None):
        # print(f'Deepcopying equation {self}')
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct

        attrs_to_avoid_copy = ['_features', '_target']
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
        new_equation.weights_internal_evald = self.weights_internal_evald
        new_equation.weights_final_evald = self.weights_final_evald
        new_equation.right_part_selected = self.right_part_selected
        new_equation.fitness_calculated = self.fitness_calculated
        new_equation.solver_form_defined = False

        try:
            new_equation._fitness_value = self._fitness_value
        except AttributeError:
            pass

    def add_history(self, add):
        self._history += add

    @property
    def history(self):
        return self._history

    @property
    def fitness_value(self):
        return self._fitness_value

    @fitness_value.setter
    def fitness_value(self, val):
        self._fitness_value = val

    def penalize_fitness(self, coeff=1.):
        self._fitness_value = self._fitness_value*coeff

    @property
    def weights_internal(self):
        if self.weights_internal_evald:
            return self._weights_internal
        else:
            raise AttributeError(
                'Internal weights called before initialization')

    @weights_internal.setter
    def weights_internal(self, weights):
        self._weights_internal = weights
        self.weights_internal_evald = True
        self.weights_final_evald = False

    @property
    def weights_final(self):
        if self.weights_final_evald:
            return self._weights_final
        else:
            print(self.text_form)
            raise AttributeError('Final weights called before initialization')

    @weights_final.setter
    def weights_final(self, weights):
        self._weights_final = weights
        self.weights_final_evald = True

    @property
    def latex_form(self):
        form = r""
        for term_idx in range(len(self.structure)):
            if term_idx != self.target_idx:
                form += str(self.weights_final[term_idx]) if term_idx < self.target_idx else str(
                    self.weights_final[term_idx-1])
                form += ' * ' + self.structure[term_idx].latex_form + ' + '
        form += str(self.weights_final[-1]) + ' = ' + \
            self.structure[self.target_idx].text_form
        return form

    @property
    def text_form(self):
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
    
    # def solver_form(self, grids: list = None):
    #     raise DeprecationWarning('To be removed from the framework!')
    #     if self.solver_form_defined:
    #         # print(self.text_form)
    #         return self._solver_form
    #     else:
    #         self._solver_form = []
    #         for term_idx in range(len(self.structure)):
    #             if term_idx != self.target_idx:
    #                 term_form = self.structure[term_idx].solver_form
    #                 weight = self.weights_final[term_idx] if term_idx < self.target_idx else self.weights_final[term_idx-1]
    #                 term_form[0] = term_form[0] * weight
    #                 term_form[0] = torch.flatten(term_form[0]).unsqueeze(
    #                     1).type(torch.FloatTensor)
    #                 self._solver_form.append(term_form)

    #         free_coeff_weight = torch.from_numpy(np.full_like(a=global_var.grid_cache.get('0'),
    #                                                           fill_value=self.weights_final[-1]))
    #         free_coeff_weight = torch.flatten(free_coeff_weight).unsqueeze(1).type(torch.FloatTensor)
    #         target_weight = torch.from_numpy(np.full_like(a=global_var.grid_cache.get('0'),
    #                                                       fill_value=-1.))
    #         target_form = self.structure[self.target_idx].solver_form
    #         target_form[0] = target_form[0] * target_weight
    #         target_form[0] = torch.flatten(target_form[0]).unsqueeze(1).type(torch.FloatTensor)

    #         self._solver_form.append([free_coeff_weight, [None], 0])
    #         self._solver_form.append(target_form)
    #         self.solver_form_defined = True
    #         return self._solver_form

    @property
    def state(self):
        return self.text_form

    @property
    def described_variables(self):
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
        del self.model
        del self._solver_form
        self.solver_form_defined = False
        gc.collect()


def solver_formed_grid(training_grid=None):
    if training_grid is None:
        keys, training_grid = global_var.grid_cache.get_all()
    else:
        keys, _ = global_var.grid_cache.get_all()

    assert len(keys) == training_grid[0].ndim, 'Mismatching dimensionalities'

    training_grid = np.array(training_grid).reshape((len(training_grid), -1))
    return torch.from_numpy(training_grid).T.type(torch.FloatTensor)


def check_metaparameters(metaparameters: dict):
    metaparam_labels = ['terms_number', 'max_factors_in_term', 'sparsity']
    return True
    # TODO: fix this check
    # if any([((label not in metaparameters.keys()) and ) for label in metaparam_labels]):
    #     print('required metaparameters:', metaparam_labels, 'metaparameters:', metaparameters)
    #     raise ValueError('Only partial metaparameter vector has been passed.')


class SoEq(moeadd.MOEADDSolution):
    def __init__(self, pool: TFPool, metaparameters: dict):
        '''

        Parameters
        ----------
        pool : epde.interface.token_familiy.TFPool
            Pool, containing token families for the equation search algorithm.
        metaparameters : dict
            Metaparameters dictionary for the search. Key - label of the parameter (e.g. 'sparsity'),
            value - tuple, containing flag for metaoptimization and initial value.

        Returns
        -------
        None.

        '''
        check_metaparameters(metaparameters)

        self.metaparameters = metaparameters
        self.tokens_for_eq = TFPool(pool.families_demand_equation)
        self.tokens_supp = TFPool(pool.families_equationless)
        self.moeadd_set = False

        self.vars_to_describe = [token_family.ftype for token_family in self.tokens_for_eq.families] # Made list from set

    def use_default_multiobjective_function(self):
        from epde.eq_mo_objectives import generate_partial, equation_fitness, equation_complexity_by_factors
        complexity_objectives = [generate_partial(equation_complexity_by_factors, eq_key)
                                 for eq_key in self.vars_to_describe]  # range(len(self.tokens_for_eq))]
        # range(len(self.tokens_for_eq))]
        quality_objectives = [generate_partial(
            equation_fitness, eq_key) for eq_key in self.vars_to_describe]
        self.set_objective_functions(
            quality_objectives + complexity_objectives)

    def use_default_singleobjective_function(self):
        from epde.eq_mo_objectives import generate_partial, equation_fitness
        quality_objectives = [generate_partial(equation_fitness, eq_key) for eq_key in self.vars_to_describe]#range(len(self.tokens_for_eq))]
        self.set_objective_functions(quality_objectives)

    def set_objective_functions(self, obj_funs):
        '''
        Method to set the objective functions to evaluate the "quality" of the system of equations.

        Parameters:
        -----------
            obj_funs - callable or list of callables;
            function/functions to evaluate quality metrics of system of equations. Can return a single 
            metric (for example, quality of the process modelling with specific system), or 
            a list of metrics (for example, number of terms for each equation in the system).
            The function results will be flattened after their application. 

        '''
        assert callable(obj_funs) or all([callable(fun) for fun in obj_funs])
        self.obj_funs = obj_funs

    def matches_complexitiy(self, complexity : Union[int, list]):
        if isinstance(complexity, (int, float)):    
            complexity = [complexity,]
        
        if not isinstance(complexity, list) or len(self.vars_to_describe) != len(complexity):
            raise ValueError('Incorrect list of complexities passed.')
        
        return list(self.obj_fun[-len(complexity):]) == complexity        

    def create_equations(self):
        structure = {}

        token_selection = self.tokens_supp
        current_tokens_pool = token_selection + self.tokens_for_eq

        for eq_idx, variable in enumerate(self.vars_to_describe):
            structure[variable] = Equation(current_tokens_pool, basic_structure=[],
                                           var_to_explain=variable,
                                           metaparameters=self.metaparameters)

        self.vals = Chromosome(structure, params={key: val for key, val in self.metaparameters.items()
                                                  if val['optimizable']})
        moeadd.MOEADDSolution.__init__(self, self.vals, self.obj_funs)
        self.moeadd_set = True

    @staticmethod
    def equation_opt_iteration(population, evol_operator, population_size, iter_index, unexplained_vars, strict_restrictions=True):
        for equation in population:
            if equation.described_variables in unexplained_vars:
                equation.penalize_fitness(coeff=0.)
        population = population_sort(population)
        population = population[:population_size]
        gc.collect()
        population = evol_operator.apply(population, unexplained_vars)
        return population

    def evaluate(self, normalize=True, grids: list = None):
        raise DeprecationWarning('Evaluation of system is not necessary')
        if len(self.vals) == 1:
            value = [equation.evaluate(normalize, return_val=True)[0] for equation in self.vals][0]
            # self.vals[0].evaluate(normalize = normalize, return_val = True)[0]
        else:
            value = np.sum([equation.evaluate(normalize, return_val=True)[0] for equation in self.vals])
        value = np.sum(np.abs(value))
        return value

    @property
    def obj_fun(self):
        return np.array(flatten([func(self) for func in self.obj_funs]))

    def __call__(self):
        assert self.moeadd_set, 'The structure of the equation is not defined, therefore no moeadd operations can be called'
        return self.obj_fun

    @property
    def text_form(self):
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
        assert self.moeadd_set, 'The structure of the equation is not defined, therefore no moeadd operations can be called'
        return (all([any([other_elem == self_elem for other_elem in other.vals]) for self_elem in self.vals]) and
                all([any([other_elem == self_elem for self_elem in self.vals]) for other_elem in other.vals]) and
                len(other.vals) == len(self.vals))  # or all(np.isclose(self.obj_fun, other.obj_fun)

    @property
    def latex_form(self):
        form = r"\begin{eqnarray*}"
        for equation in self.vals:
            form += equation.latex_form + r", \\ "
        form += r"\end{eqnarray*}"

    def __hash__(self):
        # print(f'GETTING HASH VALUE OF SoEq: {self.vals.hash_descr}')
        return hash(self.vals.hash_descr)

    def __deepcopy__(self, memo=None):
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
        for equation in self.vals:
            equation.reset_state(reset_right_part)

    def copy_properties_to(self, objective):
        for eq_label in self.vals.equation_keys:  # Not the best code possible here
            self.vals[eq_label].copy_properties_to(objective.vals[eq_label])

    def solver_params(self, full_domain, grids=None):
        '''
        Returns solver form, grid and boundary conditions
        '''
        # if len(self.vals) > 1:
        #     raise Exception('Solver form is defined only for a "system", that contains a single equation.')
        # else:
        #     form = self.vals[0].solver_form()
        #     grid = solver_formed_grid()
        #     bconds = self.vals[0].boundary_conditions(full_domain = full_domain)
        #     return form, grid, bconds

        equation_forms = []
        bconds = []

        for idx, equation in enumerate(self.vals):
            equation_forms.append(equation.solver_form(grids=grids))
            bconds.append(equation.boundary_conditions(full_domain=full_domain, grids=grids,
                                                       index=idx))

        return equation_forms, solver_formed_grid(grids), bconds

    def __iter__(self):
        return SoEqIterator(self)

    @property
    def fitness_calculated(self):
        return all([equation.fitness_calculated for equation in self.vals])

    def save(self, file_name='epde_systems.pickle'):
        dir = os.getcwd()
        with open(file_name, 'wb') as file:
            to_save = ([equation.text_form for equation in self.vals],
                       self.tokens_for_eq + self.tokens_supp)
            pickle.dump(obj=to_save, file=file)


class SoEqIterator(object):
    def __init__(self, system: SoEq):
        self._idx = 0
        self.system = system
        self.keys = list(system.vars_to_describe)

    def __next__(self):
        if self._idx < len(self.keys):
            res = self.system.vals[self.keys[self._idx]]
            self._idx += 1
            return res
        else:
            raise StopIteration
