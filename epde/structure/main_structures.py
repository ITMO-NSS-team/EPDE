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
from copy import deepcopy
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

from epde import _loop_stats
from epde.decorators import HistoryExtender, BoundaryExclusion
from epde.evaluators import simple_function_evaluator
from epde.interface.token_family import TFPool
from epde.preprocessing.domain_pruning import DomainPruner

from epde.structure.encoding import Chromosome
from epde.structure.factor import Factor
from epde.structure.structure_template import ComplexStructure, check_uniqueness, _deepcopy_slots
from epde.supplementary import filter_powers, normalize_ts, population_sort, flatten, rts, exp_form, minmax_normalize, retry_until_unique


_DEFAULT_EQUATION_METAPARAMETERS = {
    'sparsity':            {'optimizable': True,  'value': 1.},
    'terms_number':        {'optimizable': False, 'value': 5.},
    'max_factors_in_term': {'optimizable': False, 'value': 1.},
}


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

    def __init__(self, pool: 'TFPool', passed_term=None, mandatory_family: str = None,
                 max_factors_in_term: Union[int, dict] = 1,
                 create_derivs: bool = False, interelement_operator: Callable = np.multiply,
                 collapse_powers: bool = True):
        """
        Construct a single Term (a product of Factor objects).

        If ``passed_term`` is None, the term is randomized from ``pool`` honoring
        ``max_factors_in_term`` and any ``mandatory_family`` constraint. If
        ``passed_term`` is a list/str, the term is built from the supplied factors
        and ``collapse_powers`` controls whether identical factors are collapsed
        into a single factor with summed power.
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

    # TODO: non-urgent, make self.descr_variable_marker setting for defined parameter

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
        return self._descr_variable_marker

    @descr_variable_marker.setter
    def descr_variable_marker(self, marker: False):
        if not marker or isinstance(marker, str):
            self._descr_variable_marker = marker
        else:
            raise ValueError('Described variable marker shall be a family label (i.e. "u") of "False"')

    @_loop_stats.timed('Term.evaluate')
    def evaluate(self, structural, grids=None):
        assert global_var.tensor_cache is not None, 'Currently working only with connected cache'
        normalize = structural
        if self.saved[structural] or (self.factors_labels, normalize) in global_var.tensor_cache:
            value = global_var.tensor_cache.get(self.factors_labels, normalized=normalize,
                                                saved_as=self.saved_as[normalize])
            value = value.reshape(-1)
            return value
        else:
            self.prev_normalized = normalize
            value = super().evaluate(structural)
            if normalize:
                # value = (value - np.mean(value)) / np.std(value)
                # value = value / np.linalg.norm(value, 2)
                # value = minmax_normalize(value)
                value = 2 * (value - value.min()) / (value.max() - value.min()) - 1

                # value = np.ones_like(value)
                # for factor in self.structure:
                #     factor_value = factor.evaluate()
                #     factor_value_normalized = minmax_normalize(factor_value)
                #     value *= factor_value_normalized
            if grids is None:
                # Cache key is (factors_labels, normalize). factors_labels is a
                # frozenset of Factor.structural_label, which already
                # bucket-quantizes continuous params, so the cached product
                # applies to every term sharing this signature regardless of
                # how many params each individual factor carries. The legacy
                # len(factor.params)==1 gate predated the structural-label
                # quantization and was overly conservative.
                self.saved[normalize] = global_var.tensor_cache.add(self.factors_labels, value, normalized=normalize)
                if self.saved[normalize]:
                    self.saved_as[normalize] = self.factors_labels
            value = value.reshape(-1)
            return value

    def filter_tokens_by_right_part(self, reference_target, equation, equation_position,
                                    max_retries: int = 100):
        warnings.warn(message='Tokens can no longer be set as right-part-unique',
                      category=DeprecationWarning)
        taken_tokens = [factor.label for factor in reference_target.structure
			 if factor.status['unique_for_right_part']]
        meaningful_taken = any([factor.status['meaningful'] for factor in reference_target.structure
                                if factor.status['unique_for_right_part']])

        new_term = None
        for accept_term_try in range(1, max_retries + 1):
            new_term = copy.deepcopy(self)
            for factor_idx, factor in enumerate(new_term.structure):
                if factor.label in taken_tokens:
                    new_term.reset_occupied_tokens()
                    _, new_term.structure[factor_idx] = self.pool.create(create_meaningful=meaningful_taken,
                                                                         occupied=new_term.occupied_tokens_labels + taken_tokens)
            if len(equation.terms_labels) == len(equation.structure):
                self.structure = new_term.structure
                self.structure = filter_powers(self.structure)
                self.reset_saved_state()
                return
            if accept_term_try == 10 and global_var.verbose.show_warnings:
                warnings.warn('Can not create unique term, while filtering equation tokens in regards to the right part.')
            if accept_term_try >= 10:
                self.randomize(forbidden_factors=new_term.occupied_tokens_labels + taken_tokens)

        last_attempt_name = new_term.name if new_term is not None else '<no candidate>'
        raise RuntimeError(
            f'filter_tokens_by_right_part: failed to create unique term after '
            f'{max_retries} retries. Last attempted: {last_attempt_name} for '
            f'{equation.text_form} with respect to {reference_target.name}')

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

    def iter_available_tokens(self):
        """Generator equivalent of `available_tokens`; yields one filtered family at a time.

        Allows consumers that only need to iterate (rather than realize the full
        list) to avoid the per-call list materialization. Each yielded family is
        still deepcopied — that's the unavoidable per-element cost.
        """
        for token in self.pool.families:
            if not all([label in self.occupied_tokens_labels for label in token.tokens]):
                token_new = copy.deepcopy(token)
                token_new.tokens = [
                    label for label in token.tokens if label not in self.occupied_tokens_labels]
                yield token_new

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

    @property
    def latex_form(self):
        form = reduce(lambda x, y: x + r' \cdot ' + y, [factor.latex_name for
                                                        factor in self.structure])
        return form

    def contains_deriv(self, variable=None):
        if variable is None:
            return sum([factor.is_deriv and factor.deriv_code != [None,] and
                        factor.evaluator._evaluator == simple_function_evaluator
                        for factor in self.structure]) == 1
        else:
            return sum([factor.variable == variable and factor.is_deriv and factor.deriv_code != [None,] and
                        factor.evaluator._evaluator == simple_function_evaluator
                        for factor in self.structure]) == 1

    def contains_variable(self, variable):
        return any([factor.variable == variable for factor in self.structure])

    def contains_meaningful(self):
        return any([factor.status['meaningful'] for factor in self.structure])

    def contains_t_derivative(self):
        return any([factor.deriv_code[0] == 0 if not factor.deriv_code is None else False for factor in self.structure])

    def __eq__(self, other):
        return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure])
                and all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure])
                and len(other.structure) == len(self.structure))

    @HistoryExtender('\n -> was copied by deepcopy(self)', 'n')
    def __deepcopy__(self, memo=None):
        # ``pool`` is the population-wide TFPool, set once and never
        # mutated; sharing by ref skips a recursive copy of every token
        # family in every Term clone.
        return _deepcopy_slots(self, memo, attrs_to_share_by_ref=('pool',))

    @property
    def factors_labels_without_power(self) -> frozenset:
        """Return a frozenset of structural labels with the ``power`` param dropped.

        Identity is delegated to ``Factor.structural_label_without_power``,
        which quantizes continuous-tolerance params (e.g. trig ``freq``)
        into bucket indices so structural dedup stays consistent with
        ``Factor.__eq__``.
        """
        return frozenset(factor.structural_label_without_power for factor in self.structure)

    @property
    def factors_labels(self) -> frozenset:
        """Return a frozenset of structural labels for each factor in the term.

        Identity is delegated to ``Factor.structural_label``, which
        bucketises continuous-tolerance params (e.g. trig ``freq``) so
        within-bucket differences don't fracture structural identity.
        Used as a hashable identity for set/membership checks.
        """
        return frozenset(factor.structural_label for factor in self.structure)


class Equation(ComplexStructure):
    __slots__ = ['_history', 'structure', 'interelement_operator', 'n_immutable', 'pool',
                  # '_target', '_features', 'saved', 'saved_as','max_factors_in_term', 'operator',
                 'target_idx', 'right_part_selected', '_weights_final', 'weights_final_evald', 'simplified', 'is_correct_right_part',
                 '_weights_internal', 'weights_internal_evald', 'fitness_calculated', 'stability_calculated', 'aic_calculated', 'solver_form_defined',
                 '_fitness_value', '_coefficients_stability', '_aic', 'metaparameters', 'main_var_to_explain',
                 '_eval_cache', '_cached_sw_weights', '_cached_vc_score',
                 '_terms_labels_cache', '_terms_labels_without_power_cache',
                 '_gram_super'] # , '_solver_form'


    def __init__(self, pool: TFPool, basic_structure: Union[list, tuple, set], var_to_explain: str = None,
                 metaparameters: dict = None,
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

            max_factors_in_term : int, base value of 2\r\n
            Maximum number of factors, that can form a term (e.g. with 2: df/dx_1 * df/dx_2)

        """
        super().__init__(interelement_operator)
        self.reset_state()

        if metaparameters is None:
            metaparameters = copy.deepcopy(_DEFAULT_EQUATION_METAPARAMETERS)

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
        max_iter = 100
        for i in range(len(basic_structure), int(self.metaparameters['terms_number']['value'])):
            new_term = Term(self.pool, max_factors_in_term=self.metaparameters['max_factors_in_term']['value'],
                            mandatory_family=None, passed_term=None)
            def _term_mutate():
                new_term.randomize()
                new_term.reset_saved_state()
            success, _ = retry_until_unique(
                predicate=lambda: new_term.factors_labels not in self.terms_labels,
                mutate=_term_mutate,
                max_iter=max_iter,
                stats_name='Equation.__init__.unique_term',
            )
            if not success:
                # Pool can't yield a unique term against the current
                # structure -- stop, don't try further slots. Subsequent
                # ``new_term`` draws would face the same exhausted pool,
                # so the only honest outcome is a shorter equation.
                # (D1 raises for the same exhaustion class in
                # InitialParetoLevelSorting; here we warn-accept because
                # ``Equation.__init__`` is invoked during the initial
                # population draw and aborting fit() entirely would be
                # surprising user-facing behavior.)
                warnings.warn(
                    f"Equation.__init__: no unique term in {max_iter} attempts at slot {i}; "
                    "pool may be exhausted -- stopping with a shorter equation."
                )
                break
            self.structure.append(new_term)
            self._invalidate_label_cache()

        for idx, _ in enumerate(self.structure):
            self.structure[idx].use_cache()
#        self.coefficients_stability = np.inf

    def randomize(self):
        self.__init__(self.pool, [], self.main_var_to_explain, metaparameters=self.metaparameters)
        self.reset_saved_state()

    def manual_reconst(self, attribute:str, value, except_attrs:dict):
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
            self._invalidate_label_cache()

    def reset_explaining_term(self, term_idx=0):
        for idx, term in enumerate(self.structure):
            if idx == term_idx:
                assert term.contains_variable(
                    self.main_var_to_explain), f'Trying explain a variable {self.main_var_to_explain} \
                                                 with term without right family.'
                term.descr_variable_marker = self.main_var_to_explain
            else:
                term.descr_variable_marker = False

    def remove_zero_terms(self):
        if self.weights_internal_evald:
            wi = self._weights_internal
            # weights_internal: one coef per non-target term, optionally a
            # trailing intercept slot (VWSR: len == m+1; legacy LASSO: len == m).
            m = len(self.structure) - 1
            has_intercept = len(wi) == m + 1
            zero_terms = []        # structure indices to drop
            zero_coef_pos = []     # matching coefficient indices to drop
            target_bias = 0
            for i in range(len(self.structure)):
                if i == self.target_idx:
                    continue
                idx = i if i < self.target_idx else i - 1
                if wi[idx] == 0:
                    target_bias += 1 if i < self.target_idx else 0
                    zero_terms.append(i)
                    zero_coef_pos.append(idx)
            if zero_terms:
                self.structure = [term for term_idx, term in enumerate(self.structure) if term_idx not in zero_terms]
                self.target_idx -= target_bias
                # Compact weights_internal in lockstep with the structure so
                # later position-indexed reads (active_terms_labels, the
                # intercept slot, re-prune) stay aligned. weights_final already
                # holds only the non-zero entries, so it needs no surgery.
                if has_intercept or zero_coef_pos:
                    zcp = set(zero_coef_pos)
                    kept = [wi[j] for j in range(m) if j not in zcp]
                    if has_intercept:
                        kept.append(wi[-1])
                    self._weights_internal = np.array(kept)
                # ``_invalidate_label_cache`` also wipes _eval_cache, which
                # is essential here: the right-part-selector's per-target
                # sweep populates the cache keyed on target_idx, and the
                # adjusted target_idx above can collide with a swept value.
                # ``_cached_sw_weights`` was computed for the surviving
                # features and still aligns with the new structure, so it
                # is preserved.
                self._invalidate_label_cache()


    def __eq__(self, other):
        if self.weights_final_evald and other.weights_final_evald:
            return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure])
                    and all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure])
                    and len(other.structure) == len(self.structure)
                    and len(self.weights_final) == len(other.weights_final)
                    and np.all(np.isclose(self.weights_final, other.weights_final)))
        else:
            return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure])
                    and all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure])
                    and len(other.structure) == len(self.structure))

    def contains_deriv(self, variable=None):
        return any([term.contains_deriv(variable) for term in self.structure])

    def contains_variable(self, variable):
        return any([term.contains_variable(variable) for term in self.structure])

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

    def restore_property(self, deriv: bool = False, mandatory_family: bool = False, t_derivative: bool = False):
        # TODO: non-urgent, rewrite for an arbitrary equation property check
        if not (deriv or mandatory_family):
            raise ValueError('No property passed for restoration.')
        # Bound both the outer and the inner sampling loops, and reject any
        # candidate whose factor signature would collide with another
        # existing term -- see feedback-structure-dedup memory.
        max_outer = 200
        max_inner = 100

        # Prefer ADDING the new property-carrying term so existing structure
        # is preserved; fall back to REPLACING a random term only when the
        # ``terms_number`` cap is already reached.
        terms_cap = int(self.metaparameters['terms_number']['value'])
        can_add = len(self.structure) < terms_cap

        def _slot_duplicate(idx, candidate):
            """Duplicate check that ignores the slot we're about to write
            to. ``idx=None`` => add path: check against ALL existing terms.
            ``idx=k``       => replace path: skip slot k.
            """
            sig = candidate.factors_labels
            return any((idx is None or j != idx) and other.factors_labels == sig
                       for j, other in enumerate(self.structure))

        def _commit(idx, term):
            if idx is None:
                self.structure.append(term)
            else:
                self.structure[idx] = term
            self._invalidate_label_cache()

        mf_marker = self.main_var_to_explain if mandatory_family else None
        max_factors = self.metaparameters['max_factors_in_term']['value']
        outer_attempts = 0
        for _ in range(max_outer):
            outer_attempts += 1
            target_idx = None if can_add else np.random.randint(low=0, high=len(self.structure))
            temp = Term(self.pool, mandatory_family=mf_marker, max_factors_in_term=max_factors)
            if t_derivative:
                inner = 0
                while not temp.contains_t_derivative() and inner < max_inner:
                    temp = Term(self.pool, mandatory_family=mf_marker, max_factors_in_term=max_factors)
                    inner += 1
                _loop_stats.record('restore_property.t_derivative_inner', inner, max_inner)
                if not temp.contains_t_derivative():
                    continue
                if _slot_duplicate(target_idx, temp):
                    continue
                _commit(target_idx, temp)
                _loop_stats.record('restore_property.outer', outer_attempts, max_outer)
                return
            if deriv and mandatory_family and temp.contains_deriv() and temp.contains_variable(self.main_var_to_explain):
                if _slot_duplicate(target_idx, temp):
                    continue
                _commit(target_idx, temp)
                _loop_stats.record('restore_property.outer', outer_attempts, max_outer)
                return
            elif deriv and temp.contains_deriv(self.main_var_to_explain) and not mandatory_family:
                if _slot_duplicate(target_idx, temp):
                    continue
                _commit(target_idx, temp)
                _loop_stats.record('restore_property.outer', outer_attempts, max_outer)
                return
            elif mandatory_family and temp.contains_variable(self.main_var_to_explain) and not deriv:
                if _slot_duplicate(target_idx, temp):
                    continue
                _commit(target_idx, temp)
                _loop_stats.record('restore_property.outer', outer_attempts, max_outer)
                return
        _loop_stats.record('restore_property.outer', outer_attempts, max_outer)
        # Cap-hit is a configuration-failure signal, not a probabilistic
        # search miss. In healthy configs the outer loop completes in
        # single-digit attempts (observed mean 2.8-3.3, max 16 across
        # every historical thesis run). Reaching ``max_outer`` means the
        # token pool genuinely cannot produce a property-carrying term --
        # e.g. ``max_derivative_order=0`` in every domain, the derivative
        # family is missing from the pool, or ``mandatory_family`` wiring
        # is wrong. Raise loudly so the user can fix the config, rather
        # than silently returning a property-less equation.
        raise RuntimeError(
            f"Equation.restore_property: could not install requested "
            f"property (deriv={deriv}, mandatory_family={mandatory_family}, "
            f"t_derivative={t_derivative}) for main_var="
            f"{self.main_var_to_explain!r} after {max_outer} sampling "
            f"attempts. This is a configuration error -- verify that "
            f"the token pool exposes a derivative family for this "
            f"variable (max_derivative_order > 0 in at least one "
            f"domain, derivative tokens enrolled, mandatory_family "
            f"wiring correct)."
        )

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

    @_loop_stats.timed('Equation.evaluate')
    def evaluate(self, normalize: bool = True, return_val: bool = False,
                 grids: list = None) -> Tuple:
        """Evaluate the equation and return (value, target, features).

        ``target`` is the LHS term values; ``features`` is a 2-D matrix of the
        non-target term evaluations (``None`` if every other term is zero-weight
        and ``normalize=False``); ``value`` is the residual when
        ``return_val=True`` else ``None``.

        Caching policy: results are cached per
        (normalize, return_val, grids-is-None, target_idx) when
        ``grids is None`` AND ``normalize`` is True. The ``normalize=False``
        branch additionally filters ``feature_indexes`` by the current
        ``weights_internal`` (lines below); since callers update weights
        between successive ``evaluate(normalize=False)`` calls, caching that
        branch would risk returning stale (target, features) tuples with
        out-of-date feature masks. ``normalize=True`` is weight-independent
        and is the path benefitting from cache hits (sparsity then L2LRFitness
        both call ``evaluate(normalize=True)`` in one fitness invocation).
        """
        cacheable = (grids is None) and normalize
        cache_key = (normalize, return_val, grids is None, self.target_idx)
        if cacheable and hasattr(self, '_eval_cache') and cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        target = self.structure[self.target_idx].evaluate(False, grids=grids)

        if normalize:
            feature_indexes = [i for i in range(len(self.structure)) if i != self.target_idx]
        else:
            feature_indexes = []
            for idx in range(len(self.structure)):
                if idx == self.target_idx:
                    continue
                shifted = idx if idx < self.target_idx else idx - 1
                if self.weights_internal[shifted] != 0:
                    feature_indexes.append(idx)
        if len(feature_indexes) > 0:
            feat_list = [self.structure[idx].evaluate(False, grids=grids) for idx in feature_indexes]
            features = np.vstack(feat_list)
            if features.ndim == 1:
                features = np.expand_dims(features, 1).T
            features = np.transpose(features)
        else:
            features = None

        if return_val:
            temp_feats = np.vstack([features, np.ones(features.shape[1])])
            temp_feats = np.transpose(temp_feats)
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
            result = (value, target, features)
        else:
            result = (None, target, features)

        if cacheable:
            if not hasattr(self, '_eval_cache'):
                self._eval_cache = {}
            self._eval_cache[cache_key] = result
        return result

    def reset_state(self, reset_right_part: bool = True) -> None:
        """Drop all cached evaluation/fitness state on this Equation.

        Call after any structural mutation (or to discard a stale fitness/AIC
        evaluation). Set ``reset_right_part=False`` to keep target_idx and
        weight assignments — useful when only the LHS-derived caches need
        clearing.
        """
        if reset_right_part:
            self.right_part_selected = False
            self.is_correct_right_part = False
            self.simplified = False
            self.weights_internal_evald = False
            self.weights_internal = None
            # self.weights_final_evald = False
            self.weights_final = None
        # self.weights_internal_evald = False
        # self.weights_internal = None
        self.weights_final_evald = False
        # self.weights_final = None
        self.fitness_calculated = False
        self.fitness_value = None
        self.stability_calculated = False
        self.coefficients_stability = None
        self.aic_calculated = False
        self.solver_form_defined = False
        self._eval_cache = {}
        # consumed by epde.operators.common.fitness.L2LRFitness; resets here.
        self._cached_sw_weights = None
        # vcoef analogue of _cached_sw_weights: per-term stability scores from
        # the sparsity gram_setup, summed as the stability objective in fitness.
        self._cached_vc_score = None
        self._terms_labels_cache = None
        self._terms_labels_without_power_cache = None
        # Tier 3 super-Gram cache (set by EqRightPartSelector for the
        # term-sweep, never persists past one sweep). Structural reset
        # invalidates it.
        self._gram_super = None

    def _invalidate_label_cache(self):
        """Drop memoized caches keyed on the current structure; call after
        ``self.structure`` (or ``self.target_idx``) mutates.

        Covers both the terms-labels caches and the per-evaluation
        ``_eval_cache`` populated by :meth:`evaluate`. The eval cache key
        includes ``self.target_idx`` and the cached value depends on which
        terms occupy ``self.structure``, so any structural mutation must
        drop it -- otherwise callers like the right-part-selector's
        per-target sweep can leave stale entries that survive into the
        post-RPS fitness call (e.g. after ``remove_zero_terms`` adjusts
        ``target_idx`` onto a value the sweep already cached).
        """
        self._terms_labels_cache = None
        self._terms_labels_without_power_cache = None
        if hasattr(self, '_eval_cache'):
            self._eval_cache = {}
        # Super-Gram is built from term evaluations; any structural
        # change invalidates the matched-rank assumption.
        self._gram_super = None


    @HistoryExtender('\n -> was copied by deepcopy(self)', 'n')
    def __deepcopy__(self, memo=None):
        # Volatile slot caches are about to be invalidated by the next
        # mutation anyway -- skip the deep-copy. ``pool`` is the
        # population-wide TFPool (single instance, never mutated) --
        # share by reference. ``metaparameters`` IS mutated via
        # ``encoding.Gene.__setitem__``
        # (see test_main_structures_characterization::test_metaparameters_*)
        # and MUST stay deep-copied.
        new_struct = _deepcopy_slots(
            self, memo,
            attrs_to_avoid_copy=(
                '_cached_sw_weights', '_cached_vc_score',
                '_terms_labels_cache', '_terms_labels_without_power_cache',
                '_gram_super',
            ),
            attrs_to_share_by_ref=('pool',),
        )
        # ``_eval_cache`` must round-trip as a *fresh empty dict* (separate
        # ref, equal == content); the cache content itself can be heavy
        # (cached evaluated tensors) and is the cheapest thing to discard
        # since the next ``evaluate()`` repopulates lazily.
        new_struct._eval_cache = {}
        return new_struct

    def clone_shell(self):
        """Return a deepcopied equation **without** copying ``structure``.

        Useful when the caller is about to immediately replace
        ``new_equation.structure`` with a freshly built term list (e.g. in
        ``EquationCrossover.apply``) -- deepcopying the parent's
        full ``[Term, Term, ...]`` only to overwrite it is wasted work
        (Term + Factor recursion accounts for the bulk of an Equation
        deepcopy).

        The returned shell aliases ``pool`` by reference (single
        population-wide instance), gets a fresh empty ``_eval_cache``
        (the cached evaluations don't survive a structural rewrite),
        and has ``structure = []`` so the caller can ``.append`` /
        assign without aliasing the parent's term list.
        """
        clone = _deepcopy_slots(
            self, memo={},
            attrs_to_avoid_copy=(
                'structure',
                '_cached_sw_weights', '_cached_vc_score',
                '_terms_labels_cache', '_terms_labels_without_power_cache',
                '_gram_super',
            ),
            attrs_to_share_by_ref=('pool',),
        )
        clone.structure = []
        clone._eval_cache = {}
        return clone

    def copy_properties_to(self, new_equation):
        new_equation.weights_internal_evald = self.weights_internal_evald
        new_equation.weights_final_evald = self.weights_final_evald
        new_equation.right_part_selected = self.right_part_selected
        new_equation.fitness_calculated = self.fitness_calculated
        new_equation.stability_calculated = self.stability_calculated
        new_equation.aic_calculated = self.aic_calculated
        new_equation.simplified = self.simplified
        new_equation.is_correct_right_part = self.is_correct_right_part
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
        self._history += add

    def add_random_term(self, forbidden_sigs=frozenset()) -> bool:
        """Try to append one fresh, non-duplicate term to ``self.structure``.

        Returns ``True`` if a term was appended, ``False`` if either the
        ``terms_number`` cap was already reached or the token pool could
        not produce a non-duplicate within ``max_iter`` retries. Callers
        that invoke this in a loop (e.g. ``EquationMutation.apply``,
        ``Equation.__init__``) MUST stop on the first ``False`` -- once
        the pool stops yielding uniques, further calls will not yield any
        either, and continuing past the failure pushes downstream
        operators (``_break_equation_duplication``, ``EqRightPartSelector``)
        into states that violate the duplicate-term invariant.

        ``forbidden_sigs`` is an optional frozenset of ``factors_labels`` the
        new term may not take (on top of the ban on duplicating an existing
        term). ``EquationMutation`` passes the signatures it just dropped so a
        mutation never re-adds a term it removed in the same call.
        """
        cap = int(self.metaparameters['terms_number']['value'])
        if len(self.structure) >= cap:
            return False
        # Cap diverges from the 100-attempt convention shared by
        # ``Equation.__init__.unique_term`` and
        # ``simplify_equation.replace_term``: this method is invoked in
        # an outer loop (``EquationMutation.apply``) that retries by
        # drawing more terms anyway, so a tight fast-fail saves cycles
        # when the pool is exhausted.
        max_iter = 10
        new_term = Term(self.pool, max_factors_in_term=self.metaparameters['max_factors_in_term']['value'],
                        mandatory_family=None, passed_term=None)
        success, _ = retry_until_unique(
            predicate=lambda: (new_term.factors_labels not in self.terms_labels
                               and new_term.factors_labels not in forbidden_sigs),
            mutate=lambda: new_term.randomize(),
            max_iter=max_iter,
            stats_name='add_random_term',
        )
        if success:
            # ``new_term`` is locally scoped and its factors were minted
            # fresh by ``pool.create()`` (no Factor-instance caching), so
            # appending the instance directly is safe -- nothing else
            # aliases its structure. Saves a per-call Term deepcopy that
            # was ~1.6ms × 29k calls per lv_new rep.
            self.structure.append(new_term)
            self._invalidate_label_cache()
            return True
        return False

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
    def coefficients_stability(self):
        return self._coefficients_stability

    @coefficients_stability.setter
    def coefficients_stability(self, val):
        self._coefficients_stability = val

    @property
    def aic(self):
        return self._aic

    @aic.setter
    def aic(self, val):
        self._aic = val

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
        # self.weights_internal_evald = True
        # self.weights_final_evald = False

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
        # self.weights_final_evald = True

    @property
    def text_form(self):
        try:
            form = ''
            if self.weights_final_evald:
                for term_idx in range(len(self.structure)):
                    if term_idx != self.target_idx:
                        form += str(self.weights_final[term_idx]) if term_idx < self.target_idx else str(self.weights_final[term_idx-1])
                        form += ' * ' + self.structure[term_idx].name + ' + '
                form += str(self.weights_internal[-1]) + ' = ' + \
                    self.structure[self.target_idx].name
            else:
                for term_idx in range(len(self.structure)):
                    form += 'k_' + str(term_idx) + ' ' + \
                        self.structure[term_idx].name + ' + '
                form += 'k_' + str(len(self.structure)) + ' = 0'
        except (AttributeError, IndexError, TypeError):
            form = ''
        return form

    @property
    def latex_form(self):
        form = self.structure[self.target_idx].latex_form + r' = '
        digits_rounding_max = 3
        for idx, term in enumerate(self.structure):
            idx_corrected = idx if idx <= self.target_idx else idx - 1
            if idx == self.target_idx or self.weights_final[idx_corrected] == 0:
                continue

            mnt, exp = exp_form(self.weights_final[idx_corrected], digits_rounding_max)
            exp_str = r'\cdot 10^{{{0}}} '.format(str(exp)) if exp != 0 else ''
            form += str(mnt) + exp_str + term.latex_form + r' + '

        mnt, exp = exp_form(self.weights_internal[-1], digits_rounding_max)
        exp_str = r'\cdot 10^{{{0}}} '.format(str(exp)) if exp != 0 else ''

        form += str(mnt) + exp_str
        return form

    @property
    def state(self):
        return self.text_form

    @property
    def terms_labels_without_power(self) -> frozenset:
        """Frozenset of per-term factor-label sets, with the power parameter dropped.

        Skips terms whose internal weight is exactly zero (target term always
        contributes). Per-term labels are delegated to
        ``Term.factors_labels_without_power`` so structural identity rules
        (e.g. trig freq bucketization via ``Factor.structural_label``) stay
        consistent across every dedup site. Memoized in
        ``_terms_labels_without_power_cache``; the 15 call sites of
        :meth:`_invalidate_label_cache` cover every Equation-driven structure
        mutation. Term-level mutations from external operators that bypass
        the Equation must invalidate the cache themselves.
        """
        cached = getattr(self, '_terms_labels_without_power_cache', None)
        if cached is not None:
            return cached
        described = set()
        for term_idx, term in enumerate(self.structure):
            if term_idx != self.target_idx:
                weight_idx = term_idx if term_idx < self.target_idx else term_idx - 1
                if np.isclose(self.weights_internal[weight_idx], 0):
                    continue
            term_labels = term.factors_labels_without_power
            if len(term_labels) > 0:
                described.add(term_labels)
        result = frozenset(described)
        self._terms_labels_without_power_cache = result
        return result

    @property
    def terms_labels(self) -> frozenset:
        """Frozenset of per-term factor-label sets identifying this equation's structure.

        Each inner element is the ``Term.factors_labels`` of one term -- so
        per-term identity rules (e.g. trig freq bucketization via
        ``Factor.structural_label``) are applied uniformly. Used as a
        hashable structural fingerprint for membership tests against
        ``objective.history``. Memoized in ``_terms_labels_cache``; see
        ``terms_labels_without_power`` for invalidation contract.
        """
        cached = getattr(self, '_terms_labels_cache', None)
        if cached is not None:
            return cached
        result = frozenset(term.factors_labels for term in self.structure)
        self._terms_labels_cache = result
        return result

    @property
    def active_terms_labels(self) -> frozenset:
        """Frozenset of per-term factor-label sets for the equation's ACTIVE
        structure: the target term plus every non-target term with a nonzero
        internal weight. Unlike ``terms_labels`` (full structure, zero-weight
        padding included) this is the structural fingerprint of the fitted
        law itself, so two equations of a system encode the same law
        rearranged iff their values are equal. Factor powers are kept
        (``u^2`` and ``u`` stay distinct), unlike ``terms_labels_without_power``.

        Not memoized: the value depends on ``weights_internal``, whose setter
        does not invalidate the label caches. Falls back to the full-structure
        ``terms_labels`` when the weights have not been evaluated yet.
        """
        if not self.weights_internal_evald:
            return self.terms_labels
        described = set()
        for term_idx, term in enumerate(self.structure):
            if term_idx != self.target_idx:
                weight_idx = term_idx if term_idx < self.target_idx else term_idx - 1
                if np.isclose(self.weights_internal[weight_idx], 0):
                    continue
            term_labels = term.factors_labels
            if len(term_labels) > 0:
                described.add(term_labels)
        return frozenset(described)

    @property
    def factors_labels(self) -> frozenset:
        """Alias of ``terms_labels`` — naming mirror used by some operators."""
        return self.terms_labels

    @property
    def factors_labels_without_power(self) -> frozenset:
        """Alias of ``terms_labels_without_power``."""
        return self.terms_labels_without_power

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

    def __iter__(self):
        return EquationIterator(self)        

class EquationIterator(object):
    def __init__(self, equation: Equation):
        self._internal_idx = 0
        self._equation = equation

    def __next__(self) -> Tuple[Union[None, float], Term]:
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
    raise NotImplementedError('solver_formed_grid function is to be depricated')
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


class SoEq(moeadd.MOEADDSolution):
    def __init__(self, pool: TFPool, metaparameters: dict) -> None:
        '''
        Top-level solution gene: a system of one Equation per variable.

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

        self.obj_funs = None

        self.metaparameters = metaparameters
        self.tokens_for_eq = TFPool(pool.families_demand_equation)
        self.tokens_supp = TFPool(pool.families_equationless)
        self.moeadd_set = False

        self.vars_to_describe = [token_family.variable for token_family in self.tokens_for_eq.families]

    def manual_reconst(self, attribute:str, value, except_attrs:dict):
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
        if use_pic:
            # self.use_pic_multiobjective_function()
            self.use_new_multiobjective_function()
        else:
            self.use_legacy_multiobjective_function()

    def use_legacy_multiobjective_function(self):
        from epde.eq_mo_objectives import equation_fitness, equation_complexity_by_factors
        # Both functions return per-equation tuples when called without an
        # equation_key, so the overall obj_fun layout matches the NEW path
        # (one weight per objective TYPE, expanded across equations by
        # MOEA/D). See penalty_based_intersection for the expansion logic.
        self.set_objective_functions([equation_fitness, equation_complexity_by_factors])

    def use_pic_multiobjective_function(self):
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

    def use_new_multiobjective_function(self):
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
            [equation_fitness] + [equation_terms_stability])

    def use_default_singleobjective_function(self):
        # globals.single_objective_metric picks which (already-computed)
        # attribute drives selection: 'discrepancy' -> equation_fitness,
        # 'instability' -> equation_terms_stability.
        import epde.globals as global_var
        from epde.eq_mo_objectives import (generate_partial, equation_fitness,
                                           equation_terms_stability)
        metric = getattr(global_var, 'single_objective_metric', 'discrepancy')
        objective = equation_terms_stability if metric == 'instability' else equation_fitness
        quality_objectives = [generate_partial(objective, eq_key) for eq_key in self.vars_to_describe]
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
        adj_complexity = copy.copy(complexity)
        for idx, compl in enumerate(adj_complexity):
            if compl is None:
                adj_complexity[idx] = self.obj_fun[-len(complexity) + idx]

        return list(self.obj_fun[-len(adj_complexity):]) == adj_complexity

    def create(self, passed_equations: list = None):
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
        for equation in population:
            if equation.terms_labels_without_power in unexplained_vars:
                equation.penalize_fitness(coeff=0.)
        population = population_sort(population)
        population = population[:population_size]
        gc.collect()
        population = evol_operator.apply(population, unexplained_vars)
        return population

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
        form = r"\begin{eqnarray*} "
        for idx, equation in enumerate(self.vals):
            postfix = '' if idx == len(self.vals) - 1 else r", \\ "
            form += equation.latex_form + postfix
        form += r" \end{eqnarray*}"
        return form

    def __hash__(self):
        return hash(self.vals.hash_descr)

    def __deepcopy__(self, memo=None):
        # SoEq has no own __slots__; the helper iterates the inherited
        # (likely empty) ABC slots harmlessly. Then carry the __dict__ over.
        new_struct = _deepcopy_slots(self, memo)
        for k, v in self.__dict__.items():
            setattr(new_struct, k, copy.deepcopy(v, memo))
        return new_struct

    def reset_state(self, reset_right_part: bool = True) -> None:
        """Forward reset_state to every Equation in this system."""
        for equation in self.vals:
            equation.reset_state(reset_right_part)

    def copy_properties_to(self, objective):
        for eq_label in self.vals.equation_keys:  # Not the best code possible here
            self.vals[eq_label].copy_properties_to(objective.vals[eq_label])

    def solver_params(self, full_domain: bool, grids: list = None) -> Tuple:
        '''
        Return solver form, grid and boundary conditions for every equation.

        Pass ``full_domain=True`` to read from the initial-data cache (the
        complete sampled domain) instead of the active grid cache. ``grids``
        overrides the implicit grid used to evaluate solver forms.
        '''
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

    @property
    def equations_labels_without_power(self) -> Tuple[frozenset, ...]:
        """Tuple of ``Equation.terms_labels_without_power`` for each equation.

        Order matches ``self.vars_to_describe``. Useful for structural identity
        checks on the system as a whole (e.g., dedup against history).
        """
        equations_caches = []
        for equation in self.vals:
            equations_caches.append(equation.terms_labels_without_power)
        return tuple(equations_caches)

    @property
    def equations_labels(self) -> Tuple[frozenset, ...]:
        """Tuple of ``Equation.terms_labels`` for each equation in the system.

        Element order matches ``self.vars_to_describe``. The hashable per-equation
        frozensets enable ``system in objective.history`` membership checks.
        """
        equations_caches = []
        for equation in self.vals:
            equations_caches.append(equation.terms_labels)
        return tuple(equations_caches)

    @property
    def terms_labels_without_power(self):
        # TODO(deprecate): use equations_labels_without_power
        return self.equations_labels_without_power

    @property
    def terms_labels(self):
        # TODO(deprecate): use equations_labels
        return self.equations_labels


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
