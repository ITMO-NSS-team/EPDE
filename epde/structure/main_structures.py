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
from typing import Union, Callable, List, Dict, Tuple
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
    'max_terms_number':    {'optimizable': False, 'value': 5.},
    'max_factors_in_term': {'optimizable': False, 'value': 1.},
}


def _normalize_metaparameters(mp: dict) -> dict:
    """In-place: accept the legacy ``terms_number`` metaparameter key and
    rename it to ``max_terms_number`` with a one-time DeprecationWarning.

    Idempotent and safe when neither key is present (downstream reads then
    rely on the default dict). If BOTH keys are present, the new key is
    authoritative and the stale legacy key is dropped silently.
    """
    if 'terms_number' in mp:
        if 'max_terms_number' not in mp:
            warnings.warn(
                "The 'terms_number' metaparameter key is deprecated; use "
                "'max_terms_number'. The legacy key was normalized automatically.",
                DeprecationWarning, stacklevel=2)
            mp['max_terms_number'] = mp.pop('terms_number')
        else:
            mp.pop('terms_number')
    return mp


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
        self.resetSavedState()

    def manual_reconst(self, attribute:str, value, except_attrs:dict) -> None:
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
    def cache_label(self) -> tuple:
        if len(self.structure) > 1:
            structure_sorted = sorted(self.structure, key=lambda x: x.cache_label)
            cache_label = tuple([elem.cache_label for elem in structure_sorted])
        else:
            cache_label = self.structure[0].cache_label
        return cache_label

    def use_cache(self) -> None:
        self.cache_linked = True
        for idx, _ in enumerate(self.structure):
            if not self.structure[idx].cache_linked:
                self.structure[idx].use_cache()

    # TODO: non-urgent, make self.descr_variable_marker setting for defined parameter

    @singledispatchmethod
    def defined(self, passed_term) -> None:
        raise NotImplementedError(
            f'passed term should have string or list/dict types, not {type(passed_term)}')

    @defined.register
    def _(self, passed_term: list, collapse_powers = True) -> None:
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
    def _(self, passed_term: str, collapse_powers = True) -> None:
        self.structure = []
        if isinstance(passed_term, str):
            _, temp_f = self.pool.create(label=passed_term)
            self.structure.append(temp_f)
        elif isinstance(passed_term, Factor):
            self.structure.append(passed_term)
        else:
            raise ValueError('The structure of a term should be declared with str or factor.Factor obj, instead got', type(passed_term))

    def randomize(self, mandatory_family=None, forbidden_factors=None,
                  create_derivs=False, **kwargs) -> None:
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
    def descr_variable_marker(self) -> str:
        return self._descr_variable_marker

    @descr_variable_marker.setter
    def descr_variable_marker(self, marker: False) -> None:
        if not marker or isinstance(marker, str):
            self._descr_variable_marker = marker
        else:
            raise ValueError('Described variable marker shall be a family label (i.e. "u") of "False"')

    @_loop_stats.timed('Term.evaluate')
    def evaluate(self, grids: Union[List[np.ndarray], Dict[int, List[np.ndarray]]] = None) -> Dict[int, np.ndarray]:
        # assert global_var.samples_manager is not None, 'Currently working only with connected cache'
        if grids is not None:
            # FIX! grids are not used in the process...
            raise NotImplementedError("Trying to call Term.evaluate(grids != None), which is not yet implemented.")

        if self.saved[False] or (self.factors_labels, False) in global_var.samples_manager:
            values = global_var.samples_manager.get(self.factors_labels, normalized=False,
                                                    saved_as=self.saved_as[False])
            values = {key: value.reshape(-1) for key, value in values.items()}
            return values
        else:
            values = super().evaluate()
            # if normalize:
            # # As normalize is always false, the block seems to be depricated ()
            #     # value = (value - np.mean(value)) / np.std(value)
            #     # value = value / np.linalg.norm(value, 2)
            #     # value = minmax_normalize(value)
            #     value = 2 * (value - value.min()) / (value.max() - value.min()) - 1

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
                self.saved[False] = global_var.samples_manager.add(self.factors_labels, values, normalized=False)
                if self.saved[False]:
                    self.saved_as[False] = self.factors_labels
            values = {key: value.reshape(-1) for key, value in values.items()}
            return values

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
                self.resetSavedState()
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


# Registry of every Equation-level cache slot and its invalidation policy --
# the single source of truth consumed by ``reset_state``,
# ``_invalidate_label_cache``, ``__deepcopy__`` and ``clone_shell``. Adding a
# new cache means adding ONE entry here (plus the slot in
# ``Equation.__slots__``; TestStateRegistrySync asserts the two stay in
# sync). Policies:
#   'structure'  -- keyed on the current structure/target: wiped by BOTH
#                   ``reset_state`` and ``_invalidate_label_cache``.
#   'reset-only' -- wiped by ``reset_state`` but DELIBERATELY survives
#                   ``_invalidate_label_cache`` / ``remove_zero_terms``:
#                   PhysicsInformedLasso.fit recomputes the sparsity caches on
#                   the CONVERGED active mask, so they still align with the
#                   zero-pruned structure and Instability.compute reads them
#                   right after the prune (fitness.py). Pinned by
#                   TestSparsityCachePreservation.
_EQ_CACHE_FIELDS = (
    ('_eval_cache', 'structure'),                        # evaluate() memo, wiped to a FRESH {}
    ('_terms_labels_cache', 'structure'),
    ('_terms_labels_without_power_cache', 'structure'),
    ('_gram_super', 'structure'),                        # EqRPS tier-3 super-Gram, one sweep only
    ('_cached_sw_weights', 'reset-only'),                # axis-mode sliding-window weights
    ('_cached_vc_score', 'reset-only'),                  # vcoef per-term stability scores
    # (metric, value) memo for the survival/tile estimators. UNLIKE the two
    # sparsity caches above it is a pure Instability.compute memo of the OLD
    # structure (nothing recomputes it on the converged mask), so it must not
    # survive a structural mutation. No-op under the default vcoef metric,
    # which never populates it.
    ('_cached_alt_instability', 'structure'),
)
_EQ_STRUCTURE_CACHES = tuple(f for f, p in _EQ_CACHE_FIELDS if p == 'structure')
# Slots skipped by the copy paths. ``_eval_cache`` is exempt: it is traversed
# by ``_deepcopy_slots`` and then replaced with a fresh dict, so the copy owns
# an EMPTY-but-equal dict (pinned by
# test_eval_cache_after_deepcopy_is_fresh_dict).
_EQ_CACHE_AVOID_COPY = tuple(f for f, _ in _EQ_CACHE_FIELDS if f != '_eval_cache')


# Registry of every non-cache Equation slot a reset clears, grouped by the
# EVENT that invalidates it. The per-group ``reset_*`` primitives below are
# generated from this table, so adding a flag/value pair means adding ONE entry
# here (plus the slot in ``Equation.__slots__``; TestStateRegistrySync asserts
# the two stay in sync). Entries are ``(slot, cleared_value)`` and are written
# with a plain ``setattr`` -- every one of these slots is either private or
# backed by a pass-through property setter, so the raw write is equivalent.
_EQ_STATE_GROUPS = (
    # The identity-tracked right part. Cleared in lockstep with the weights:
    # both weight vectors are indexed relative to the target position.
    ('target', (
        ('_target_term', None),
    )),
    # RPS's durable verdict: "this equation has been through right-part
    # selection". Read by ``rps_cond`` in both strategy builders to decide
    # whether the selector runs at all.
    ('selection', (
        ('right_part_selected', False),
    )),
    # The SUPPORT decision -- which terms survive sparsity.
    ('weights_internal', (
        ('weights_internal_evald', False),
        ('_weights_internal', None),
    )),
    # The fitted MAGNITUDES -- the only vector a residual can be built from
    # (see ``Equation.residual``).
    ('weights_final', (
        ('weights_final_evald', False),
        ('_weights_final', None),
    )),
    # Every objective value and its calculated-flag, paired so a stale value
    # can never outlive its flag.
    ('scores', (
        ('fitness_calculated', False),
        ('_fitness_value', None),
        ('stability_calculated', False),
        ('_coefficients_stability', None),
        ('complexity_calculated', False),
        ('_complexity_value', None),
        ('aic_calculated', False),
        ('_aic', None),
    )),
)
_EQ_STATE_GROUP_SLOTS = {name: slots for name, slots in _EQ_STATE_GROUPS}


class Equation(ComplexStructure):
    __slots__ = ['_history', 'structure', 'interelement_operator', 'n_immutable', 'pool',
                  # '_target', '_features', 'saved', 'saved_as','max_factors_in_term', 'operator',
                 # ``simplified`` / ``is_correct_right_part`` used to live here.
                 # They are RPS-internal loop control -- written and read only
                 # by EqRightPartSelector.apply's outer ``while`` -- and are now
                 # locals of that method. Persisting them let an offspring
                 # inherit "already simplified", skip the loop body (and with it
                 # the ``reset_state`` on its first line), and leave RPS still
                 # carrying its parent's fit.
                 '_target_term', 'right_part_selected', '_weights_final', 'weights_final_evald',
                 '_weights_internal', 'weights_internal_evald', 'fitness_calculated', 'stability_calculated', 'aic_calculated',
                 'complexity_calculated',
                 '_fitness_value', '_coefficients_stability', '_aic', '_complexity_value', 'metaparameters', 'main_var_to_explain',
                 '_eval_cache', '_cached_sw_weights', '_cached_vc_score',
                 '_cached_alt_instability',
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
        self._target_term = None   # identity-tracked right-part target (see ``target`` property)
        self.reset_state()

        if metaparameters is None:
            metaparameters = copy.deepcopy(_DEFAULT_EQUATION_METAPARAMETERS)
        else:
            _normalize_metaparameters(metaparameters)   # legacy 'terms_number' -> 'max_terms_number'

        self.n_immutable = len(basic_structure)
        self.pool = pool
        self.structure = []
        self.metaparameters = metaparameters
        if (self.metaparameters['max_terms_number']['value'] < self.n_immutable):
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
        max_terms = int(self.metaparameters['max_terms_number']['value'])
        # Variable per-equation birth size: draw a target term count in
        # [low, max_terms] (inclusive). ``low`` pins the floor at 2 but never
        # below the immutable head terms and never above the configured max,
        # so the term count is a genuine per-equation property bounded by the
        # max rather than a value that births every equation at full size.
        low = min(max(2, self.n_immutable), max_terms)
        high = max_terms
        birth_n = np.random.randint(low, high + 1) if low <= high else max_terms
        for i in range(len(basic_structure), birth_n):
            new_term = Term(self.pool, max_factors_in_term=self.metaparameters['max_factors_in_term']['value'],
                            mandatory_family=None, passed_term=None)
            def _term_mutate():
                new_term.randomize()
                new_term.resetSavedState()
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
        self.resetSavedState()

    @property
    def target(self):
        """The right-part Term, identity-validated against the live structure.

        Returns the stored target Term iff it is still present in
        ``self.structure`` (by identity, ``is``); otherwise ``None``. This is
        the safety net for the whole right-part model: a target dropped by any
        structural mutation degrades to ``None`` instead of dangling as a stale
        index. Access the target through this property (a call) rather than
        ``self.structure[self.target_idx]``.
        """
        tgt = self._target_term
        if tgt is None:
            return None
        for term in self.structure:
            if term is tgt:
                return tgt
        return None

    @property
    def target_idx(self):
        """Position of the target Term in ``self.structure`` (int), or ``None``.

        Derived from the target Term's identity, so it auto-tracks term
        drops/reorders -- no manual reindexing on structure change. O(n);
        capture into a local before using it inside a loop/comprehension.
        """
        tgt = self._target_term
        if tgt is None:
            return None
        for i, term in enumerate(self.structure):
            if term is tgt:
                return i
        return None

    @target_idx.setter
    def target_idx(self, value):
        # Anchor the target to the Term currently at ``value`` so the position
        # survives later drops/reorders of OTHER terms. ``None`` clears it.
        if value is None:
            self._target_term = None
        else:
            self._target_term = self.structure[value]

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
            # Both weight vectors are length m+1: one coef per non-target term,
            # then the intercept (Equation._validate_weight_layout). The former
            # ``has_intercept = len(wi) == m + 1`` sniff existed only because the
            # sparsity operators used to emit a bare length-m ``estimator.coef_``.
            m = len(self.structure) - 1
            # Capture the PRE-drop target position once for the weight-index
            # map below. The target Term is never zero-weight-dropped (the
            # ``i == tgt`` skip keeps it), so its identity survives and the
            # derived ``target_idx`` recomputes correctly against the compacted
            # structure -- no manual reindex needed.
            tgt = self.target_idx
            zero_terms = []        # structure indices to drop
            zero_coef_pos = []     # matching coefficient indices to drop
            for i in range(len(self.structure)):
                if i == tgt:
                    continue
                idx = self.weight_index(i, tgt)
                if wi[idx] == 0:
                    zero_terms.append(i)
                    zero_coef_pos.append(idx)
            if zero_terms:
                self.structure = [term for term_idx, term in enumerate(self.structure) if term_idx not in zero_terms]
                # No ``self.target_idx -= ...`` -- the identity-tracked target
                # auto-tracks the surviving target Term.
                # Compact BOTH vectors in lockstep with the structure so later
                # position-indexed reads (active_terms_labels, the renderers,
                # re-prune) stay aligned, each keeping its own trailing
                # intercept slot. weights_final used to be skipped here -- "it
                # already holds only the non-zero entries" was true of the old
                # zero-filtered nnz+1 layout and is false under the unified one.
                zcp = set(zero_coef_pos)
                keep = [j for j in range(m) if j not in zcp]
                self._weights_internal = np.array([wi[j] for j in keep] + [wi[-1]])
                if self.weights_final_evald:
                    wf = self._weights_final
                    self._weights_final = np.array([wf[j] for j in keep] + [wf[-1]])
                # ``_invalidate_label_cache`` also wipes _eval_cache, which
                # is essential here: the right-part-selector's per-target
                # sweep populates the cache keyed on target_idx, and the
                # post-drop target_idx can collide with a swept value.
                # ``_cached_sw_weights`` / ``_cached_vc_score`` are preserved:
                # PhysicsInformedLasso.fit recomputes them on the CONVERGED
                # active mask (sparsity.py:449-471), so their columns cover
                # exactly the features surviving this prune -- they still
                # align with the compacted structure, and Instability.compute
                # reads them right after this call (fitness.py:141->155-157).
                # Pinned by TestSparsityCachePreservation.
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
        raise NotImplementedError("Depricated method!")
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
        # ``max_terms_number`` cap is already reached.
        terms_cap = int(self.metaparameters['max_terms_number']['value'])
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
        raise NotImplementedError("Tokens can no longer be set as right-part-unique.")
        new_eq = copy.deepcopy(self)
        self.copy_properties_to(new_eq)
        new_eq.target_idx = right_part_idx
        if any([factor.status['unique_for_right_part'] for factor in new_eq.structure[right_part_idx].structure]):
            for term_idx, term in enumerate(new_eq.structure):
                if term_idx != right_part_idx:
                    term.filter_tokens_by_right_part(
                        new_eq.structure[right_part_idx], self, term_idx)

        new_eq.resetSavedState()
        return new_eq

    def _feature_indexes(self, active_only: bool, tgt: int) -> List[int]:
        """Structure positions that become feature columns, in structure order.

        Every term but the target, or -- under ``active_only`` -- only those the
        sparsity step left with a non-zero ``weights_internal`` slot. This is
        what makes ``evaluate`` emit two widths from one structure.
        ``Equation.active_mask`` is the same predicate expressed over weight
        positions, so ``weights_final[:-1][active_mask]`` narrows a coefficient
        vector onto exactly the ``active_only`` column set.
        """
        if not active_only:
            return [idx for idx in range(len(self.structure)) if idx != tgt]
        return [idx for idx in range(len(self.structure))
                if idx != tgt and self.weights_internal[self.weight_index(idx, tgt)] != 0]

    @_loop_stats.timed('Equation.evaluate')
    def evaluate(self, *, active_only: bool = False) -> Tuple[Dict[int, np.ndarray],
                                                              Union[None, Dict[int, np.ndarray]]]:
        """Evaluate the equation into ``(target, features)``, per trajectory.

        ``target`` holds the LHS term's values; ``features`` is the design
        matrix of the remaining terms, one column each, or ``None`` when no term
        qualifies.

        ``active_only`` picks the COLUMN SET -- it applies no scaling of any
        kind. The flag was called ``normalize`` for years, from the days when
        ``Term`` carried a normalisation flag of its own; ``Term.evaluate`` no
        longer does (the block survives only commented out, above). ``False``
        (the default) makes a column of every non-target term; ``True`` keeps
        only those with a non-zero ``weights_internal`` slot -- the narrow width
        that ``objectives._extract_coefs_intercept`` masks ``weights_final``
        down to.

        Caching policy: only the default, wide result is memoized, keyed on
        ``target_idx``. The ``active_only`` branch reads ``weights_internal``
        and callers update the weights between successive calls, so caching it
        would hand back a stale column mask. The wide path is weight-independent
        and is where the hits are -- the sparsity step and the fitness fillers
        both take it within one fitness invocation.
        """
        tgt = self.target_idx          # identity-derived position, captured once
        cacheable = not active_only
        if cacheable and hasattr(self, '_eval_cache') and tgt in self._eval_cache:
            return self._eval_cache[tgt]

        targets = self.target.evaluate()

        feature_indexes = self._feature_indexes(active_only, tgt)
        if len(feature_indexes) > 0:
            feats_list = [self.structure[idx].evaluate() for idx in feature_indexes]

            samples_by_features: Dict[int, List[np.ndarray]] = dict()
            for feat_idx in range(len(feats_list)):
                for key, vals in feats_list[feat_idx].items():
                    if feat_idx == 0:
                        samples_by_features[key] = [vals,]
                    else:
                        samples_by_features[key].append(vals)

            features: Dict[int, np.ndarray] = dict()
            for key, feat_list in samples_by_features.items():
                features[key] = np.vstack(feat_list)

            del samples_by_features

            # features = np.vstack(feat_list)
            for key in features.keys():
                if features[key].ndim == 1:
                    features[key] = np.expand_dims(features[key], 1).T


            if any([feature.ndim == 1 for feature in features.values()]):
                assert all([feature.ndim == 1 for feature in features.values()])
                for key in features.keys():
                    features[key] = np.expand_dims(features[key], 1).T

            for key in features.keys():
                features[key] = np.transpose(features[key])

        else:
            features = None

        result = (targets, features)
        if cacheable:
            if not hasattr(self, '_eval_cache'):
                self._eval_cache = {}
            self._eval_cache[tgt] = result
        return result

    def residual(self, *, active_only: bool = False) -> Dict[int, np.ndarray]:
        """``target - (features @ coefs + intercept)``, one 1-D array per
        trajectory.

        Split out of ``evaluate``'s old ``return_val`` flag, which selected no
        evaluation option but a different operation -- and computed it wrongly.
        Four defects, every one of them invisible because the only caller
        (``epde.interface.logger.Logger.add_log``) is itself unreachable:

        * the intercept column was built as ``np.vstack([features, ones])``,
          appending a ROW to an ``(n_points, n_features)`` matrix, so after the
          transpose the weighted sum came out ``n_features`` long and the
          subtraction against an ``(n_points, 1)`` target could only broadcast
          when the two happened to be equal;
        * that ones-column was then never read -- the sum ran over the feature
          indexes alone, so the intercept was silently dropped;
        * ``targets[idx]`` leaked the ``idx`` of the feature-index loop, where it
          must be ``targets[key]``, the trajectory;
        * the weight vectors were indexed by full-STRUCTURE position although
          both skip the target, so every term past it read its neighbour's
          coefficient.

        The wide path also read ``weights_internal`` -- the sparsity step's
        SUPPORT decision -- in place of the fitted magnitudes. ``weights_final``
        is the only vector a residual can be built from, and serves both column
        sets here: under the unified layout (``_validate_weight_layout``) it
        carries one slot per non-target term plus the intercept, so the narrow
        case is an ``active_mask`` selection and nothing more.

        Not memoized, for the reason ``evaluate`` does not cache its narrow
        branch: the weights move between calls.
        """
        targets, features = self.evaluate(active_only=active_only)
        weights = self.weights_final
        intercept = weights[-1]
        if features is None:
            return {key: np.asarray(target, dtype=float).reshape(-1) - intercept
                    for key, target in targets.items()}
        coefs = np.asarray(weights[:-1], dtype=float)
        if active_only:
            coefs = coefs[self.active_mask]
        return {key: (np.asarray(target, dtype=float).reshape(-1) -
                      (np.asarray(features[key], dtype=float) @ coefs + intercept))
                for key, target in targets.items()}

    def _reset_groups(self, *groups: str) -> None:
        """Clear the named state groups (see ``_EQ_STATE_GROUPS``)."""
        for group in groups:
            for slot, cleared in _EQ_STATE_GROUP_SLOTS[group]:
                setattr(self, slot, cleared)

    def assert_state_invariants(self, where: str) -> None:
        """Check the fitted-state contract; no-op unless ``EPDE_LOOP_STATS=1``.

        Structure mutation is a raw slot write -- ``structure`` has no property
        setter, and ``_validate_weight_layout`` fires only when a weight vector
        is ASSIGNED -- so nothing stops an operator from reshaping the term list
        under weights that still claim to describe it. These are the invariants
        that the flag would otherwise be lying about, checked where the contract
        is consumed (RPS exit, fitness entry) rather than everywhere.

        Gated rather than always-on because the length checks run on the hottest
        object in the search; ``EPDE_LOOP_STATS`` is the switch the A/B harness
        already sets.
        """
        if not _loop_stats.enabled():
            return
        n = len(self.structure)
        if self.weights_final_evald and not self.weights_internal_evald:
            raise AssertionError(
                f'{where}: fitted magnitudes without a support decision '
                f'({self.main_var_to_explain!r}). weights_final may never '
                'outlive weights_internal.')
        for name in ('weights_internal', 'weights_final'):
            if not getattr(self, f'{name}_evald'):
                continue
            vector = getattr(self, f'_{name}')
            if vector is None:
                raise AssertionError(
                    f'{where}: {name}_evald is up with no vector behind it '
                    f'({self.main_var_to_explain!r}).')
            if len(vector) != n:
                raise AssertionError(
                    f'{where}: {name} has {len(vector)} slots for {n} terms '
                    f'({self.main_var_to_explain!r}) -- the structure changed '
                    'under a live fit.')
            if self.target is None:
                raise AssertionError(
                    f'{where}: {name}_evald is up with no installed target '
                    f'({self.main_var_to_explain!r}); the weights are indexed '
                    'relative to one.')
        if self.right_part_selected and self.target is None:
            raise AssertionError(
                f'{where}: right_part_selected with no target '
                f'({self.main_var_to_explain!r}).')

    def reset_for_structure_change(self) -> None:
        """The equation's term set changed: everything derived from it dies.

        Drops RPS's ``right_part_selected`` verdict, both weight vectors with
        their flags, every score, and every registered cache -- but KEEPS the
        installed target, so a caller mid-selection does not lose the right part
        it is working on. ``reset_state(reset_right_part=False)`` is this.

        This replaces the old soft reset, which kept ``weights_internal_evald``
        up and its data intact while dropping only the ``weights_final`` flag.
        That asymmetry is retired: every one of its call sites followed a real
        structural change, so the retained support decision was stale at all of
        them, and on the one path the outer RPS loop did not re-enter it flowed
        into ``remove_zero_terms`` and pruned on the wrong coefficients.
        """
        self._reset_groups('selection', 'weights_internal', 'weights_final', 'scores')
        # Wipe EVERY registered cache -- both policies (see _EQ_CACHE_FIELDS
        # for the per-field docs).
        for field, _policy in _EQ_CACHE_FIELDS:
            setattr(self, field, {} if field == '_eval_cache' else None)

    def reset_state(self, reset_right_part: bool = True) -> None:
        """Drop all cached evaluation/fitness state on this Equation.

        Call after any structural mutation (or to discard a stale fitness/AIC
        evaluation). ``reset_right_part=False`` keeps the installed target;
        everything else goes either way -- see
        :meth:`reset_for_structure_change`.
        """
        if reset_right_part:
            self._reset_groups('target')
        self.reset_for_structure_change()

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
        # Only the 'structure'-policy caches (see _EQ_CACHE_FIELDS): the
        # sparsity caches deliberately survive this call.
        for field in _EQ_STRUCTURE_CACHES:
            if field == '_eval_cache':
                if hasattr(self, '_eval_cache'):
                    self._eval_cache = {}
            else:
                setattr(self, field, None)


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
            attrs_to_avoid_copy=_EQ_CACHE_AVOID_COPY,
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
                '_target_term',   # shell has no structure; target must be None until the caller rebuilds
            ) + _EQ_CACHE_AVOID_COPY,
            attrs_to_share_by_ref=('pool',),
        )
        clone.structure = []
        clone._target_term = None
        clone._eval_cache = {}
        return clone

    def copy_properties_to(self, new_equation):
        new_equation.weights_internal_evald = self.weights_internal_evald
        new_equation.weights_final_evald = self.weights_final_evald
        new_equation.right_part_selected = self.right_part_selected
        new_equation.fitness_calculated = self.fitness_calculated
        new_equation.stability_calculated = self.stability_calculated
        new_equation.complexity_calculated = getattr(self, 'complexity_calculated', False)
        new_equation.aic_calculated = self.aic_calculated

        try:
            new_equation._fitness_value = self._fitness_value
        except AttributeError:
            pass

        try:
            new_equation._coefficients_stability = self._coefficients_stability
        except AttributeError:
            pass

        try:
            new_equation._complexity_value = self._complexity_value
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
        ``max_terms_number`` cap was already reached or the token pool could
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
        cap = int(self.metaparameters['max_terms_number']['value'])
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
    def term_number(self) -> int:
        """Actual number of terms currently in this equation's structure
        (a per-equation property). Bounded ABOVE by the
        ``max_terms_number`` metaparameter, but after the variable-birth
        draw it is generally less than that configured max. Read-only."""
        return len(self.structure)

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

    def _validate_weight_layout(self, weights, name: str):
        """Enforce THE coefficient-vector contract on every assignment.

        Both ``weights_internal`` and ``weights_final`` are length
        ``len(structure)`` == ``m + 1`` where ``m = len(structure) - 1`` is the
        number of NON-TARGET terms::

            index i in 0..m-1  ->  structure term at ``weight_index``-th position
            index -1 (== m)    ->  the fitted intercept / free coefficient

        Zeros are RETAINED in both vectors -- a term is inactive iff its
        ``weights_internal`` slot is 0, the intercept is absent iff
        ``weights_internal[-1]`` is 0. The two vectors differ only in VALUES
        (support-selection coefficients vs final physical magnitudes).

        This used to be producer-dependent: the sparsity operators emitted
        ``estimator.coef_`` (length m, NO intercept slot) alongside a
        zero-filtered ``weights_final`` (length nnz+1), while the translator
        emitted the full m+1 layout for both -- so every consumer had to sniff
        the length (``len(wi) == m + 1``) or silently depend on
        ``remove_zero_terms`` having already collapsed nnz onto m. Validating
        here is what makes the contract real instead of documented.

        ``None`` passes (``reset_state`` nulls both), and so does an assignment
        made before ``structure`` exists -- ``__init__`` calls ``reset_state``
        before building the structure list.
        """
        if weights is None:
            return
        structure = getattr(self, 'structure', None)
        if structure is None:
            return
        expected = len(structure)
        actual = len(weights)
        if actual != expected:
            raise ValueError(
                f'{name} must hold one coefficient per non-target term plus a '
                f'trailing intercept slot: expected length {expected} for a '
                f'{expected}-term structure, got {actual}. See '
                'Equation._validate_weight_layout for the contract.')

    def weight_index(self, term_idx: int, tgt: int = None) -> int:
        """Position of ``structure[term_idx]``'s coefficient inside
        ``weights_internal`` / ``weights_final``.

        The target term is skipped by both vectors, so every index past it
        shifts down by one. This shift was open-coded at a dozen call sites
        (renderers, the complexity cores, the legacy refit, the solver forms);
        routing them all through here keeps a single definition. Raises for the
        target itself -- it has no coefficient (its weight is implicitly -1).

        ``tgt`` lets a loop hoist the target lookup: ``target_idx`` is an O(n)
        identity scan, so calling this per term without it makes the caller
        O(n^2) (the ``target_idx`` docstring asks callers to capture it).
        """
        if tgt is None:
            tgt = self.target_idx
        if term_idx == tgt:
            raise ValueError(
                f'Term {term_idx} is the target; it carries no weight slot.')
        return term_idx if term_idx < tgt else term_idx - 1

    @property
    def active_mask(self):
        """Boolean mask over the NON-TARGET terms: which ones the sparsity step
        kept. Length ``len(structure) - 1``, aligned by ``weight_index``.

        ``weights_internal`` is the vector that DECIDES support (the sparsity
        output); ``weights_final`` merely carries the refit magnitudes at the
        same positions.
        """
        return np.asarray(self.weights_internal[:-1]) != 0

    @property
    def intercept(self):
        """The fitted intercept (free coefficient) -- the trailing slot of
        ``weights_final``. Exactly ``0.0`` when the sparsity step regularized
        it away; see ``Equation._validate_weight_layout``."""
        return self.weights_final[-1]

    @property
    def weights_internal(self):
        if self.weights_internal_evald:
            return self._weights_internal
        else:
            raise AttributeError(
                'Internal weights called before initialization')

    @weights_internal.setter
    def weights_internal(self, weights):
        self._validate_weight_layout(weights, 'weights_internal')
        self._weights_internal = weights
        # self.weights_internal_evald = True
        # self.weights_final_evald = False

    @property
    def weights_final(self):
        if self.weights_final_evald:
            return self._weights_final
        else:
            raise AttributeError(
                f'Final weights called before initialization on {self.text_form}')

    @weights_final.setter
    def weights_final(self, weights):
        self._validate_weight_layout(weights, 'weights_final')
        self._weights_final = weights
        # self.weights_final_evald = True

    @property
    def text_form(self):
        try:
            form = ''
            if self.weights_final_evald:
                tgt = self.target_idx
                for term_idx in range(len(self.structure)):
                    if term_idx != tgt:
                        form += str(self.weights_final[self.weight_index(term_idx, tgt)])
                        form += ' * ' + self.structure[term_idx].name + ' + '
                # The fitted intercept is the TRAILING entry of weights_final,
                # guaranteed by Equation._validate_weight_layout. Reading
                # ``weights_internal[-1]`` (as this used to) printed the last
                # TERM's coefficient in the constant position, back when the
                # sparsity producers emitted a bare ``estimator.coef_``.
                form += str(self.weights_final[-1]) + ' = ' + \
                    self.target.name
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
        if self.target is None or not self.weights_final_evald:
            return ''
        tgt = self.target_idx
        form = self.target.latex_form + r' = '
        digits_rounding_max = 3
        for idx, term in enumerate(self.structure):
            if idx == tgt:
                continue
            # Skipping zero-weight terms is correct WITHOUT a preceding
            # remove_zero_terms now: weights_final keeps its zeros and stays
            # aligned to the full structure (see _validate_weight_layout).
            idx_corrected = self.weight_index(idx, tgt)
            if self.weights_final[idx_corrected] == 0:
                continue

            mnt, exp = exp_form(self.weights_final[idx_corrected], digits_rounding_max)
            exp_str = r'\cdot 10^{{{0}}} '.format(str(exp)) if exp != 0 else ''
            form += str(mnt) + exp_str + term.latex_form + r' + '

        # Trailing weights_final entry = the fitted intercept (see text_form).
        mnt, exp = exp_form(self.weights_final[-1], digits_rounding_max)
        exp_str = r'\cdot 10^{{{0}}} '.format(str(exp)) if exp != 0 else ''

        form += str(mnt) + exp_str
        return form

    @property
    def state(self):
        return self.text_form

    def _active_term_label_set(self, *, drop_power: bool) -> frozenset:
        """Per-term factor-label signatures over the ACTIVE structure: the
        target term plus every non-target term whose internal weight is
        non-zero. ``drop_power`` selects each term's
        ``factors_labels_without_power`` (True) or ``factors_labels`` (False).
        The zero-weight skip applies only when a target is selected; empty
        per-term label sets are dropped. Shared by ``terms_labels_without_power``
        and ``active_terms_labels``.
        """
        tgt = self.target_idx
        described = set()
        for term_idx, term in enumerate(self.structure):
            if tgt is not None and term_idx != tgt:
                if np.isclose(self.weights_internal[self.weight_index(term_idx, tgt)], 0):
                    continue
            term_labels = (term.factors_labels_without_power if drop_power
                           else term.factors_labels)
            if term_labels:
                described.add(term_labels)
        return frozenset(described)

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
        result = self._active_term_label_set(drop_power=True)
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
        return self._active_term_label_set(drop_power=False)

    def __iter__(self):
        return EquationIterator(self)        

class EquationIterator(object):
    def __init__(self, equation: Equation):
        self._internal_idx = 0
        self._equation = equation

    def __next__(self) -> Tuple[Union[None, float], Term]:
        if self._internal_idx < len(self._equation.structure):
            if self._equation.weights_final_evald:
                tgt = self._equation.target_idx
                while True:
                    # Target first, THEN the weight slot: weight_index
                    # rejects the target (it has no coefficient; its weight
                    # is implicitly -1). Zero entries are skipped rather
                    # than absent -- weights_final retains them.
                    if self._internal_idx == tgt:
                        coeff = -1.
                        break
                    idx_in_weights = self._equation.weight_index(self._internal_idx, tgt)
                    if self._equation.weights_final[idx_in_weights] == 0:
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

def check_metaparameters(metaparameters: dict):
    metaparam_labels = ['max_terms_number', 'max_factors_in_term', 'sparsity']  # noqa: F841
    _normalize_metaparameters(metaparameters)
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
                # attrs_from_dict bypasses Equation.__init__, so a pre-rename
                # serialized chromosome can carry the legacy 'terms_number'
                # metaparameter key; normalize it here.
                if hasattr(eq, 'metaparameters'):
                    _normalize_metaparameters(eq.metaparameters)
                equations[self.vars_to_describe[idx]] = eq
            if hasattr(self, 'metaparameters'):
                _normalize_metaparameters(self.metaparameters)
            self.vals = Chromosome(equations, {key: val for key, val in self.metaparameters.items()
                                               if val['optimizable']})

    def use_default_multiobjective_function(self, second_objective: str = None):
        # Lockstep site #2 of the selectable second axis (the others: the
        # strategy's filler assembly and the MOEA/D ideal point). ``None``
        # defers to the ``second_objective`` global, which resolves to
        # 'instability' -- what the removed ``use_pic=True`` default meant.
        from epde.interface.search_config import active_config
        if second_objective is None:
            second_objective = active_config().objectives.second_objective
        if second_objective == 'instability':
            # self.use_pic_multiobjective_function()
            self.use_new_multiobjective_function()
        else:
            self.use_legacy_multiobjective_function()

    def use_legacy_multiobjective_function(self):
        from epde.eq_mo_objectives import equation_fitness, equation_complexity
        # Both functions return per-equation tuples when called without an
        # equation_key, so the overall obj_fun layout matches the NEW path
        # (one weight per objective TYPE, expanded across equations by
        # MOEA/D). See penalty_based_intersection for the expansion logic.
        # ``equation_complexity`` is the dispatching family reader: it serves
        # the Complexity filler's stored value on the live path and falls
        # back to the lazy cores ('factors' default = bit-compatible with the
        # old equation_complexity_by_factors wiring) for translated systems.
        self.set_objective_functions([equation_fitness, equation_complexity])

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
        from epde.eq_mo_objectives import equation_fitness, equation_terms_stability
        # Both objectives return per-equation tuples when called without an
        # equation_key (flattened by ``obj_fun``), so no per-variable
        # generate_partial expansion is needed here.
        self.set_objective_functions([equation_fitness] + [equation_terms_stability])

    def use_default_singleobjective_function(self):
        # globals.single_objective_metric picks which (already-computed)
        # attribute drives selection: 'discrepancy' -> equation_fitness,
        # 'instability' -> equation_terms_stability.
        from epde.interface.search_config import active_config
        from epde.eq_mo_objectives import (generate_partial, equation_fitness,
                                           equation_terms_stability)
        metric = active_config().objectives.single_objective_metric
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
                    form += r' \ ' + equation.text_form + '\n'
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
        # Identity-flavored: ``Chromosome.hash_descr`` returns BOUND METHODS
        # for Equation genes (ComplexStructure.hash_descr is a plain method,
        # not a property), so equal-by-``__eq__`` systems generally hash
        # differently -- a latent hash/eq contract violation. Left alone on
        # purpose: no live set/dict of SoEq exists (``objective.history``
        # stores equations_labels tuples), and changing the hash would
        # perturb any future hash-based iteration order.
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

    def reset_for_structure_change(self) -> None:
        """Forward :meth:`Equation.reset_for_structure_change` to every gene."""
        for equation in self.vals:
            equation.reset_for_structure_change()

    def copy_properties_to(self, objective):
        for eq_label in self.vals.equation_keys:  # Not the best code possible here
            self.vals[eq_label].copy_properties_to(objective.vals[eq_label])

    def __iter__(self):
        return SoEqIterator(self)

    @property
    def fitness_calculated(self):
        return all([equation.fitness_calculated for equation in self.vals])

    def _equation_label_tuple(self, *, drop_power: bool) -> Tuple[frozenset, ...]:
        """Per-equation label frozensets in ``self.vars_to_describe`` order.
        ``drop_power`` selects each equation's ``terms_labels_without_power``
        (True) vs ``terms_labels`` (False)."""
        return tuple(eq.terms_labels_without_power if drop_power else eq.terms_labels
                     for eq in self.vals)

    @property
    def equations_labels_without_power(self) -> Tuple[frozenset, ...]:
        """Tuple of ``Equation.terms_labels_without_power`` for each equation.

        Order matches ``self.vars_to_describe``. Useful for structural identity
        checks on the system as a whole (e.g., dedup against history).
        """
        return self._equation_label_tuple(drop_power=True)

    @property
    def equations_labels(self) -> Tuple[frozenset, ...]:
        """Tuple of ``Equation.terms_labels`` for each equation in the system.

        Element order matches ``self.vars_to_describe``. The hashable per-equation
        frozensets enable ``system in objective.history`` membership checks.
        """
        return self._equation_label_tuple(drop_power=False)


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
