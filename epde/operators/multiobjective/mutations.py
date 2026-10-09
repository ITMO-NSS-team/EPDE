#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiobjective mutation operators (MOEA/D-D pipeline).

Ownership contract: the sole caller, ``OffspringUpdater.apply``, pops
the offspring from ``ParetoLevels.unplaced_candidates`` -- that pop is
the only live reference to the SoEq. The whole mutation hierarchy
(``SystemMutation`` -> ``EquationMutation`` -> ``TermMutation``)
therefore mutates in place and returns the same object; no defensive
deepcopies are made on the hot path.

Created on Wed Jun  2 15:46:31 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
from functools import partial
from typing import Union

from epde.optimizers.moeadd.moeadd import ParetoLevels

from epde.structure.main_structures import Equation, SoEq, Term, is_complexity_key
from epde.structure.structure_template import check_uniqueness
from epde.supplementary import filter_powers
from epde.operators.utils.template import CompoundOperator, add_base_param_to_operator

from epde import _loop_stats


from epde.decorators import HistoryExtender

# Bounded regenerate-retry budget for TermMutation: how many times a
# freshly randomized term that duplicates an existing one is re-rolled
# before falling back to the drop / floor-revert dedup policy.
MAX_REGEN_RETRIES = 3


def _budget_level(value, lo, hi) -> int:
    """The integer level a budget value stands for, inside ``[lo, hi]``.

    Nearest integer, half up, clipped into the bounds. A non-finite or
    non-positive value means "no budget" to the sparsity operator, i.e. every
    term may stay, so it reads as ``hi`` -- the loosest level -- never ``lo``.
    """
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return hi
    return int(min(max(np.floor(value + 0.5), lo), hi))


def step_complexity_budget(value, bounds) -> int:
    """One move of an integer complexity budget: ``c + {-1, 0, +1}``.

    The ``('complexity', var)`` gene does not go through
    :class:`MetaparameterMutation`. That operator is ``x + N(0, std*|x|)``
    reflected at 0 -- a multiplicative walk with ``E[log|1 + Z|] < 0`` at the
    default ``std=1``, which drifts a budget DOWN, and once the value fell
    below 1 the proportional noise would keep it there: ``c = 1`` absorbing.
    Here the three steps are equally likely (``np.random``) and a step out of
    ``bounds`` reflects about the half-integer edge (``lo - 1 -> lo``,
    ``hi + 1 -> hi``). That keeps the transition matrix symmetric -- no drift,
    and the uniform distribution the population is seeded from is also the
    walk's stationary one.
    """
    lo, hi = int(bounds[0]), int(bounds[1])
    proposed = _budget_level(value, lo, hi) + int(np.random.randint(-1, 2))
    if proposed < lo:
        proposed = 2 * lo - 1 - proposed
    elif proposed > hi:
        proposed = 2 * hi + 1 - proposed
    return proposed


class SystemMutation(CompoundOperator):
    key = 'SystemMutation'
    @_loop_stats.timed('SystemMutation.apply')
    def apply(self, objective : SoEq, arguments : dict): # TODO: add setter for best_individuals & worst individuals
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        # The only caller is ``OffspringUpdater.apply``, which pops the
        # offspring from ``unplaced_candidates`` (no other refs) and feeds
        # it here via ``chromosome_mutation``. Mutating in place is
        # observationally equivalent to deepcopying first -- the caller's
        # alias is the only handle on this SoEq, and the suboperators
        # (EquationMutation, MetaparameterMutation) were already updated
        # to mutate in place. Saves a full SoEq deepcopy per offspring
        # iteration (heaviest single deepcopy in the multi-objective hot
        # path).
        altered_objective = objective

        eqs_keys = altered_objective.vals.equation_keys; params_keys = altered_objective.vals.params_keys
        # eq_key = np.random.choice(eqs_keys)
        # altered_eq = self.suboperators['equation_mutation'].apply(altered_objective.vals[eq_key],
        #                                                           subop_args['equation_mutation'])
        affected_by_mutation = True
        for eq_key in eqs_keys:
            if len(eqs_keys) > 1:
                affected_by_mutation = np.random.random() < self.params['indiv_mutation_prob']

            if affected_by_mutation:
                altered_eq = self.suboperators['equation_mutation'].apply(altered_objective.vals[eq_key],
                                                                          subop_args['equation_mutation'])

                altered_objective.vals.replace_gene(gene_key = eq_key, value = altered_eq)

        for param_key in params_keys:
            if is_complexity_key(param_key):
                # An integer budget, stepped by its own rule; it only exists
                # under a budget realization, so a default run never gets here
                # and its MetaparameterMutation draws are untouched.
                self._step_complexity(altered_objective, param_key)
                continue
            altered_param = self.suboperators['param_mutation'].apply(altered_objective.vals[param_key],
                                                                      subop_args['param_mutation'])
            altered_objective.vals.replace_gene(gene_key = param_key, value = altered_param)
            altered_objective.vals.pass_parametric_gene(key = param_key, value = altered_param)

        if any(is_complexity_key(param_key) for param_key in params_keys):
            # Budget runs only: a default chromosome has no such gene, so its
            # metaparameters dicts stay exactly the objects they were.
            self._reconcile_metaparameters(altered_objective)
        return altered_objective

    @staticmethod
    def _reconcile_metaparameters(system: SoEq):
        """Give the system ONE private metaparameters dict, rebuilt from its
        genes, shared by the system and every equation.

        That is the layout ``SoEq.create`` builds, and crossover breaks it.
        ``EquationCrossover``'s ``clone_shell`` deep-copies the metaparameters
        into a fresh dict per offspring equation, and ``ChromosomeCrossover``'s
        gene swap moves an equation into the other offspring still holding its
        old system's dict -- which then belongs to two systems at once. The
        genes and the equations' own dicts stay right (the loop above writes
        every gene into its equation's dict), but ``system.metaparameters``
        does not: ``SoEq.text_form`` -- and the ``_pareto_history`` snapshots
        built from it -- printed a stale budget, and ``SoEq.create`` (the
        offspring updater's fallback) re-seeded the genes from it AND handed
        the stale dict, still aliased to the other system's equation, to every
        fresh equation, so this system's next mutation would rewrite that
        other system's budget behind its back.

        Rebuilt here, after the loop has written every gene, so the private
        dict's values ARE the genes and the equations read exactly the values
        they just received. Entries the genes do not cover (the term and
        factor caps) come from the system's own dict; for a passed system
        whose translated equations carried their own caps, the system's caps
        -- the ones ``SoEq.create`` would give them -- win.
        """
        private = deepcopy(system.metaparameters)
        for param_key in system.vals.params_keys:
            private[param_key]['value'] = system.vals[param_key]
        system.metaparameters = private
        for eq_key in system.vals.equation_keys:
            system.vals[eq_key].metaparameters = private

    @staticmethod
    def _step_complexity(system: SoEq, param_key):
        """Step one equation's complexity budget; re-open its fit if it moved.

        The budget decides which subset the sparsity operator returns, so an
        equation whose budget changed carries a fit made for a budget it no
        longer has -- the staleness a changed term set leaves, and reset the
        way ``EquationMutation`` resets that: RPS runs only while some
        equation's ``right_part_selected`` is down (the strategy's
        ``rps_cond``), so without the reset an unchanged structure would keep
        its old subset under the new gene. "Moved" is judged against the value
        in the EQUATION's own dict, the one its current fit was read from,
        rather than the gene, which a gene swap can leave out of step with it.
        """
        equation = system.vals[param_key[1]]
        entry = system.metaparameters[param_key]          # carries the bounds
        lo, hi = int(entry['bounds'][0]), int(entry['bounds'][1])
        old = system.vals[param_key]
        # Typed like the gene: Gene.value's setter keeps the old value without
        # a word when handed a value of another type.
        new = type(old)(step_complexity_budget(old, (lo, hi)))
        fitted_under = equation.metaparameters.get(param_key, {}).get('value')
        # pass_parametric_gene asserts the key on the equation; one that came
        # without it (none should) gets the system's entry, no budget so far.
        equation.metaparameters.setdefault(param_key, dict(entry))
        system.vals.replace_gene(gene_key=param_key, value=new)
        system.vals.pass_parametric_gene(key=param_key, value=new)
        if fitted_under is None or _budget_level(fitted_under, lo, hi) != new:
            equation.reset_state(reset_right_part=True)

    def use_default_tags(self):
        self._tags = {'mutation', 'chromosome level', 'contains suboperators'}


class EquationMutation(CompoundOperator):
    key = 'EquationMutation'
    @_loop_stats.timed('EquationMutation.apply')
    @HistoryExtender(f'\n -> mutating equation', 'ba')
    def apply(self, objective : Equation, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        # Per the module ownership contract, the enclosing SoEq is the
        # caller's exclusive object (OffspringUpdater pops the only live
        # reference; SystemMutation mutates it in place), so mutating the
        # equation in place is safe and saves a per-call deepcopy
        # (was ~5ms per call, ~5000 calls per lv_new rep).
        equation = objective

        # Snapshot the term-set fingerprint before mutation; reset_state runs
        # below only if the SET changes, so an untouched equation keeps its
        # right part and skips the term-sweep. Set granularity is sound (the
        # fit depends only on the term set); the one same-set/different-order
        # case (drop then re-add the same signature) is blocked by dropped_sigs.
        structure_before = equation.terms_labels

        # Per-term Bernoulli term-replace via the ``mutation`` sub-operator
        # (TermMutation), governed by ``r_mutation``. Without it, mature
        # chromosomes at the max_terms_number cap spin on add_random_term no-ops
        # and structural exploration collapses to crossover alone. Skip
        # ``n_immutable`` head terms so the right-part anchor and any
        # mandatory_family terms survive across mutations.
        r_mutation = self.params['r_mutation']
        replace_attempts = 0
        mutable_count = max(1, len(equation.structure) - equation.n_immutable)
        # Reverse order so a full-drop dedup inside TermMutation (which can
        # remove ``term`` and shrink ``structure``) only shifts indices we
        # have already processed -- forward iteration would skip terms or
        # run off the end of the now-shorter structure.
        for term_idx in reversed(range(equation.n_immutable, len(equation.structure))):
            if np.random.uniform(0, 1) <= r_mutation:
                replace_attempts += 1
                self.suboperators['mutation'].apply(
                    objective=(term_idx, equation),
                    arguments=subop_args['mutation'],
                )
        _loop_stats.record('EquationMutation.replace_terms',
                           replace_attempts, mutable_count)

        # Signatures dropped by the replace step. The add step below must not
        # re-add them: that is the only way to restore the original set while
        # permuting structure order (desyncing the position-indexed weights),
        # so forbidding it keeps the set-based change detection complete.
        dropped_sigs = structure_before - equation.terms_labels

        # Probabilistic term-add: each of the ``n_added_terms`` slots fires a
        # Bernoulli trial at ``term_addition_prob``, so the genome no longer
        # grows unconditionally on every mutation call.
        # The ``max_terms_number`` metaparameter (chromosome-wide ceiling,
        # enforced inside ``add_random_term``) still caps growth; cap-hit
        # or pool exhaustion breaks the loop.
        n_added = int(self.params['n_added_terms'])
        term_addition_prob = self.params['term_addition_prob']
        add_attempts = 0
        for _ in range(n_added):
            if np.random.random() >= term_addition_prob:
                continue
            add_attempts += 1
            if not equation.add_random_term(forbidden_sigs=dropped_sigs):
                break
        _loop_stats.record('EquationMutation.add_terms', add_attempts, n_added)

        assert len(equation.terms_labels) == len(equation.structure)

        # Reset RPS state iff the term set actually changed; a no-op mutation
        # leaves the equation's right-part selection intact (not re-swept).
        if equation.terms_labels != structure_before:
            equation.reset_state(reset_right_part=True)

        return equation

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'contains suboperators'}


class MetaparameterMutation(CompoundOperator):
    key = 'MetaparameterMutation'

    def apply(self, objective : Union[int, float], arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        # Canonical Gaussian perturbation using the declared ``mean`` /
        # ``std`` params; ``std`` scales relative to the current value so
        # the perturbation stays proportionate across metaparameter
        # magnitudes. Negative results are reflected at 0. With the JSON
        # defaults (mean=0.0, std=1.0) this matches the previous
        # ``normal(objective, objective)`` distribution.
        altered_objective = objective + np.random.normal(self.params['mean'],
                                                         self.params['std'] * np.abs(objective))
        if altered_objective < 0:
            altered_objective = - altered_objective

        return np.float64(altered_objective)

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'no suboperators'}

    
class TermMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term is replaced with a randomly created new one.
    """
    key = 'TermMutation'
    
    def apply(self, objective : tuple, arguments : dict): #term_idx, equation):
        """
        Randomize the term at ``term_idx`` IN PLACE, keeping it unique among
        the equation's terms.

        Parameters:
        -----------
        objective : tuple
            ``(term_idx, equation)`` -- index of the mutated term and the
            equation that owns it.

        Returns:
        ----------
        term : Term object
            The same Term instance, randomized in place. If every regenerate
            retry produced a duplicate, the term is instead REMOVED from the
            equation (above the 2-term floor) or reverted to its unique
            pre-mutation structure (at the floor) -- see the dedup policy
            below.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        term_idx, equation = objective
        term = equation.structure[term_idx]
        # ``randomize()`` REPLACES ``term.structure`` with a freshly
        # built list of fresh ``Factor`` instances (see
        # ``Term.randomize`` at main_structures.py:203). The old list
        # survives as ``original_structure`` here -- no deepcopy needed,
        # the alias is intentional. Same trick as the ``add_random_term``
        # optimization. Saves a ~10ms Term deepcopy on every replace_terms
        # iter (was the largest remaining mutation-path deepcopy cost).
        original_structure = term.structure
        # Bounded regenerate-retry: re-roll a duplicate-producing
        # randomize() up to MAX_REGEN_RETRIES times before invoking the
        # dedup policy below. ``original_structure`` keeps aliasing the
        # pre-mutation factor list throughout -- every randomize() call
        # builds a fresh list, so the floor-revert stays valid.
        attempts = 0
        for _ in range(MAX_REGEN_RETRIES):
            attempts += 1
            term.randomize()
            term.resetSavedState()
            equation._invalidate_label_cache()
            signatures = {t.factors_labels for t in equation.structure}
            duplicate = len(signatures) != len(equation.structure)
            if not duplicate:
                break
        _loop_stats.record('TermMutation.regen_attempts', attempts, MAX_REGEN_RETRIES)

        # Dedup policy after the retry budget: if the mutated term still
        # duplicates another term in the equation, remove it outright.
        # Only at the 2-term floor (nothing safe to drop) do we restore
        # the unique pre-mutation term. This keeps duplicates out of the
        # population (otherwise caught a generation later at the RPS
        # entry / crossover assert).
        if duplicate and len(equation.structure) > 2:
            # Identity-tracked target auto-updates when ``term`` is removed; no
            # manual target_idx decrement needed. If ``term`` was the target,
            # ``equation.target`` degrades to None and the subsequent
            # reset_state(reset_right_part=True) in EquationMutation.apply
            # re-selects via RPS.
            equation.structure = [t for t in equation.structure if t is not term]
            equation._invalidate_label_cache()
            _loop_stats.record('TermMutation.unique_term.DROP', 1, 1)
        elif duplicate:
            # Floor: restore the (unique) pre-mutation term in place.
            term.structure = original_structure
            term.resetSavedState()
            equation._invalidate_label_cache()
            _loop_stats.record('TermMutation.unique_term.FLOOR_REVERT', 1, 1)
        else:
            _loop_stats.record('TermMutation.unique_term', 1, 1)
        return term

    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploration', 'no suboperators'}


class TermParameterMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term parameters are changed with a random increment.
    """
    key = 'TermParameterMutation'    
    
    def apply(self, objective : tuple, arguments : dict): # term_idx, objective
        """ 
        Specific operator of the term mutation, where the term parameters are changed with a random increment.
        
        Parameters:
        -----------
        term_idx : integer
            The index of the mutating term in the equation.
            
        equation : Equation object
            The equation object, in which the term is present.
        
        Returns:
        ----------
        new_term : Term object
            The new, created from the previous one with random parameters increment, term.
            
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        unmutable_params = {'dim', 'power'}
        term_idx, equation = objective
        # NB: no target is set here. The ``term_idx == equation.target_idx``
        # guard below skips param-mutating the right-part term when one is
        # selected; when the target is None (unselected) the comparison is
        # simply False and every term is eligible -- RPS selects the right
        # part, not this operator.

        # Cap the retry loop so a constrained token pool can't deadlock
        # the optimizer (same hazard fixed in ``enforce_rps_uniqueness``).
        max_iter = 100
        attempts = 0
        for _ in range(max_iter):
            attempts += 1
            term = equation.structure[term_idx]
            for factor in term.structure:
                if term_idx == equation.target_idx:
                    continue
                parameter_selection = deepcopy(factor.params)
                for param_idx, param_properties in factor.params_description.items():
                    if np.random.random() < self.params['r_param_mutation'] and param_properties['name'] not in unmutable_params:
                        interval = param_properties['bounds']
                        if interval[0] == interval[1]:
                            shift = 0
                            continue
                        if isinstance(interval[0], int):
                            shift = np.rint(np.random.normal(loc=0, scale=self.params['multiplier']*(interval[1] - interval[0]))).astype(int)
                        elif isinstance(interval[0], float):
                            shift = np.random.normal(loc=0, scale=self.params['multiplier']*(interval[1] - interval[0]))
                        else:
                            raise ValueError('In current version of framework only integer and real values for parameters are supported')
                        if self.params['strict_restrictions']:
                            parameter_selection[param_idx] = np.min((np.max((parameter_selection[param_idx] + shift, interval[0])), interval[1]))
                        else:
                            parameter_selection[param_idx] = parameter_selection[param_idx] + shift
                    factor.params = parameter_selection
            term.structure = filter_powers(term.structure)
            equation._invalidate_label_cache()
            signatures = {t.factors_labels for t in equation.structure}
            if len(signatures) == len(equation.structure):
                break
        _loop_stats.record('TermParameterMutation.unique', attempts, max_iter)
        term.resetSavedState()
        # A factor-parameter change is a STRUCTURAL change: it rewrites
        # ``factors_labels``, so the fitted features are no longer the ones
        # the weights describe. Invalidating only the label caches left both
        # ``*_evald`` flags up over a support decision for different columns.
        # Dormant today -- the mutation builders wire only ``TermMutation`` --
        # but silently wrong the moment it is re-enabled.
        equation.reset_for_structure_change()
        return term
    
    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploitation', 'no suboperators'}


def get_basic_mutation(mutation_params):
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = mutation_params)

    term_mutation = TermMutation([])

    equation_mutation = EquationMutation(['r_mutation', 'n_added_terms', 'term_addition_prob'])
    add_kwarg_to_operator(operator = equation_mutation)
    
    metaparameter_mutation = MetaparameterMutation(['std', 'mean'])
    add_kwarg_to_operator(operator = metaparameter_mutation)

    chromosome_mutation = SystemMutation(['indiv_mutation_prob'])
    add_kwarg_to_operator(operator = chromosome_mutation)

    equation_mutation.set_suboperators(operators = {'mutation' : term_mutation})#, [term_param_mutation, ]
                                       # probas = {'equation_crossover' : [0.0, 1.0]})

    chromosome_mutation.set_suboperators(operators = {'equation_mutation' : equation_mutation, 
                                                      'param_mutation' : metaparameter_mutation})
    return chromosome_mutation
