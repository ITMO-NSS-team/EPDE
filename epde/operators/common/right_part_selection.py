#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:50:55 2021

@author: mike_ubuntu
"""
import time

import numpy as np
from copy import deepcopy
import warnings

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.decorators import HistoryExtender
from epde.structure.main_structures import Term, Equation
from epde.supplementary import GramSetup, retry_until_unique
from epde import _loop_stats

class EqRightPartSelector(CompoundOperator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the CompoundOperator class
        key - str, value - instance of a class, inhereted from the CompoundOperator. 
        Suboperators, performing tasks of equation processing. In this case, only one suboperator is present: 
        fitness_calculation, dedicated to calculation of fitness function value.

    Methods:
    -----------
    apply(equation)
        return None
        Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``

    
    '''
    key = 'FitnessCheckingRightPartSelector'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Negative cache: structure hashes that produced ``inf`` fitness on
        # every eligible target_idx. Scope is one operator-tree lifetime,
        # which matches one ``EpdeSearch.fit()`` call -- the strategy
        # builder in ``epde/optimizers/moeadd/strategy.py`` constructs a
        # fresh ``EqRightPartSelector()`` per build. Repeat encounters
        # skip the term-sweep via internal ``objective.randomize()``.
        self._bad_structures: set = set()

    @staticmethod
    def _precompute_super_gram(objective: Equation) -> None:
        """Build a per-equation super-Gram over all structure terms and
        attach it to ``objective._gram_super`` for the upcoming term-sweep.

        Each candidate target_idx in the sweep then derives its
        ``GramSetup`` view via :meth:`GramSetup.from_full` (pure slicing,
        no recompute). On any failure (e.g. non-finite term evaluations,
        non-grid-shaped weights) clear the slot and let downstream
        ``VWSRSparsity.apply`` fall back to its legacy per-target path.
        """
        try:
            if global_var.grid_cache is None:
                objective._gram_super = None
                return
            sample_weights = global_var.grid_cache.g_func[
                global_var.grid_cache.g_func_mask]
            grid_shape = global_var.grid_cache.inner_shape
            feat_list = [term.evaluate(False, grids=None)
                         for term in objective.structure]
            Z = np.vstack(feat_list).T
            if not np.all(np.isfinite(Z)):
                objective._gram_super = None
                _loop_stats.record('EqRPS.gram_super_skip', 1, 1)
                return
            objective._gram_super = GramSetup.precompute_super(
                Z, sample_weights, grid_shape)
            _loop_stats.record('EqRPS.gram_super_built', 1, 1)
        except Exception:
            # Defensive: any unexpected failure (shape mismatch, missing
            # cache) means we silently fall back -- numerics are
            # preserved, only the speedup is lost.
            objective._gram_super = None
            _loop_stats.record('EqRPS.gram_super_skip', 1, 1)

    @HistoryExtender('\n -> The equation structure was detected: ', 'a')
    def apply(self, objective : Equation, arguments : dict):
        """Select a right-part term for ``objective`` in-place.

        Handles two recoverable failure modes inside the outer loop via
        ``objective.randomize()`` (cheap, single-equation reroll) rather
        than bubbling up to the chromosome-level offspring loop -- the
        chromosome regen path is ~100x more expensive than a single
        equation reroll and floods the EA when many candidates fail.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        # Duplicate-term detection: a frozenset of per-term factor signatures
        # has the same length as ``structure`` iff every term is distinct.
        # Comparing against ``terms_labels`` here would be dimensionally wrong
        # (see the same family of bugs fixed in ``enforce_rps_uniqueness`` and
        # ``simplify_equation``).
        signatures = {term.factors_labels for term in objective.structure}
        assert len(signatures) == len(objective.structure), \
            'Equation has duplicate terms; randomize before right-part selection.'

        outer_max_iter = 50
        inner_max_iter = 100
        outer_attempts = 0
        while not (objective.simplified and objective.is_correct_right_part):
            outer_attempts += 1
            if outer_attempts > outer_max_iter:
                warnings.warn(
                    'EqRightPartSelector.apply: outer loop did not converge '
                    f'after {outer_max_iter} iterations; accepting current state.'
                )
                break
            objective.reset_state(True)
            min_fitness = np.inf
            weights_internal = np.zeros(len(objective.structure) - 1)
            min_idx = 0
            inner_attempts = 0
            # ``restore_property(deriv=True)`` injects a derivative-family
            # token into the structure; it's a refinement op, not a regen
            # signal. The randomize() fallback below would only fire if the
            # 200-iter restore_property loop failed 100 times in a row -- a
            # ~20 000-attempt impossibility in practice.
            while not any(term.contains_deriv(objective.main_var_to_explain) for term in objective.structure):
                inner_attempts += 1
                if inner_attempts > inner_max_iter:
                    warnings.warn(
                        'EqRightPartSelector.apply: restore_property failed to '
                        f'introduce a deriv of {objective.main_var_to_explain!r} '
                        f'after {inner_max_iter} attempts; randomizing equation.'
                    )
                    objective.randomize()
                    break
                objective.restore_property(mandatory_family=False, deriv=True)
            _loop_stats.record('EqRPS.inner_derivative', inner_attempts, inner_max_iter)

            # Negative cache: this structure already zeroed out for every
            # target_idx on a prior call (sparsity is deterministic on
            # the same input). Skip the term-sweep, reroll, retry.
            if objective.terms_labels in self._bad_structures:
                _loop_stats.record('EqRPS.bad_structure_skip', 1, 1)
                objective.randomize()
                continue

            # Tier 3: precompute the super-Gram over all terms ONCE per
            # outer iter so the term-sweep below derives per-target
            # GramSetup views via pure slicing instead of rebuilding the
            # windowed XTWX matmul for every candidate.
            self._precompute_super_gram(objective)

            for target_idx, target_term in enumerate(objective.structure):
                if not objective.structure[target_idx].contains_deriv(objective.main_var_to_explain):
                    continue
                objective.target_idx = target_idx
                fitness = self.suboperators['fitness_calculation'].apply(objective, arguments = subop_args['fitness_calculation'], force_out_of_place = True)
                if fitness is not None and fitness < min_fitness:
                    min_fitness = fitness
                    min_idx = target_idx
                    weights_internal = objective.weights_internal
                    weights_final = objective.weights_final
                    sw_weights = objective._cached_sw_weights

                objective.weights_internal_evald = False
                objective.weights_final_evald = False

            if np.isinf(min_fitness):
                # Every eligible target produced inf fitness for the
                # post-restore structure -- reroll this single equation
                # locally (cheap) and continue the outer loop. The
                # negative cache below remembers the structure shape so
                # future outer iters / future calls skip the term-sweep.
                _loop_stats.record('EqRPS.inf_fitness_regen', 1, 1)
                self._bad_structures.add(objective.terms_labels)
                objective.randomize()
                continue

            objective.weights_internal = weights_internal
            objective.weights_final = weights_final
            objective._cached_sw_weights = sw_weights
            objective.weights_internal_evald = True
            objective.weights_final_evald = True
            objective.target_idx = min_idx

            if not self.simplify_equation(objective):
                objective.simplified = True
            if objective.structure[objective.target_idx].contains_deriv(objective.main_var_to_explain):
                objective.is_correct_right_part = True

        _loop_stats.record('EqRPS.outer', outer_attempts, outer_max_iter)
        # Drop the super-Gram so a downstream consumer (e.g. fitness
        # recomputation outside the term-sweep) falls back to the
        # per-target ``GramSetup.__init__`` path; the cached super-Gram
        # is only valid for the structure observed during the sweep.
        objective._gram_super = None
        objective.right_part_selected = True
        objective.remove_zero_terms()

    def simplify_equation(self, objective: Equation):
        # Get nonzero terms
        nonzero_terms_mask = np.array([False if weight == 0 else True for weight in objective.weights_internal], dtype=np.int32)
        nonrs_terms = [term for i, term in enumerate(objective.structure) if i != objective.target_idx]
        nonzero_terms = [item for item, keep in zip(nonrs_terms, nonzero_terms_mask) if keep]
        nonzero_terms.append(objective.structure[objective.target_idx])
        equation_terms = [term.factors_labels_without_power for term in nonzero_terms]

        if len(equation_terms) <= 1:
            return False
        common_factors = list(frozenset.intersection(*equation_terms))
        if not common_factors:
            return False

        for common_factor in common_factors:
            # Min power across the matching factor in every nonzero term.
            min_order = np.inf
            for term in nonzero_terms:
                for factor in term.structure:
                    if factor.structural_label_without_power == common_factor:
                        if factor.cache_label[1][0] < min_order:
                            min_order = factor.cache_label[1][0]

            # Reduce order of common factor in every term; drop zero-power factors.
            max_iter = 100
            for term in nonzero_terms:
                factors_simplified = []
                for factor in term.structure:
                    if factor.structural_label_without_power == common_factor:
                        for i, value in enumerate(factor.params_description):
                            if factor.params_description[i]["name"] == "power":
                                factor.set_param(factor.params[i] - min_order, idx=i)
                                if factor.params[i] == 0:
                                    factors_simplified.append(factor)
                            else:
                                continue
                term.structure = [factor for factor in term.structure if factor not in factors_simplified]
                term.reset_saved_state()

                # If term's order became zero -- replace term.
                # Cap retries so a constrained token pool can't
                # deadlock the optimizer (same hazard fixed in
                # ``enforce_rps_uniqueness``).
                def _replacement_acceptable():
                    empty = len(term.structure) == 0
                    not_meaningful = not term.contains_meaningful()
                    signatures = {t.factors_labels for t in objective.structure}
                    duplicate = len(signatures) != len(objective.structure)
                    return not (empty or not_meaningful or duplicate)

                # Cap-hit policy is silently-accept-whatever-state: the
                # term may still be empty/non-meaningful/duplicate when
                # the cap fires, and the outer RPS loop is expected to
                # detect that on its next pass.
                retry_until_unique(
                    predicate=_replacement_acceptable,
                    mutate=lambda: term.randomize(),
                    max_iter=max_iter,
                    stats_name='simplify_equation.replace_term',
                )

            # Structure changed: invalidate stale fitness /
            # weights / AIC caches while leaving RPS to the
            # caller's outer loop.
            try:
                objective.reset_state(reset_right_part=False)
            except TypeError:
                objective.reset_state()
            return True
        return False

    def use_default_tags(self):
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}

        
class RandomRHPSelector(CompoundOperator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the CompoundOperator class
        key - str, value - instance of a class, inhereted from the CompoundOperator. 
        Suboperators, performing tasks of equation processing. In this case, only one suboperator is present: 
        fitness_calculation, dedicated to calculation of fitness function value.

    Methods:
    -----------
    apply(equation)
        return None
        Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``

    
    '''
    key = 'RandomRightPartSelector'

    @HistoryExtender('\n -> The equation structure was detected: ', 'a')
    def apply(self, objective : Equation, arguments : dict):
        # print(f'CALLING RIGHT PART SELECTOR FOR {objective.text_form}')
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        if not objective.right_part_selected:
            term_selection = [term_idx for term_idx, term in enumerate(objective.structure)
                              if term.contains_deriv(variable = objective.main_var_to_explain)]

            if len(term_selection) == 0:
                idx = np.random.choice([term_idx for term_idx, _ in enumerate(objective.structure)])
                prev_term = objective.structure[idx]
                # Bounded retry + dedup check: never spin against a finite
                # token pool, never introduce a duplicate term (see
                # feedback-structure-dedup memory).
                max_iter = 100
                candidate_term = None
                attempts = 0
                for _ in range(max_iter):
                    attempts += 1
                    candidate_term = Term(pool = prev_term.pool, mandatory_family = objective.main_var_to_explain,
                                          max_factors_in_term = len(prev_term.structure),
                                          create_derivs = True)
                    if not candidate_term.contains_deriv(variable = objective.main_var_to_explain):
                        continue
                    sig = candidate_term.factors_labels
                    if any(j != idx and t.factors_labels == sig
                           for j, t in enumerate(objective.structure)):
                        continue
                    break
                else:
                    warnings.warn(
                        f'RandomRHPSelector: could not produce a unique deriv term '
                        f'for {objective.main_var_to_explain!r} after {max_iter} '
                        f'attempts; keeping last candidate (may duplicate).'
                    )
                _loop_stats.record('RandomRHPSelector.candidate_gen', attempts, max_iter)

                objective.structure[idx] = candidate_term
            else:
                idx = np.random.choice(term_selection)

            objective.target_idx = idx
            # print('Selected right part term', objective.structure[idx].name)
            objective.reset_explaining_term(idx)
            objective.right_part_selected = True


    def use_default_tags(self):
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}


def _scrub_conflicting_terms(equation: Equation, fixed_rps, *, max_iter: int = 2000,
                              skip_idx=None) -> bool:
    """Replace any term in ``equation.structure`` whose factor signature is a
    superset of one of the ``fixed_rps`` signatures (each a ``frozenset`` of
    factor labels).

    Term-similarity semantics here are **superset**, unlike
    ``detect_similar_terms`` (exact match) and ``simplify_equation``'s
    duplicate check (set-cardinality on ``factors_labels``). A term
    "conflicts" with a fixed RPS when its factor set contains every
    factor of the RPS plus optional extras -- this is the
    cross-equation interference pattern SoEqRPS needs to break, and it
    is intentionally stricter than exact equality. Future maintainers
    changing one of the three predicates should NOT propagate it here
    without re-deriving the bidirectional-RPS proof.

    When ``skip_idx`` is passed, the term at that index is left alone
    -- used by the bidirectional pass below to preserve an equation's
    own already-selected RPS.

    Returns True if at least one term was randomized; the equation's
    cached fitness/weight state is reset on the way out.
    """
    if not fixed_rps:
        return False

    def _conflicts(t):
        return any(rs.issubset(t.factors_labels) for rs in fixed_rps)

    changed = False
    for idx, term in enumerate(equation.structure):
        if idx == skip_idx:
            continue
        if not _conflicts(term):
            continue
        attempts = 0
        for _ in range(max_iter):
            attempts += 1
            term.randomize()
            term.reset_saved_state()
            signatures = {t.factors_labels for t in equation.structure}
            duplicate = len(signatures) != len(equation.structure)
            if not _conflicts(term) and not duplicate:
                break
        _loop_stats.record('scrub_conflicting_terms', attempts, max_iter)
        changed = True

    if changed:
        try:
            equation.reset_state(reset_right_part=False)
        except TypeError:
            equation.reset_state()
    return changed


class SoEqRightPartSelector(CompoundOperator):
    """Chromosome-level RPS that enforces bidirectional cross-equation
    uniqueness.

    Forward sequential pass (pre-scrub each equation against
    already-selected RPS, then run the per-equation sweep) handles the
    case where equation_k > equation_j re-uses equation_j's RPS as a
    non-target term. A second bidirectional convergence pass closes the
    other direction: equation_j's structure is also scrubbed of any term
    whose factor set is a superset of equation_k's (k > j) RPS. Without
    the second pass the FIRST equation in ``vars_to_describe`` could keep
    a later equation's target as a non-RPS term (e.g. LV's eq for u
    keeping ``dv/dx0``), since at the time it was processed the later
    RPS was not yet known.

    The bidirectional pass is bounded by ``max_bidirectional_passes`` and
    exits as soon as a full sweep produces no scrubbing changes
    (fixed-point). Each pass also re-runs the per-equation selector when
    its structure changed, since the prior target_idx may no longer be
    optimal under the new structure.
    """
    key = 'SoEqRightPartSelector'

    def apply(self, objective, arguments: dict):
        """Run per-equation RPS forward + bidirectional passes in-place.

        Failures inside ``EqRightPartSelector.apply`` are handled locally
        via per-equation ``objective.randomize()`` -- this method has no
        regen signal to forward.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)
        eq_selector = self.suboperators['eq_right_part_selector']
        eq_args = subop_args.get('eq_right_part_selector', arguments)

        equations = list(objective)
        rps_signatures = [None] * len(equations)

        # Forward sequential pass: pre-scrub each equation against
        # already-fixed RPS signatures, then run the per-equation selector.
        for eq_idx, equation in enumerate(equations):
            other_rps = [rs for rs in rps_signatures[:eq_idx] if rs is not None]
            if other_rps:
                _scrub_conflicting_terms(equation, other_rps)
            eq_selector.apply(objective=equation, arguments=eq_args)
            try:
                rps_signatures[eq_idx] = equation.structure[
                    equation.target_idx].factors_labels
            except (AttributeError, IndexError, TypeError):
                rps_signatures[eq_idx] = None

        # Bidirectional convergence: each equation now knows the others'
        # RPS, so re-scrub against the full set (skipping own target) and
        # re-select when scrubbing changes the structure. Iterates until
        # a full pass yields no changes.
        max_passes = 50
        passes_used = 0
        for _ in range(max_passes):
            passes_used += 1
            any_changes = False
            for eq_idx, equation in enumerate(equations):
                other_rps = [rs for i, rs in enumerate(rps_signatures)
                             if i != eq_idx and rs is not None]
                if not other_rps:
                    continue
                target_idx = getattr(equation, 'target_idx', None)
                changed = _scrub_conflicting_terms(
                    equation, other_rps, skip_idx=target_idx,
                )
                if not changed:
                    continue
                # Scrubbing mutated non-target terms: force re-selection so
                # the post-scrub structure is evaluated for the best RPS.
                equation.right_part_selected = False
                equation.simplified = False
                equation.is_correct_right_part = False
                eq_selector.apply(objective=equation, arguments=eq_args)
                try:
                    rps_signatures[eq_idx] = equation.structure[
                        equation.target_idx].factors_labels
                except (AttributeError, IndexError, TypeError):
                    pass
                any_changes = True
            if not any_changes:
                break
        _loop_stats.record('SoEqRPS.bidirectional', passes_used, max_passes)

    def use_default_tags(self):
        self._tags = {'right part selection', 'chromosome level',
                      'contains suboperators', 'inplace'}
