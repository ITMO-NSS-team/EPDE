#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:50:55 2021

@author: mike_ubuntu
"""
import itertools
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import List, Dict, Tuple, Any

import numpy as np
from copy import deepcopy
import warnings

from functools import partial
from collections import defaultdict

import epde.globals as global_var
from epde.interface.search_config import (ASSIGNMENT_MAX_SUBSETS, active_config,
                                          assignment_subset_count, rps_param)

from epde.operators.utils.template import CompoundOperator, dictApplyUFunc, dictAdd
from epde.decorators import HistoryExtender
from epde.structure.main_structures import Term, Equation
from epde.structure.weak_rows import slot_of, term_columns
from epde.structure.factor import Factor

from epde.operators.common.stability import (GramSetup, VaryingCoefSetup)

from epde import _loop_stats

# One-shot guard so a permanently-broken super-Gram fast path is reported
# once instead of being silently swallowed on every term sweep.
_SUPER_GRAM_FAILURE_REPORTED = False

#: Outer-pass cap of ``EqRightPartSelector.apply``. A pass that does not
#: converge shrinks the structure -- a term dropped, a factor power lowered --
#: so a candidate runs out of rewrites long before this. The cap is the
#: backstop for a form still being rewritten, and such a form is declined
#: with the structure the entry validator left: the one it came in with,
#: unless it had no derivative term of its variable and the validator drew
#: one. That drawn term is kept. Put back without it, the label would come
#: back to the validator, which would draw anew on every visit, and one of
#: the draws could be placed; with it, the label comes back to the same
#: passes and is declined at the cap again.
RPS_OUTER_MAX_ITER = 50

#: ``simplify_equation``'s third verdict, beside True (rewritten: select
#: again) and False (nothing left to rewrite): the candidate has no valid
#: reduced form without drawing new terms, so ``EqRightPartSelector.apply``
#: declines it.
DECLINED = 'declined'


def loDs_to_DoLs(arg: List[Dict[int, Any]]) -> Dict[int, list]:
    res = defaultdict(list)
    for feature in arg:
        for samp_key, samp_item in feature.items():
            res[samp_key].append(samp_item)
    return res

class EqRightPartSelector(CompoundOperator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    THIS OPERATOR OWNS THE FIT. Each candidate target is sparsified, given
    coefficients and only then scored (``_fit_and_score``) -- the fitness
    hosts are pure scorers that neither fit nor prune. It is also the single
    place that prunes: the one ``remove_zero_terms`` at the end of ``apply``,
    after the sweep has chosen a winner. (That prune cannot move into the
    sparsity operator, which runs per candidate and must keep every
    candidate term.)

    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the CompoundOperator class
        key - str, value - instance of a class, inhereted from the CompoundOperator.
        Suboperators, performing tasks of equation processing. Three are
        present, applied in this order per candidate target: ``sparsity``
        (support selection), ``coeff_calc`` (physical magnitudes) and
        ``fitness_calculation`` (the discrepancy candidates are ranked by).

    Methods:
    -----------
    apply(equation)
        return None
        Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``

    
    '''
    key = 'FitnessCheckingRightPartSelector'

    @staticmethod
    @_loop_stats.timed('EqRPS.gram_super')
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
            # Guard the source this function actually reads: the per-trajectory
            # manager (the old ``grid_cache`` check survived the multisample
            # port but no longer gates anything used below).
            if getattr(global_var, 'samples_manager', None) is None:
                objective._gram_super = None
                return
            slot = slot_of(objective)
            sample_weights = global_var.samples_manager.row_weights(slot, 'dm')
            grid_shapes = global_var.samples_manager.row_shapes(slot)
            feats = [term_columns(term, slot) # grids=None
                     for term in objective.structure]
            feats = loDs_to_DoLs(feats)
            
            Z = dictApplyUFunc(lambda x: np.vstack(x).T, feats)
            if not all([np.all(np.isfinite(z)) for z in Z.values()]):
                objective._gram_super = None
                _loop_stats.record('EqRPS.gram_super_skip', 1, 1)
                return
            gram_mode = active_config().objectives.gram_mode
            if gram_mode == 'vcoef':
                objective._gram_super = VaryingCoefSetup.precompute_super(Z, sample_weights, grid_shapes,
                                                                          main_var=objective.main_var_to_explain)
            elif gram_mode == 'axis':  # backup
                objective._gram_super = GramSetup.precompute_super(Z, sample_weights, grid_shapes)
            else:
                # Basis-free instability metric: no Gram is needed, but the
                # cached Z still earns its keep -- it is what lets the sweep
                # slice out each candidate's target/features without
                # re-evaluating the terms.
                objective._gram_super = {'mode': None, 'Z': Z}
            _loop_stats.record('EqRPS.gram_super_built', 1, 1)
        except Exception as exc:
            # Defensive: any unexpected failure (shape mismatch, missing
            # cache) means we fall back -- numerics are preserved, only the
            # speedup is lost. Reported ONCE per process rather than
            # swallowed: a permanently-throwing fast path is invisible
            # otherwise (it cost every sweep of the multisample port, where
            # the dict of inner shapes was iterated as if it were a tuple).
            objective._gram_super = None
            _loop_stats.record('EqRPS.gram_super_skip', 1, 1)
            global _SUPER_GRAM_FAILURE_REPORTED
            if not _SUPER_GRAM_FAILURE_REPORTED:
                _SUPER_GRAM_FAILURE_REPORTED = True
                warnings.warn(
                    'EqRightPartSelector._precompute_super_gram: the tier-3 '
                    f'super-Gram fast path is disabled ({type(exc).__name__}: '
                    f'{exc}). Falling back to the per-target path for the rest '
                    'of this run.')

    def set_suboperators(self, operators: dict, probas: dict = {}):
        """Wire the suboperators, then tell ``coeff_calc`` what scale the
        support decision it will be handed was fitted on.

        The sparsity operator declares this about itself
        (``fits_physical_scale``); pushing it across here once per run keeps the
        two strategy builders -- which already hand this operator both
        suboperators together -- unaware of the coupling, and keeps
        ``coeff_calc`` from having to reach back into the equation for a marker.
        """
        super().set_suboperators(operators, probas)
        sparsity = operators.get('sparsity')
        coeff_calc = operators.get('coeff_calc')
        if sparsity is not None and coeff_calc is not None:
            coeff_calc.sparsity_fits_physical = bool(
                getattr(sparsity, 'fits_physical_scale', False))

    def _fit_and_score(self, objective: Equation, subop_args: dict):
        """Fit ``objective`` for its CURRENT ``target_idx``, then score it.

        This operator owns the fit. ``sparsity`` decides the support and
        ``coeff_calc`` puts physical magnitudes on it; only then is the
        fitness host -- a pure scorer that neither fits nor prunes -- asked
        for the candidate's discrepancy. ``None`` comes back for a candidate
        the host declines (all-zero support, a discrepancy over the degeneracy
        threshold, or an amplified near-null fit).
        """
        self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])
        return self.suboperators['fitness_calculation'].apply(
            objective, arguments=subop_args['fitness_calculation'],
            force_out_of_place=True)

    @_loop_stats.timed('EqRPS.apply')
    @HistoryExtender('\n -> The equation structure was detected: ', 'a')
    def apply(self, objective : Equation, arguments : dict):
        """Select a right-part term for ``objective`` in-place.

        THE RESULT IS A FUNCTION OF THE CANDIDATE. Nothing is drawn here
        after entry: the term sweep fits every derivative-carrying target, and
        ``simplify_equation`` only drops terms or lowers factor powers of the
        winner, so the equation that leaves is built from the terms that came
        in. A candidate with no valid fitted form of that kind leaves DECLINED
        (:meth:`_decline`): every eligible target declined by the host, a
        reduction that would leave no term with a derivative of the variable,
        leave the target with no meaningful factor or leave fewer than two
        terms, or a form still being rewritten at ``RPS_OUTER_MAX_ITER``.
        Every decline is decided before the pass that makes it rewrites
        anything; at the cap, where earlier passes did rewrite, the structure
        the first pass received is put back (:func:`_restore_structure`).
        Redrawing it is the population operators' job:
        ``regenerate_degenerate_equations`` redraws a declined equation up to
        ``moeadd_specific.RESAMPLE_BUDGET`` times, as many as this operator's
        outer cap once allowed, and then the initial sort retries or the
        offspring updater drops the system. This operator used to reroll such
        a candidate itself (``objective.randomize()``, ``term.randomize()``)
        until a form was accepted, so its output did not depend on its input,
        and under a strict keep rule the population collapsed onto the few
        forms the rerolls reached.
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

        # The one validity repair left, on ENTRY only: a candidate with no
        # derivative of its variable has no admissible right part at all, and
        # ``Equation.__init__`` and the crossover leave that guarantee to this
        # point. ``restore_property`` is a validator -- it installs a
        # derivative-carrying term or raises on a pool that cannot make one --
        # so one call suffices. Every later pass keeps a term with such a
        # derivative in the structure (``_cancellation_leaves_no_derivative``)
        # and draws nothing.
        restored = not any(term.contains_deriv(objective.main_var_to_explain)
                           for term in objective.structure)
        if restored:
            objective.restore_property(mandatory_family=False, deriv=True)
        _loop_stats.record('EqRPS.inner_derivative', int(restored), 1)
        # What the cap puts back. Read after the validator, which draws: a
        # label declined at the cap comes back to these same passes, not to
        # a new draw.
        entry = _structure_snapshot(objective)

        outer_attempts = 0
        # Loop control is LOCAL: an offspring must not inherit "already
        # simplified" from its parent and skip the ``reset_state(True)`` below,
        # so the body always runs at least once and the equation is always
        # refitted for a target this invocation chose.
        while True:
            outer_attempts += 1
            if outer_attempts > RPS_OUTER_MAX_ITER:
                # Every pass that did not converge shrank the structure, so a
                # form still being rewritten here is declined. Accepting it, as
                # this used to, let a derivative-free target and forms
                # ``simplify_equation`` would still rewrite leave RPS. Declined
                # as the first pass received it (see ``RPS_OUTER_MAX_ITER``):
                # left rewritten, it came back with a fresh pass budget and
                # could be placed.
                warnings.warn(
                    'EqRightPartSelector.apply: outer loop did not converge '
                    f'after {RPS_OUTER_MAX_ITER} iterations; declining the candidate.')
                _restore_structure(objective, entry)
                return self._decline(objective, 'outer_cap', outer_attempts - 1)
            objective.reset_state(True)
            min_fitness = np.inf
            # One slot per non-target term plus the trailing intercept.
            weights_internal = np.zeros(len(objective.structure))
            min_idx = 0

            # Tier 3: precompute the super-Gram over all terms ONCE per
            # outer iter so the term-sweep below derives per-target
            # GramSetup views via pure slicing instead of rebuilding the
            # windowed XTWX matmul for every candidate.
            self._precompute_super_gram(objective)

            with _loop_stats.timer('EqRPS.term_sweep'):
                for target_idx, target_term in enumerate(objective.structure):
                    if not objective.structure[target_idx].contains_deriv(objective.main_var_to_explain):
                        continue
                    objective.target_idx = target_idx
                    fitness = self._fit_and_score(objective, subop_args)
                    if fitness is not None and fitness < min_fitness:
                        min_fitness = fitness
                        min_idx = target_idx
                        weights_internal = objective.weights_internal
                        weights_final = objective.weights_final
                        sw_weights = objective._cached_sw_weights
                        vc_score = objective._cached_vc_score

                    objective.weights_internal_evald = False
                    objective.weights_final_evald = False

            if np.isinf(min_fitness):
                # Every eligible target declined by the host (all-zero
                # support, a discrepancy over the degeneracy threshold, an
                # amplified fit): the candidate has no right part. Declined,
                # not rerolled -- see the docstring.
                return self._decline(objective, 'all_targets_declined', outer_attempts)

            objective.weights_internal = weights_internal
            objective.weights_final = weights_final
            objective._cached_sw_weights = sw_weights
            objective._cached_vc_score = vc_score
            objective.weights_internal_evald = True
            objective.weights_final_evald = True
            objective.target_idx = min_idx

            outcome = self.simplify_equation(objective)
            if outcome == DECLINED:
                return self._decline(objective, 'simplify', outer_attempts)
            if not outcome:
                break

        _loop_stats.record('EqRPS.outer', outer_attempts, RPS_OUTER_MAX_ITER)
        # Drop the super-Gram so a downstream consumer (e.g. fitness
        # recomputation outside the term-sweep) falls back to the
        # per-target ``GramSetup.__init__`` path; the cached super-Gram
        # is only valid for the structure observed during the sweep.
        objective._gram_super = None
        objective.right_part_selected = True
        # EXIT CONTRACT: the loop leaves only by converging -- the sweep's
        # winner installed with its fit, and ``simplify_equation`` changing
        # nothing -- so the equation is FITTED for the target it carries, and
        # that target holds a derivative of the explained variable. The exit
        # guarantee and the refit that used to stand here served the cap break
        # and the rerolls, which now decline instead.
        objective.remove_zero_terms()
        objective.assert_state_invariants('EqRightPartSelector.apply exit')
        # Hard invariant: no duplicate terms may leave RPS. simplify and
        # scrub drop rather than redraw, so a surviving duplicate is a
        # logic error to surface HERE -- not one generation later at the
        # crossover/mutation assert (the crash site that "lies").
        _final_sigs = {term.factors_labels for term in objective.structure}
        assert len(_final_sigs) == len(objective.structure), \
            'EqRightPartSelector.apply: duplicate terms survived RPS.'

    def _decline(self, objective: Equation, reason: str, passes: int) -> None:
        """Leave ``objective`` DECLINED: fitted-but-degenerate, its objectives
        stamped ``LOSS_NAN_VAL`` (:func:`_mark_unassignable`, the verdict the
        assignment mode gives an equation it cannot fit). It is the stamp the
        fitness host gives a degenerate fit, which the population operators
        read (``moeadd_specific._equation_is_degenerate``) to redraw or drop
        the equation and the single-objective sort reads to rank it last.

        The stamp stands through the chromosome fitness. Alone in its system,
        the equation is not scored at all: the raised ``fitness_calculated``
        fails the chromosome condition. Beside an unscored equation, the
        solver-free host re-scores it in place and stamps it again (all-zero
        term weights are degenerate, ``is_degenerate``), and the solver host
        does not solve the system (``SolverBasedFitness.apply``). Zero weights
        for an installed target keep the fitted-state contract, and
        ``right_part_selected`` keeps the chromosome-level condition (some
        equation unselected) from re-running selection for it alone.

        The structure stays as the declining pass received it, unpruned: no
        pass rewrites anything before it declines (``simplify_equation``
        decides each of its declines before it reduces or drops). So the label
        left behind is the one that pass declined, and as RPS is a function of
        the candidate, that label is declined again whenever it comes back.
        At the outer cap the earlier passes did rewrite it, so ``apply`` first
        puts back the structure the first pass received
        (:func:`_restore_structure`): that label comes back to the same
        passes and is declined at the cap again, where the rewritten one came
        back with a fresh pass budget and could be placed."""
        _loop_stats.record('EqRPS.outer', passes, RPS_OUTER_MAX_ITER)
        _loop_stats.record('EqRPS.declined.' + reason, 1, 1)
        _mark_unassignable([objective], self.suboperators['fitness_calculation'])

    def simplify_equation(self, objective: Equation):
        """Rewrite the fitted winner into its canonical form, in place.

        Returns True when the structure changed (the caller selects again),
        False when there is nothing left to rewrite, and ``DECLINED`` when the
        candidate has no valid reduced form: a cancellation that would leave no
        term with a derivative of the variable, one that leaves the target with
        no meaningful factor, or a reduction that would leave fewer than two
        terms. Every rewrite drops terms or lowers factor powers; none draws a
        term. Each decline is decided before anything is rewritten, so a
        declined candidate keeps the structure it came in with."""
        # Get nonzero terms
        tgt = objective.target_idx
        # ``[:-1]`` drops the trailing intercept slot explicitly. Without it the
        # mask is one longer than ``nonrs_terms`` and only the ``zip`` below
        # hides the mismatch.
        nonzero_terms_mask = np.array([False if weight == 0 else True
                                       for weight in objective.weights_internal[:-1]], dtype=np.int32)
        nonrs_terms = [term for i, term in enumerate(objective.structure) if i != tgt]
        nonzero_terms = [item for item, keep in zip(nonrs_terms, nonzero_terms_mask) if keep]
        nonzero_terms.append(objective.target)
        equation_terms = [term.factors_labels_without_power for term in nonzero_terms]

        if len(equation_terms) <= 1:
            return False

        # Ratio-fit scrub: a nonzero feature term related to the target by
        # DIVISION in either direction (see ``_term_divides``) makes the
        # ``Lambda * g * (true identity)`` FD-error soak representable:
        #
        # * a COMPONENT (term divides target) -- the wave composite trap
        #   ``u*u_tt = 3.75*u_tt - 0.15*u_xx + 0.04*u*u_xx`` (g = 1), whose
        #   padded pair cancels exactly under the true relation;
        # * a MULTIPLE (target divides term) -- the hitchhiker shape
        #   ``0.687*u_t*u_tt - 0.0275*u_t*u_xx`` riding on the canonical
        #   target ``u_tt`` (g = u_t; the coefficient ratio is again
        #   exactly the true 0.04).
        #
        # Dropping the target-side member breaks every such family (the
        # partner alone cannot cancel and dies at the refit -- measured
        # under both the chi2 and vcoef keep-rules). Component padding
        # also BLOCKS the common-factor cancellation below; dropping it
        # first lets the surviving reduced identity (which DOES share the
        # factor) collapse to its canonical form on a later pass. The term is
        # dropped, not redrawn: a redraw put a random term in its place.
        divisors = [term for term in nonzero_terms[:-1]
                    if _related_to_target(term, objective.target)]
        if divisors:
            # Two terms cannot lose one: a candidate the drops would take
            # below two terms is the soak alone, its target and related
            # terms. Decided before anything is dropped, so the declined
            # candidate keeps the structure it came in with.
            if len(divisors) > len(objective.structure) - 2:
                return DECLINED
            for term in divisors:
                _loop_stats.record('simplify_equation.divisor_scrub', 1, 1)
                status = _drop_term(objective, term)
                assert status == 'dropped', status
            objective.reset_for_structure_change()
            return True

        # Degree reduction: when a SINGLE non-target term remains, the
        # equation is ``coef * f = g`` with f, g products of powered
        # factors. If every factor power in BOTH f and g shares a common
        # divisor p >= 2, the whole equation is a p-th power -- take the
        # p-th root (divide every power by p; the coefficient is recomputed
        # downstream). E.g. ``c*(u_xx)^2 = (u_tt)^2`` -> ``sqrt(c)*u_xx = u_tt``.
        # Keeps the lowest-degree equivalent form so it is not penalised /
        # mistaken for a distinct higher-order structure.
        if len(nonzero_terms) == 2:
            powers, integral = [], True
            for term in nonzero_terms:
                for factor in term.structure:
                    for i in factor.params_description:
                        if factor.params_description[i]["name"] == "power":
                            p = factor.params[i]
                            if float(p) != int(p) or int(p) < 1:
                                integral = False
                            powers.append(int(p))
            if integral and powers:
                root = int(np.gcd.reduce(np.array(powers, dtype=int)))
                if root >= 2:
                    # p-th root: divide every factor power by the gcd,
                    # collapsing the equation to its lowest equivalent
                    # degree (c*(u_xx)^2=(u_tt)^2 -> sqrt(c)*u_xx=u_tt).
                    for term in nonzero_terms:
                        for factor in term.structure:
                            for i in factor.params_description:
                                if factor.params_description[i]["name"] == "power":
                                    factor.set_param(int(factor.params[i]) // root, idx=i)
                        term.resetSavedState()
                    # The reduction can collapse a survivor onto a zero-weight
                    # candidate already in the structure (u^2 -> u when a u
                    # term exists). Such a colliding copy is ALWAYS a
                    # zero-weight non-survivor -- two genuinely nonzero terms
                    # cannot collide via a p-th root unless they were already
                    # equal, which the entry assert forbids -- so drop the
                    # redundant copies outright (drop-immediately). The reduced
                    # low-degree form survives on the kept terms; no revert.
                    keep_ids = {id(t) for t in nonzero_terms}
                    kept_labels = {t.factors_labels for t in nonzero_terms}
                    redundant = [t for t in objective.structure
                                 if id(t) not in keep_ids
                                 and t.factors_labels in kept_labels]
                    for t in redundant:
                        _regen_or_drop_term(
                            objective, t, max_iter=0,
                            stats_name='simplify_equation.degree_reduction')
                    objective.reset_for_structure_change()
                    return True
        common_factors = list(frozenset.intersection(*equation_terms))
        if not common_factors:
            # No symbolic reduction applies -- run the numeric completion of
            # the duplicate-term invariant: exactly linearly dependent
            # nonzero feature columns (the generalization of a duplicate
            # term the label test cannot see, e.g. two symbolically distinct
            # products evaluating to proportional fields) make the active-
            # support OLS refit rank-deficient ("infinite solutions"), so
            # such terms are dropped like duplicates. The target is
            # never touched: a target in the SPAN of independent features is
            # a perfect fit (a valid identity), not a degeneracy.
            return self._drop_dependent_terms(objective, nonzero_terms[:-1])

        for common_factor in common_factors:
            # Min power across the matching factor in every nonzero term.
            min_order = np.inf
            for term in nonzero_terms:
                for factor in term.structure:
                    if factor.structural_label_without_power == common_factor:
                        if factor.cache_label[1][0] < min_order:
                            min_order = factor.cache_label[1][0]

            # A reduction that would leave NO term with a derivative of the
            # variable leaves no right part to select: the candidate is a
            # factored identity (``x^2*u' = c*u'`` is ``(x^2 - c)*u' = 0``).
            # Declined before anything is reduced, rather than reduced and
            # handed to ``restore_property`` for a random derivative term on
            # the next pass, as it used to be. A target that sheds its
            # derivative while another term keeps one is no such case: the
            # next pass re-selects among the reduced terms
            # (``f*u' = c*u'*u''`` becomes ``u'' = f/c``) -- provided the
            # clean-up below accepts what the reduction leaves.
            if _cancellation_leaves_no_derivative(objective, nonzero_terms,
                                                  common_factor, min_order):
                return DECLINED
            # The clean-up after the reduction declines as well: a target
            # left with no meaningful factor (``sin*u'`` sheds ``u'`` into
            # ``sin``, a term the grammar does not allow), or a feature left
            # with none that cannot be dropped above the two-term floor. The
            # reduction runs in place and cannot be undone, so both verdicts
            # are read off the structure here, before it runs: a declined
            # candidate keeps the structure it came in with, and that label
            # is declined again when it comes back.
            if _cancellation_leaves_no_valid_form(objective, nonzero_terms,
                                                  common_factor, min_order):
                return DECLINED

            # Reduce order of common factor in every term; drop zero-power factors.
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
                term.resetSavedState()

            # Reduction collisions are ALWAYS against zero-weight
            # non-survivors: two NONZERO terms cannot collide by shedding a
            # factor they both carried (they would have been duplicates
            # before), so a colliding copy is dead structure -- drop the
            # copy, keep the reduced carrier (the degree-reduction policy
            # above). Regenerating the CARRIER instead -- the old policy --
            # scattered exact identities into random terms and broke the
            # canonical collapse (the wave ratio-fit chain).
            keep_ids = {id(t) for t in nonzero_terms}
            kept_labels = {t.factors_labels for t in nonzero_terms}
            redundant = [t for t in objective.structure
                         if id(t) not in keep_ids
                         and t.factors_labels in kept_labels]
            for t in redundant:
                _regen_or_drop_term(
                    objective, t, max_iter=0,
                    stats_name='simplify_equation.cancel_collision')

            # A term that consisted of nothing but the common factor is now
            # empty / non-meaningful: DROP it (``max_iter=0``). An empty term
            # is the constant the intercept already fits, and a term with no
            # meaningful factor is one the term grammar does not allow;
            # redrawing it, as this used to, put a random term in its place.
            # A duplicate must never ride out of RPS -- see the exit assert in
            # ``apply``.
            for term in nonzero_terms:
                status = _regen_or_drop_term(
                    objective, term, max_iter=0,
                    stats_name='simplify_equation.replace_term')
                # A target left with no meaningful factor and a drop below
                # the two-term floor were declined before the reduction
                # (``_cancellation_leaves_no_valid_form``).
                assert status in ('ok', 'dropped'), \
                    f'simplify_equation: cancellation clean-up returned {status!r}'

            # Structure changed: invalidate stale fitness /
            # weights / AIC caches while leaving RPS to the
            # caller's outer loop.
            objective.reset_for_structure_change()
            return True
        # Unreachable: the loop body above returns on every path.
        raise AssertionError('simplify_equation: common-factor loop fell through')

    def _drop_dependent_terms(self, objective: Equation, feature_terms):
        """Drop nonzero non-target terms whose evaluated columns are EXACTLY
        linearly dependent (incl. against the intercept).

        Such columns leave the active-support coefficient solve with
        infinitely many solutions (the observed lv / lorenz failure). Where a
        column is flagged it lies in the span of the intercept and the columns
        kept before it, so it is dropped like a duplicate label, and nothing is
        drawn in its place (a redraw used to be). Strictly machine-exact
        (``_LIN_DEP_RTOL`` on unit columns): near-collinear physics --
        including valid analytical identities -- is never touched, in keeping
        with the no-collinearity-machinery principle.

        MULTISAMPLE SEMANTICS. The coefficient fit is run PER TRAJECTORY and
        the resulting coefficient vectors are averaged (``VWSRSparsity.apply``),
        so a rank-deficient solve in ONE sample already corrupts that sample's
        coefficients and thereby the average. A term is therefore flagged when
        its column is exactly dependent in ANY sample (union). Each sample
        keeps its OWN basis, normalised by its OWN column norm -- the same
        convention ``_flag_dependent_columns`` uses -- and nothing is ever
        concatenated across samples: trajectories differ in length and grid,
        so a stacked residual would not be comparable to ``_LIN_DEP_RTOL``.

        Returns True iff a term was dropped (the caller's outer RPS loop then
        re-runs the term sweep and re-checks), False when nothing is flagged,
        and ``DECLINED`` when a flagged term cannot go without leaving fewer
        than two terms.
        """
        if not feature_terms:
            return False

        cols = loDs_to_DoLs([term_columns(term, slot_of(objective))
                             for term in feature_terms]) # grids=None
        cols: Dict[int, List[np.ndarray]] = dictApplyUFunc(lambda x: [np.asarray(elem, dtype=float).reshape(-1) for elem in x],
                                                           cols)

        scan: Dict[int, Tuple[List[int], List[np.ndarray]]] = dictApplyUFunc(_flag_dependent_columns, cols)
        if any([sc is None for sc in scan.values()]):
            # Non-finite column: dependence undecidable, mirror the
            # super-Gram skip rather than dropping on a data problem.
            _loop_stats.record('simplify_equation.lin_dep_skip', 1, 1)
            return False

        # Per-sample flagged column indices. They address ``feature_terms``,
        # which is sample-independent, so they are directly unionisable.
        flagged_union = sorted({idx for elem in scan.values() for idx in elem[0]})

        if not flagged_union:
            return False

        # Two terms cannot lose one: with two terms the only feature is a
        # constant or zero column, which the intercept already is. Decided
        # before anything is dropped, so the declined candidate keeps the
        # structure it came in with.
        if len(flagged_union) > len(objective.structure) - 2:
            return DECLINED
        for i in flagged_union:
            _loop_stats.record('simplify_equation.lin_dep', 1, 1)
            status = _drop_term(objective, feature_terms[i])
            assert status == 'dropped', status
        objective.reset_for_structure_change()
        return True

    def use_default_tags(self):
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}

        
class RandomRHPSelector(CompoundOperator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    THIS OPERATOR OWNS THE FIT. Each candidate target is sparsified, given
    coefficients and only then scored (``_fit_and_score``) -- the fitness
    hosts are pure scorers that neither fit nor prune. It is also the single
    place that prunes: the one ``remove_zero_terms`` at the end of ``apply``,
    after the sweep has chosen a winner. (That prune cannot move into the
    sparsity operator, which runs per candidate and must keep every
    candidate term.)

    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the CompoundOperator class
        key - str, value - instance of a class, inhereted from the CompoundOperator.
        Suboperators, performing tasks of equation processing. Three are
        present, applied in this order per candidate target: ``sparsity``
        (support selection), ``coeff_calc`` (physical magnitudes) and
        ``fitness_calculation`` (the discrepancy candidates are ranked by).

    Methods:
    -----------
    apply(equation)
        return None
        Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``

    
    '''
    key = 'RandomRightPartSelector'

    @HistoryExtender('\n -> The equation structure was detected: ', 'a')
    def apply(self, objective : Equation, arguments : dict):
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
            objective.reset_explaining_term(idx)
            objective.right_part_selected = True


    def use_default_tags(self):
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}


# Relative projection residual (on unit-normalized columns) below which an
# evaluated term column counts as EXACTLY linearly dependent on the columns
# before it. This is a machine-exactness test for rank deficiency ("infinite
# solutions" in the OLS/Gram solves), NOT a collinearity threshold: exact
# dependence projects to ~eps*conditioning, while even severely near-collinear
# physics (incl. valid analytical identities) sits many orders of magnitude
# above 1e-10.
_LIN_DEP_RTOL = 1e-10


def _flag_dependent_columns(cols, rtol: float = _LIN_DEP_RTOL) -> Tuple[List[int], List[np.ndarray]]:
    """Greedy exact-dependence scan over evaluated feature columns.

    Seeds the kept basis with the intercept column (the coefficient fits
    include one, so an exactly-constant term column is redundant with it),
    then walks ``cols`` in order: a column whose unit-normalized projection
    residual on the kept basis falls below ``rtol`` -- or an all-zero
    column -- is flagged as dependent; otherwise it joins the basis.
    Dependence is weight-invariant (a diagonal sample weighting preserves
    exact linear dependence), so the unweighted columns are checked.

    Returns ``(flagged_indices, basis)`` where ``basis`` is the list of
    kept unit columns (intercept first), or ``None`` when any column is
    non-finite -- the check is not applicable then (mirrors the
    ``_precompute_super_gram`` skip).
    """
    n = cols[0].size
    basis = [np.full(n, 1.0 / np.sqrt(n))]
    flagged = []
    for i, col in enumerate(cols):
        if not np.all(np.isfinite(col)):
            return None
        nrm = np.linalg.norm(col)
        if nrm == 0.0:
            flagged.append(i)
            continue
        u = col / nrm
        K = np.stack(basis, axis=1)
        coef, *_ = np.linalg.lstsq(K, u, rcond=None)
        if np.linalg.norm(u - K @ coef) < rtol:
            flagged.append(i)
        else:
            basis.append(u)
    return flagged, basis


def amplification_ratio(objective: Equation) -> float:
    """Amplification ratio of the CURRENT candidate right-part fit:

        A = sum_j |c_j| * ||col_j||  /  ||target col||

    over the nonzero non-target terms plus the fitted intercept
    (``weights_internal = [*term_coefs, intercept]``, the unified layout of
    ``Equation._validate_weight_layout``). The intercept's design column is the
    constant ones-vector, so its contribution ``|c_0| * sqrt(n)`` is the SAME
    ``|c_j| * ||col_j||`` rule -- ``sqrt(n) == ||1||_2`` -- just written out
    because that column is never materialized. It used to be unreachable: the
    guard was ``len(w) > len(nonrs)``, false while the sparsity operators
    emitted a length-m ``weights_internal``, so every recorded A (truth anchors
    1.0-6.65 against the cap of 100.0) was measured WITHOUT it. A ~ 1 when the terms combine
    without cancellation to the target's magnitude (every truth anchor
    measures A in [1.0, 6.65]); A >> 1 is the amplified-identity parasite,
    where near-cancelling giant coefficients fit the target out of the
    noise of a valid near-null feature combination. Returns ``inf`` on a
    non-finite/degenerate evaluation so the caller declines the candidate.
    """
    # READ-ONLY evaluations: ``grids=()`` (non-None) makes Term.evaluate skip
    # its tensor_cache.add while serving cache hits normally and computing
    # misses identically (the Term-level override never forwards ``grids`` to
    # the actual computation). The ratio is a pure diagnostic -- it must not
    # mutate cache state (adds/evictions/saved flags), or its extra
    # evaluations perturb later fits at float precision and break run
    # reproducibility (observed as 1e-7-level legacy A/B drift when the ratio
    # started running for every sweep candidate).
    w = np.asarray(objective.weights_internal, dtype=float)
    if not np.all(np.isfinite(w)):
        return np.inf
    tgt = objective.target_idx
    slot = slot_of(objective)
    t = dictApplyUFunc(lambda x: np.asarray(x, dtype=float).reshape(-1),
                       term_columns(objective.structure[tgt], slot)) # grids=()
    den = dictApplyUFunc(np.linalg.norm, t)
    if (not all(list(dictApplyUFunc(np.isfinite, den).values())) or
        any([d == 0.0 for d in den.values()])):
        return np.inf
    nonrs = [term for i, term in enumerate(objective.structure) if i != tgt]

    num = defaultdict(float) # 0.0
    for wj, term in zip(w[:-1], nonrs):
        if wj == 0.0:
            continue
        col = dictApplyUFunc(lambda x: np.asarray(x, dtype=float).reshape(-1), term_columns(term, slot)) # grids=()
        num = dictAdd(num, dictApplyUFunc(lambda x: abs(wj) * np.linalg.norm(x), col), suppressed=True)

    if w[-1] != 0.0:
        # Intercept column of ones: its norm is sqrt(n) PER SAMPLE, and ``t`` is
        # the dict of per-sample target columns -- not a single array.
        num = dictAdd(num, dictApplyUFunc(lambda x: abs(w[-1]) * np.sqrt(x.size), t),
                      suppressed=True)

    if not num:
        # Every term weight and the intercept are 0: nothing reaches the
        # target, so nothing is amplified -- the empty sum is 0, the natural
        # ratio (A >= 0, and A < 1 is "less than the target's magnitude").
        # The per-sample divide below would get an empty numerator dict
        # against the target's keys and raise.
        return 0.0

    num = dictApplyUFunc(np.divide, num, den)

    return np.mean(list(num.values()))


def _term_divides(term, target_term) -> bool:
    """True iff ``term`` DIVIDES ``target_term``: every factor of ``term``
    appears in ``target_term`` (same ``structural_label_without_power``)
    with at least the same power (``cache_label[1][0]``, the
    ``simplify_equation`` power convention).

    Used by the ratio-fit scrub in ``simplify_equation``: a divisor
    feature spans an exact algebraic component of the target (``u_tt``
    inside the composite target ``u * u_tt``), which is what makes the
    ``Lambda * (true identity)`` FD-error soak representable. Measured on
    the wave pool: with the divisor excluded, the leftover padding dies
    under both the chi2 and vcoef keep-rules, leaving the reduced
    identity that common-factor cancellation collapses to the canonical
    form."""
    for f in term.structure:
        matched = False
        for g in target_term.structure:
            if (g.structural_label_without_power
                    == f.structural_label_without_power
                    and g.cache_label[1][0] >= f.cache_label[1][0]):
                matched = True
                break
        if not matched:
            return False
    return True


def _related_to_target(term, target_term) -> bool:
    """True iff ``term`` is a COMPONENT of ``target_term`` (divides it) or a
    MULTIPLE of it (is divided by it) -- exactly the terms the ratio-fit scrub
    in ``simplify_equation`` drops."""
    return _term_divides(term, target_term) or _term_divides(target_term, term)


def _cancellation_remainder(term, common_factor, min_order) -> list:
    """What cancelling ``common_factor`` leaves of the fitted ``term``,
    read-only: ``(factor, params)`` for every factor that stays, ``params``
    being its param vector after the reduction. The reduction is the one
    ``simplify_equation`` performs in this pass: the factor's power drops by
    ``min_order`` (its lowest power across the fitted terms), and the factor
    goes where that leaves 0; the other factors are untouched."""
    remainder = []
    for factor in term.structure:
        params = factor.params
        if factor.structural_label_without_power == common_factor:
            params = np.array(params, copy=True)
            powers = [i for i in factor.params_description
                      if factor.params_description[i]["name"] == "power"]
            for i in powers:
                params[i] = params[i] - min_order
            if any(params[i] == 0 for i in powers):
                continue
        remainder.append((factor, params))
    return remainder


def _remainder_labels(remainder) -> frozenset:
    """The ``Term.factors_labels`` a reduced term will carry: each factor's
    ``Factor.structural_label``, read on a stand-in holding its reduced
    params."""
    return frozenset(
        (factor.cache_label[0],
         Factor._quantized_params(SimpleNamespace(
             label=factor.label, params=params,
             params_description=factor.params_description,
             equality_ranges=factor.equality_ranges)))
        for factor, params in remainder)


def _cancellation_leaves_no_derivative(objective: Equation, nonzero_terms,
                                       common_factor, min_order) -> bool:
    """True when cancelling ``common_factor`` out of ``nonzero_terms`` (the
    fitted terms, target last) would leave NO term of ``objective.structure``
    with a derivative of the equation's variable -- the state in which the
    next pass used to call ``restore_property`` for a random derivative term.

    The reduction is :func:`_cancellation_remainder`'s; the zero-weight terms
    are untouched. The derivative test is ``Term.contains_deriv`` itself --
    the term sweep's eligibility test and the entry restore's -- read on the
    factors that would remain, so a cancellation can also make a term
    eligible (``u'*u''`` sheds ``u'`` into ``u''``). Whether the TARGET keeps
    its derivative does not matter: the next pass selects among the terms
    that carry one."""
    variable = objective.main_var_to_explain
    remainders = {id(term): _cancellation_remainder(term, common_factor, min_order)
                  for term in nonzero_terms}

    for term in objective.structure:
        if id(term) in remainders:
            term = SimpleNamespace(structure=[factor for factor, _ in remainders[id(term)]])
        if Term.contains_deriv(term, variable):
            return False
    return True


def _cancellation_leaves_no_valid_form(objective: Equation, nonzero_terms,
                                       common_factor, min_order) -> bool:
    """True when the clean-up after cancelling ``common_factor`` out of
    ``nonzero_terms`` (the fitted terms, target last) would decline the
    candidate. Read-only: ``simplify_equation`` asks before it reduces, as the
    reduction runs in place and cannot be undone.

    The clean-up is ``simplify_equation``'s, in its order. The zero-weight
    terms a reduced term now duplicates go first; that never reaches the
    floor, as the fitted terms alone make two. Each fitted feature left with
    no meaningful factor goes next, and a drop below two terms is the
    decline. A target left with no meaningful factor (or none at all) is the
    decline too, as the target cannot be dropped. Nothing else is dropped: no
    two fitted terms can duplicate each other after the reduction, as they
    would have before it. The tests are the clean-up's own,
    ``Term.contains_meaningful`` and ``Term.factors_labels``, read on what the
    reduction would leave (:func:`_cancellation_remainder`)."""
    remainders = [_cancellation_remainder(term, common_factor, min_order)
                  for term in nonzero_terms]
    reduced_labels = {_remainder_labels(remainder) for remainder in remainders}
    nonzero_ids = {id(term) for term in nonzero_terms}
    left = len(objective.structure) - sum(
        1 for term in objective.structure
        if id(term) not in nonzero_ids and term.factors_labels in reduced_labels)

    def meaningful(remainder) -> bool:
        return Term.contains_meaningful(
            SimpleNamespace(structure=[factor for factor, _ in remainder]))

    *features, target = remainders
    for remainder in features:
        if not meaningful(remainder):
            if left <= 2:
                return True
            left -= 1
    return not meaningful(target)


def _structure_snapshot(equation: Equation):
    """What the rewrites of ``simplify_equation`` can change in ``equation``,
    for :func:`_restore_structure`. The scrubs and the collision clean-up
    replace the term list; the cancellation and the degree reduction rewrite
    a term in place, its factor list and its factors' powers (``set_param``
    writes the params array in place). Nothing else of a Term or a Factor is
    written and nothing is drawn, so copies of the two lists and of each
    params array hold it all: a few small copies, and no draw."""
    return (list(equation.structure),
            [(term, list(term.structure),
              [(factor, factor.params.copy()) for factor in term.structure])
             for term in equation.structure])


def _restore_structure(equation: Equation, snapshot) -> None:
    """Put ``equation`` back as :func:`_structure_snapshot` read it: the same
    Term and Factor objects in their places, with their params. Written
    through the paths the rewrites use, so the memoized labels and saved
    evaluations go with them (the ``Factor.params`` setter,
    ``Term.resetSavedState``, ``Equation.reset_for_structure_change``)."""
    structure, terms = snapshot
    for term, factors, params in terms:
        changed = (len(term.structure) != len(factors)
                   or any(now is not then for now, then in zip(term.structure, factors)))
        for factor, values in params:
            if not np.array_equal(factor.params, values):
                factor.params = values
                changed = True
        if changed:
            term.structure = factors
            term.resetSavedState()
    equation.structure = structure
    equation.reset_for_structure_change()


def _drop_term(equation: Equation, term, *, min_terms: int = 2) -> str:
    """Remove the NON-target ``term`` from ``equation.structure`` together
    with its weight slot -- the drop half of ``_regen_or_drop_term``, which
    the scrubs of ``simplify_equation`` use on their own: they never redraw.

    Refuses to go below ``min_terms`` (``'floor'``, nothing removed) and to
    remove the RPS target (``'target'``); ``'dropped'`` otherwise. No manual
    ``target_idx`` decrement: the identity-tracked target follows its Term.
    Both weight vectors that are up lose the matching slot, so they keep
    describing the structure they are indexed against
    (``Equation._validate_weight_layout``).
    """
    target_term = equation.target
    if target_term is not None and term is target_term:
        equation._invalidate_label_cache()
        return 'target'
    if len(equation.structure) <= min_terms:
        equation._invalidate_label_cache()
        return 'floor'
    # Capture the dropped term's weight slot BEFORE mutating the structure.
    drop_idx = next((j for j, t in enumerate(equation.structure) if t is term), None)
    tgt_idx = equation.target_idx
    equation.structure = [t for t in equation.structure if t is not term]
    # Kept in step even though every caller drops the fit right after: a
    # vector one slot too long would be off by one for every term past the
    # dropped one if it ever reached ``remove_zero_terms``.
    if drop_idx is not None and tgt_idx is not None and drop_idx != tgt_idx:
        if getattr(equation, 'weights_internal_evald', False):
            wpos = equation.weight_index(drop_idx, tgt_idx)
            equation.weights_internal = np.delete(np.asarray(equation.weights_internal), wpos)
        if getattr(equation, 'weights_final_evald', False):
            wpos = equation.weight_index(drop_idx, tgt_idx)
            equation.weights_final = np.delete(np.asarray(equation.weights_final), wpos)
    equation._invalidate_label_cache()
    return 'dropped'


def _regen_or_drop_term(equation: Equation, term, *, max_iter: int = 100,
                        min_terms: int = 2,
                        stats_name: str = 'simplify_equation.regen_or_drop') -> str:
    """Make ``equation.structure`` unique w.r.t. ``term`` by regenerating
    ``term`` up to ``max_iter`` times; if it is still empty / non-meaningful
    / a duplicate, DROP it from the structure (:func:`_drop_term`).

    This is the *regenerate-n-then-drop* policy, deliberately distinct from
    the *keep-or-revert* ``retry_until_unique`` policy used by
    ``Equation.__init__`` and the mutation operators. ``max_iter == 0`` means
    "drop immediately if unacceptable" (no regeneration) -- the only form
    ``simplify_equation`` uses, since right-part selection draws no terms. The
    regenerating form serves the system rule's dedup (``_reroll_term``).

    The acceptability predicate ranges over the FULL structure, so a
    duplicate against a zero-weight candidate elsewhere is caught too. The
    RPS target is never dropped: if ``term`` is the target AND a duplicate,
    the OTHER member of its duplicate group is dropped instead; if the
    target is merely empty/non-meaningful, ``'target'`` is returned for the
    caller to handle. Refuses to drop below ``min_terms`` (returns
    ``'floor'``). On a drop, ``target_idx`` is reindexed exactly as in
    ``Equation.remove_zero_terms``.

    Returns one of ``'ok'`` (already acceptable), ``'regenerated'``,
    ``'dropped'``, ``'target'``, ``'floor'``.
    """
    idx = next((j for j, t in enumerate(equation.structure) if t is term), None)
    if idx is None:
        return 'ok'  # already dropped earlier in this pass

    def _acceptable():
        if len(term.structure) == 0 or not term.contains_meaningful():
            return False
        signatures = {t.factors_labels for t in equation.structure}
        return len(signatures) == len(equation.structure)

    cap = max_iter if max_iter > 0 else 1
    if _acceptable():
        _loop_stats.record(stats_name, 1, cap)
        return 'ok'

    attempts = 0
    for _ in range(max_iter):
        attempts += 1
        term.randomize()
        term.resetSavedState()
        if _acceptable():
            _loop_stats.record(stats_name, attempts, cap)
            equation._invalidate_label_cache()
            return 'regenerated'
    _loop_stats.record(stats_name, max(attempts, 1), cap)

    # Exhausted (or max_iter == 0): drop the offending term, if legal.
    target_term = equation.target          # the Term, or None
    drop_term = term
    if target_term is not None and term is target_term:
        # Can't drop the RPS target. If it duplicates another term, drop
        # that other (non-target) member; if it is merely empty/non-
        # meaningful, leave it for the caller to resolve.
        my_label = term.factors_labels
        other = next((t for t in equation.structure
                      if t is not term and t.factors_labels == my_label), None)
        if other is None:
            equation._invalidate_label_cache()
            return 'target'
        drop_term = other
    # ``drop_term`` is never the target: the branch above reroutes a target
    # collision to the other duplicate.
    return _drop_term(equation, drop_term, min_terms=min_terms)


def _equations_share_active_structure(eq_a: Equation, eq_b: Equation) -> bool:
    """True iff two equations of a system encode the same law rearranged:
    their ACTIVE structures (the target plus every nonzero-weight term, factor
    powers kept -- ``Equation.active_terms_labels``) are identical and
    non-empty.

    This is deliberately EXACT equality, not subset/superset: this predicate
    must NOT flag a system where one equation's active terms are a strict
    superset of another's (e.g. ``dv/dx0 = v + u + du/dx0`` alongside
    ``du/dx0 = u + v``). Such a superset does carry the other equation's
    target, so ``_carries_target_of`` repairs it. Which term each equation explains
    does not enter: the same term set under two different targets is one law
    solved for two quantities. The degenerate case this predicate targets is
    two equations collapsing into the SAME law (e.g. both Navier-Stokes
    velocity equations turning into the continuity law), which leaves the
    system with fewer independent equations than variables. A declined
    equation (:func:`_rps_declined`) states no law, so it shares none.
    """
    if _rps_declined(eq_a) or _rps_declined(eq_b):
        return False
    sig_a = eq_a.active_terms_labels
    return bool(sig_a) and sig_a == eq_b.active_terms_labels


def _rps_declined(equation) -> bool:
    """True for an equation right-part selection DECLINED
    (``EqRightPartSelector._decline``, :func:`_mark_unassignable`): zero
    weights and the ``LOSS_NAN_VAL`` stamp the population operators redraw it
    by. Its target is a placeholder, so it takes no part in the system's
    cross-equation rules: it would only reroll a healthy equation's term, or
    have its own placeholder rerolled, ahead of the redraw it is waiting for.
    The raised ``fitness_calculated`` tells the stamp from a sweep score, which
    the host writes without raising the flag."""
    from epde.operators.common.objectives import LOSS_NAN_VAL
    fitness = getattr(equation, 'fitness_value', None)
    return bool(getattr(equation, 'fitness_calculated', False)
                and fitness is not None and fitness >= LOSS_NAN_VAL)


def _reserved_target_signature(equation: Equation):
    """The factor signature of ``equation``'s target, or None when it reserves
    nothing: no target, a target without a derivative of the equation's own
    variable, or a declined equation (:func:`_rps_declined`). The second case
    is the placeholder :func:`_mark_unassignable` installs when no term holds
    such a derivative (``u*v``, say); reserving that ordinary feature would
    reroll it out of a healthy equation that holds it as a true term.
    """
    if _rps_declined(equation):
        return None
    target = equation.target
    if target is None or not target.contains_deriv(equation.main_var_to_explain):
        return None
    return target.factors_labels


def _carries_target_of(eq_with_target: Equation, eq_other: Equation) -> bool:
    """True iff ``eq_other`` carries ``eq_with_target``'s TARGET as a whole,
    standalone term of its ACTIVE structure (``Equation.active_terms_labels``:
    the target plus every nonzero-weight term, factor powers kept).

    Whole-term equality only. The target inside a product is coupling and is
    never flagged: continuity's ``v_y`` in the v-momentum convective term
    ``v*v_y``, the double pendulum's ``theta2''`` in ``cos(D)*theta2''``.
    Banning it as a factor would make both true systems unreachable. A
    different power is a different term too (``(dv/dx0)^2`` is not
    ``dv/dx0``).

    ``eq_other``'s own target counts: two equations explaining one composite
    target (``du/dx0 * dv/dx0``) are flagged. So does a strict superset of
    another equation's active set, which always holds that equation's target.
    The rule therefore subsumes the duplicate-equation guard, since equal
    active sets always hold each other's targets. Only a valid right part is
    reserved (``_reserved_target_signature``), and a declined ``eq_other``
    carries nothing (:func:`_rps_declined`).
    """
    sig = _reserved_target_signature(eq_with_target)
    return (sig is not None and not _rps_declined(eq_other)
            and sig in eq_other.active_terms_labels)


def _reroll_term(equation: Equation, term, banned_sigs, *, max_iter: int,
                 stats_name: str) -> None:
    """Redraw ``term`` in place until it leaves ``banned_sigs`` and duplicates
    no other term of ``equation``, then drop the equation's stale fit.

    The redraw keeps the Term object, so a rerolled TARGET stays the installed
    target (``reset_for_structure_change`` keeps it) and the caller's
    re-selection decides afresh which term explains the equation.

    On cap-hit a surviving duplicate is dropped via the ``_regen_or_drop_term``
    drop policy; a still-banned but unique term is tolerated, since the
    caller's convergence loop re-checks it.
    """
    attempts = 0
    for _ in range(max_iter):
        attempts += 1
        term.randomize()
        term.resetSavedState()
        signatures = {t.factors_labels for t in equation.structure}
        if (term.factors_labels not in banned_sigs
                and len(signatures) == len(equation.structure)):
            break
    _loop_stats.record(stats_name, attempts, max_iter)
    # Cap-hit may leave the rerolled term as a DUPLICATE -- a duplicate must
    # never ride out of RPS, so drop it (regenerate attempts already spent).
    _regen_or_drop_term(equation, term, max_iter=0,
                        stats_name=stats_name + '.drop')
    signatures = {t.factors_labels for t in equation.structure}
    if len(signatures) != len(equation.structure):
        # The drop was refused by the two-term floor. Uniqueness is the hard
        # invariant -- the next ``EqRightPartSelector.apply`` asserts on it --
        # while leaving ``banned_sigs`` is only what the loop above aimed for,
        # so redraw on uniqueness alone rather than hand a duplicate to the
        # caller.
        _regen_or_drop_term(equation, term, max_iter=100,
                            stats_name=stats_name + '.dedup')

    equation.reset_for_structure_change()


def _break_target_leak(equation: Equation, leak_sig, banned_sigs, *,
                       max_iter: int = 2000) -> bool:
    """Reroll the term of ``equation`` whose factor signature is ``leak_sig``,
    another equation's target (``_carries_target_of``).

    ``banned_sigs`` is every OTHER equation's target signature; a redraw onto
    any of them would only move the leak, so it is redrawn. The term may be
    ``equation``'s own target (one composite target explained twice); it is
    rerolled like any other term. Returns False, changing nothing, when no
    term carries ``leak_sig``; otherwise True, with the fit reset so the
    caller can re-run right-part selection.
    """
    term = next((t for t in equation.structure if t.factors_labels == leak_sig), None)
    if term is None:
        return False
    _reroll_term(equation, term, banned_sigs, max_iter=max_iter,
                 stats_name='break_target_leak')
    return True


def _break_equation_duplication(equation: Equation, shared_sigs, *,
                                preferred_sigs=(), max_iter: int = 2000) -> bool:
    """Break a system-level degeneracy -- ``equation`` encoding the same law
    as another equation of the system (``_equations_share_active_structure``)
    -- by rerolling ONE of its non-target terms whose factor signature belongs
    to ``shared_sigs``, the active term set the two equations share.

    The term matching ``preferred_sigs`` (the other equation's target
    signature, as ``SoEqRightPartSelector`` passes it) goes first, so the
    rerolled equation stops restating the other equation's explained quantity
    before a genuinely shared coupling term is touched. One rerolled term is
    enough to make the two structures differ; whether the re-selected fit
    stays different is for the caller's convergence loop to check. The target
    itself is never rerolled -- the caller re-selects the right part anyway.

    A redraw is accepted once it leaves ``shared_sigs`` and duplicates no
    other term of the equation (``_reroll_term``). Since the target-term rule
    landed this guard is a backstop: every duplicate pair also leaks a target
    (``_carries_target_of``), and ``SoEqRightPartSelector`` repairs that first.

    Returns False, changing nothing, when there is no candidate: the shared
    structure is carried by the target alone (the equation's active set is
    just its target), so no non-target term can be rerolled. The caller's
    postcondition reports such a pair. Otherwise returns True, with the cached
    fitness/weight state reset so the caller can re-run right-part selection.
    """
    candidates = [term for idx, term in enumerate(equation.structure)
                  if idx != getattr(equation, 'target_idx', None)
                  and term.factors_labels in shared_sigs]
    if not candidates:
        return False
    preferred = [t for t in candidates if t.factors_labels in preferred_sigs]
    term = preferred[0] if preferred else candidates[0]
    _reroll_term(equation, term, shared_sigs, max_iter=max_iter,
                 stats_name='break_equation_duplication')
    return True


# ---------------------------------------------------------------------------
# 'assignment' mode (search_space.system_rps_mode): a system's targets and
# supports are chosen together, by enumeration and dominance.
# ---------------------------------------------------------------------------

#: Stack of the population's system-label histories, pushed by the population
#: operators around their right-part selection (:func:`system_history`).
_SYSTEM_HISTORY = []

#: Upper bound on the system variants compared for one system. The product of
#: the per-equation fronts is small in practice (a few subsets per equation, a
#: few target assignments); the cap keeps a pathological library bounded.
MAX_SYSTEM_VARIANTS = 4096


@contextmanager
def system_history(history):
    """Expose the population's history of system labels (``SoEq.equations_labels``
    tuples) to the right-part selector for the duration of the block.

    ``'assignment'`` mode skips the variants already in it before it compares
    the rest. ``'repair'`` mode never reads it, so pushing it is a no-op there.
    """
    _SYSTEM_HISTORY.append(history)
    try:
        yield
    finally:
        _SYSTEM_HISTORY.pop()


def _current_system_history():
    return _SYSTEM_HISTORY[-1] if _SYSTEM_HISTORY else None


#: Open :func:`resample_scope` blocks, pushed by the population operators
#: around the re-selection of the equations they redrew.
_RESAMPLE_SCOPE = []


@contextmanager
def resample_scope():
    """Mark the right-part selection inside the block as a RESAMPLE: the
    population operators redrew the degenerate equations of a system --
    declined by selection or stamped by the fitness host
    (``moeadd_specific.regenerate_degenerate_equations``) -- and the rest of
    it is still fitted.

    In ``'repair'`` mode the forward pass of :class:`SoEqRightPartSelector`
    then selects only the equations not selected yet (``right_part_selected``
    False: the redrawn ones). The others keep their fits instead of being
    selected again on their pruned structure. That re-selection is not
    always a no-op. Measured on 32 placed random equations over
    ``u = sin 2t + 0.3 t`` (1-D, so 'dyad' cuts time slabs alone), it returned 17 fits bit for bit
    and 13 to round-off, and 2 got back the intercept the first selection had
    pruned. It also cost one full term sweep per fitted equation on every
    redraw, about ten times the in-place re-score that follows it. The
    cross-equation rules after the forward pass still run on the whole
    system. ``'assignment'`` and ``'plain'`` modes never read the scope: the
    first chooses a system's targets together, the second already keeps the
    equations still fitted.
    """
    _RESAMPLE_SCOPE.append(True)
    try:
        yield
    finally:
        _RESAMPLE_SCOPE.pop()


#: The MOEA/DD sector whose offspring are being built: ``(weight, normalizer,
#: focus)``, recorded by the mating selection (``set_mating_sector``) and read
#: by ``'assignment'`` mode's ``variant_pick='sector'``. ``focus`` is the equation the sector serves
#: under ``niching='per_equation'``, else None.
_MATING_SECTOR = {'value': None}


def set_mating_sector(weight, normalizer=None, focus=None) -> None:
    """Record the weight direction (and the population's objective normalizer)
    of the sector the next offspring are generated for. ``weight=None`` clears
    it -- done at the start of every initial population, so a fit never
    inherits the previous fit's sector."""
    _MATING_SECTOR['value'] = (None if weight is None else
                               (np.asarray(weight, dtype=float), normalizer, focus))


def _sector_pick(variants, candidates, value_attrs):
    """``variant_pick='sector'``: the candidate closest to the current mating
    sector's direction, by PBI in each equation's own plane (the population's
    normalizer, ideal at the origin, ``evolution.PBI_penalty``). Under
    per-equation niching the served equation decides and the worst of the
    others breaks ties; otherwise the worst equation decides (minimax: no
    values are added across equations). None when no sector is known yet
    (the initial population)."""
    sector = _MATING_SECTOR['value']
    if sector is None:
        return None
    weight, normalizer, focus = sector
    theta = active_config().evolution.PBI_penalty
    direction = weight / max(np.linalg.norm(weight), 1e-300)
    n_eq = len(variants[0])
    n_types = len(value_attrs)

    worst = getattr(normalizer, '_worst_vals', None)
    if normalizer is not None and worst is not None and np.size(worst) != n_eq * n_types:
        return None           # a normalizer of another objective layout: no usable sector

    def plane_pbis(variant):
        full = np.array([getattr(eq, attr) for attr in value_attrs for eq in variant], dtype=float)
        if normalizer is not None:
            full = normalizer(full)
        pbis = []
        for e in range(n_eq):
            point = full[[t * n_eq + e for t in range(n_types)]]
            d1 = float(point @ direction)
            pbis.append(d1 + theta * float(np.linalg.norm(point - d1 * direction)))
        return pbis

    def key(k):
        pbis = plane_pbis(variants[k])
        if focus is not None and focus < n_eq:
            others = [p for e, p in enumerate(pbis) if e != focus]
            return (pbis[focus], max(others) if others else 0.0, k)
        return (max(pbis), sorted(pbis), k)

    return min(candidates, key=key)


def _ensure_own_derivative(equation, max_iter: int = 100) -> bool:
    """Give ``equation`` a term with a derivative of its own variable, the only
    admissible right parts, through the same ``restore_property`` step
    ``EqRightPartSelector`` uses. Bounded; False when it never appeared."""
    var = equation.main_var_to_explain
    attempts = 0
    while not any(term.contains_deriv(var) for term in equation.structure):
        attempts += 1
        if attempts > max_iter:
            return False
        equation.restore_property(mandatory_family=False, deriv=True)
    return True


def _pareto_mask_2d(a, b) -> np.ndarray:
    """Non-dominated mask for minimisation of the pairs ``(a[i], b[i])``.

    Point i is dominated when some j has ``a[j] <= a[i]``, ``b[j] <= b[i]`` and
    one of them strictly: some ``a[j] < a[i]`` with ``b[j] <= b[i]``, or some
    ``a[j] == a[i]`` with ``b[j] < b[i]``. With every value finite that is read
    off one sort by ``a`` (the smallest ``b`` before i's run of equal ``a``, and
    the smallest ``b`` within it) -- comparisons only, so the mask is the
    pairwise definition's exactly. Anything non-finite takes the pairwise loop.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        keep = np.ones(a.size, dtype=bool)
        for i in range(a.size):
            dominated = (a <= a[i]) & (b <= b[i]) & ((a < a[i]) | (b < b[i]))
            keep[i] = not dominated.any()
        return keep
    keep = np.ones(a.size, dtype=bool)
    if a.size == 0:
        return keep
    order = np.argsort(a, kind='stable')
    a_s, b_s = a[order], b[order]
    starts = np.flatnonzero(np.concatenate([[True], a_s[1:] != a_s[:-1]]))
    run_min = np.minimum.reduceat(b_s, starts)
    # The smallest b of every run strictly before each run (inf for the first).
    before = np.concatenate([[np.inf], np.minimum.accumulate(run_min)[:-1]])
    run_of = np.repeat(np.arange(starts.size), np.diff(np.append(starts, a_s.size)))
    dominated = (before[run_of] <= b_s) | (run_min[run_of] < b_s)
    keep[order] = ~dominated
    return keep


def _admissible_front(eligible, a, b, admissible) -> list:
    """The positions of ``eligible`` that are admissible and non-dominated
    among the ADMISSIBLE ones on ``(a, b)``, in ``eligible`` order --
    ``admissible(i)`` asked only of positions that reach a front.

    Inadmissible front members are dropped and the front recomputed until every
    member is admissible. Dominance is a strict partial order, so a position
    dominated by a dropped one is dominated by a non-dominated one too, and the
    result is the front of the admissible positions exactly: a few checks
    instead of one per position (``_subset_rank_deficient`` is a QR over every
    row).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    verdict = {}
    while eligible:
        front = [i for i, keep in zip(eligible, _pareto_mask_2d(a[eligible], b[eligible])) if keep]
        for i in front:
            if i not in verdict:
                verdict[i] = bool(admissible(i))
        refused = {i for i in front if not verdict[i]}
        if not refused:
            return front
        eligible = [i for i in eligible if i not in refused]
    return []


#: Without ``'assignment'`` mode's ``selected_terms_max_number``, above this many
#: columns (features + constant) the exhaustive subset enumeration is refused:
#: 2^12 subsets, each with an instability estimate, per target and per
#: equation.
MAX_ASSIGNMENT_COLUMNS = 12
#: Under the cap: the most subsets of at most that many columns scored per
#: target and equation -- a bound on their number, not on memory (see
#: ``search_config.ASSIGNMENT_MAX_SUBSETS``).
MAX_ASSIGNMENT_SUBSETS = ASSIGNMENT_MAX_SUBSETS
#: Under ``instability_metric='dyad'``: score each subset once per
#: :func:`_apply_assignment` call, from the call's union slab Grams
#: (:meth:`_ScreenCallState.dyad_scores`), instead of once per key.
#: ``'coupled'`` (default) does so for systems of two or more equations, whose
#: target assignments ask for the same subsets again (41% of lorenz's
#: scorings repeat within a call, 1.8x on its calls); a single equation has
#: nothing to reuse. ``True`` / ``False`` force it on / off: the A/B switch of
#: the benchmark harness, not a user setting.
_DYAD_MEMO = 'coupled'


def _use_dyad_memo(n_equations: int) -> bool:
    """Whether an assignment call over ``n_equations`` equations scores its
    dyad subsets through the memo (``_DYAD_MEMO``)."""
    if active_config().objectives.instability_metric != 'dyad':
        return False
    if _DYAD_MEMO == 'coupled':
        return n_equations > 1
    return bool(_DYAD_MEMO)


def _capped_masks(n_cols: int, selected_max: int) -> list:
    """The bitmasks over ``n_cols`` columns with 1 .. ``selected_max`` bits
    set, ascending: with the cap at or above ``n_cols`` exactly
    ``range(1, 1 << n_cols)``, the uncapped search's subsets in its order."""
    top = min(int(selected_max), n_cols)
    return sorted(sum(1 << j for j in combo) for k in range(1, top + 1)
                  for combo in itertools.combinations(range(n_cols), k))


def _factor_powers(term) -> list:
    """The integer power of every factor of ``term`` (None for a non-integral
    one), as ``simplify_equation``'s degree reduction reads them."""
    powers = []
    for factor in term.structure:
        for i in factor.params_description:
            if factor.params_description[i]["name"] == "power":
                p = factor.params[i]
                powers.append(int(p) if float(p) == int(p) and int(p) >= 1 else None)
    return powers


def _subset_reducible(target, feature_terms, cols, with_constant) -> bool:
    """True for a support ``simplify_equation`` would reduce rather than keep:
    every selected term shares a factor with the target (``u*u_t = c*u*u_xx`` is
    ``u_t = c*u_xx`` multiplied through), or a single selected term and the
    target are p-th powers of one equation (``c*u_xx^2 = u_tt^2``).

    The shared factor is looked for in the terms alone, as ``simplify_equation``
    does: a constant beside a law multiplied through is round-off on exact data
    (``u^2 u'' = a u^3 + 6e-17``), and counting it kept such a support. The
    price is a law whose constant is real, ``u_t = a u_xx + c/u``, which only
    its multiplied form expresses. A constant does block the p-th root, which
    it does not pass through: ``u'^2 = c - u^2`` is the energy identity."""
    if not cols:
        return False
    labels = [target.factors_labels_without_power] + \
             [feature_terms[j].factors_labels_without_power for j in cols]
    if frozenset.intersection(*labels):
        return True
    if len(cols) == 1 and not with_constant:
        powers = _factor_powers(target) + _factor_powers(feature_terms[cols[0]])
        if powers and None not in powers and int(np.gcd.reduce(np.array(powers))) >= 2:
            return True
    return False


def _subset_rank_deficient(unit_cols, unit_constant, cols, with_constant) -> bool:
    """True when the support's columns (and the constant) are EXACTLY linearly
    dependent -- the rank-deficient fits ``simplify_equation`` regenerates
    (``_flag_dependent_columns``, same machine-exactness tolerance)."""
    members = [unit_cols[:, j] for j in cols] + ([unit_constant] if with_constant else [])
    if len(members) < 2:
        return False
    r = np.linalg.qr(np.stack(members, axis=1), mode='r')
    return bool(np.min(np.abs(np.diag(r))) < _LIN_DEP_RTOL)


def _support_shares(variant) -> np.ndarray:
    """The share of the target each non-target term of ``variant`` and its
    constant (last) carry: :func:`amplification_ratio`'s per-term measure
    ``|c_j| * ||col_j|| / ||target col||``, the constant's column the
    ones-vector, the largest over the trajectories whose evaluations are
    finite."""
    w = np.asarray(variant.weights_internal, dtype=float)
    features = [term for i, term in enumerate(variant.structure) if i != variant.target_idx]
    slot = slot_of(variant)
    cols = [term_columns(term, slot) for term in features]
    shares = np.zeros(len(features) + 1)
    for key, y in term_columns(variant.target, slot).items():
        y = np.asarray(y, dtype=float).reshape(-1)
        xs = [np.asarray(col[key], dtype=float).reshape(-1) for col in cols]
        norm = np.linalg.norm(y)
        if not (np.isfinite(norm) and norm > 0 and all(np.all(np.isfinite(x)) for x in xs)):
            continue
        sample = [abs(wj) * np.linalg.norm(x) for wj, x in zip(w[:-1], xs)] + [abs(w[-1]) * np.sqrt(y.size)]
        shares = np.maximum(shares, np.asarray(sample) / norm)
    return shares


def _pruned_to_reducible(variant) -> bool:
    """True when the support the fit keeps is one the candidate rules refuse
    (:func:`_subset_reducible`), its round-off terms not counted. On exact data
    a term can get a coefficient of exactly 0 or one at round-off: on a pure
    sinusoid the finite-difference u'' is exactly proportional to u, so
    u^2 u'' = a u^3 + b u'' u' + c fits b and c at 0 or at round-off (1e-15),
    and what is left is u^2 u'' = a u^3, the law multiplied through by u^2. A term
    or the constant is round-off when its share of the target
    (:func:`_support_shares`) is below ``_LIN_DEP_RTOL`` on every trajectory --
    the machine-exactness tolerance of exact linear dependence; real terms sit
    many orders above it. The rules hold for the support that is kept."""
    shares = _support_shares(variant)
    features = [term for i, term in enumerate(variant.structure) if i != variant.target_idx]
    cols = [j for j in range(len(features)) if shares[j] >= _LIN_DEP_RTOL]
    return _subset_reducible(variant.target, features, cols, bool(shares[-1] >= _LIN_DEP_RTOL))


class _Candidate(object):
    """One equation variant before it is built: its subset-table objective
    values (as the fitness host's attributes), its structure label, and the
    build that fits, prunes and scores it. Systems are compared on these
    values; only the variant a system finally takes is built."""

    def __init__(self, values, terms_labels, build):
        for attr, value in values.items():
            setattr(self, attr, value)
        self.terms_labels = terms_labels
        self._build = build
        self._built = False
        self._variant = None

    def materialize(self):
        if not self._built:
            self._variant = self._build()
            self._built = True
        return self._variant


class _ScreenCallState(object):
    """The exact screen's state for ONE :func:`_apply_assignment` call
    (``subset_search._ENGINE == 'exact'``).

    Within a call every equation's structure is fixed, so the screen works on
    one union per equation: the weighted Gram of ALL its terms' columns and
    the constant, per trajectory, built once (:meth:`_union`). A key's
    problem -- its target's cross moments, its feature columns -- is gathered
    from the union (``subset_search.assemble_from_union``) and is bitwise the
    sub-block of any superset's, so the lists of a removed set are derived
    from the deep lists (``2 * screen_size`` per size) of a searched superset
    (``subset_search.derive``) and searched only when that cannot be
    certified. The first search of a target is of its whole feature set (no
    term removed), the superset of every removed set.

    The masks are those of the canonical statistic
    (``subset_search.canonical_values``), not of :func:`screened_masks`'s
    BLAS-built Gram: the two keep different subsets only where RSS values tie
    to round-off at the cut. Columns are ordered by their terms' labels, the
    constant last, ties at the cut broken in that order -- the order
    :func:`_equation_variants` gives the legacy screen.
    """

    def __init__(self, equations, selected_max, screen_size, dyad_memo=False):
        self.equations = list(equations)
        self.M = None if selected_max is None else int(selected_max)
        self.B = None if screen_size is None else int(screen_size)
        self.D = None if screen_size is None else 2 * self.B
        self.dyad_memo = bool(dyad_memo)
        self._unions = {}
        self._raws = {}
        self._targets_checked = {}
        self._deep = defaultdict(list)
        self._dyad_blocks = {}
        self._dyad_scores = {}
        self.stats = dict(keys=0, searches=0, derived=0, fallbacks=0, legacy=0,
                          dyad_scored=0, dyad_reused=0, dyad_direct=0)

    def _union(self, i):
        """Equation ``i``'s terms (copies, so the population's are not
        touched), their label sort keys, and per trajectory the weighted Gram
        of ``[every term | 1]`` with each column's finiteness. A non-finite
        column is zeroed before the product, so it reaches no other entry."""
        if i in self._unions:
            return self._unions[i]
        terms = [deepcopy(term) for term in self.equations[i].structure]
        slot = slot_of(self.equations[i])
        values = [term_columns(term, slot) for term in terms]
        gfuncs = global_var.samples_manager.row_weights(slot, 'dm')
        per_key, columns = {}, {}
        for key in values[0].keys():
            cols = [np.asarray(value[key], dtype=float).reshape(-1) for value in values]
            X = np.column_stack(cols + [np.ones(len(cols[0]))])
            finite = np.all(np.isfinite(X), axis=0)
            sw = np.asarray(gfuncs[key], dtype=float).reshape(-1)
            if not np.all(np.isfinite(sw)) or np.any(sw < 0):
                raise ValueError('sample weights must be finite and non-negative.')
            Xw = X * np.sqrt(sw)[:, None]
            Xw[:, ~finite] = 0.0
            per_key[key] = (Xw.T @ Xw, finite)
            if self.dyad_memo:
                X[:, ~finite] = 0.0
                columns[key] = (X, sw)
        labels = [term.factors_labels for term in terms]
        union = dict(terms=terms, per_key=per_key, keys=list(per_key), columns=columns,
                     sort_keys=[tuple(sorted(repr(lab) for lab in term.factors_labels))
                                for term in terms],
                     duplicate_labels=len(set(labels)) < len(labels))
        self._unions[i] = union
        return union

    def _raw(self, i, traj):
        """The union Gram of equation ``i`` summed over the trajectories
        ``traj``, in their order (one sum per call, shared by every key)."""
        if (i, traj) not in self._raws:
            per_key = self._unions[i]['per_key']
            raw = per_key[traj[0]][0].copy()
            for key in traj[1:]:
                raw = raw + per_key[key][0]
            self._raws[(i, traj)] = raw
        return self._raws[(i, traj)]

    def _canonical(self, union, features):
        """The union columns of ``features`` (term indices) in canonical
        order, the constant last, and each one's bit in the key's masks
        (feature ``c`` of the key is bit ``c``, the constant bit
        ``len(features)``)."""
        order = sorted(range(len(features)),
                       key=lambda c: (union['sort_keys'][features[c]], features[c]))
        cols = [features[c] for c in order] + [len(union['terms'])]
        bits = order + [len(features)]
        return cols, dict(zip(cols, bits))

    def _target_is_its_column(self, i, t, finite):
        """The union's cross moments use the target term's own column; the
        key's target must be that column bit for bit (checked once per target
        and trajectory)."""
        for key, _X, y, _sw in finite:
            if (i, t, key) not in self._targets_checked:
                column = term_columns(self._unions[i]['terms'][t], slot_of(self.equations[i]))[key]
                self._targets_checked[(i, t, key)] = np.array_equal(
                    y, np.asarray(column, dtype=float).reshape(-1))
            if not self._targets_checked[(i, t, key)]:
                return False
        return True

    def _search(self, i, t, traj, cols, store=True):
        """The deep lists of the union columns ``cols`` (canonical order) for
        target ``t`` on ``traj``, stored for derivation when ``store``."""
        from epde.operators.common.subset_search import ExactScreen, assemble_from_union
        raw = self._raw(i, traj)
        A, b, yy, kept = assemble_from_union(raw, raw[:, t], raw[t, t], cols)
        deep = ExactScreen(A, b, yy, self.M, self.D).levels()
        self.stats['searches'] += 1
        if store:
            self._deep[(i, t)].append((tuple(int(c) for c in kept), traj, deep))
            self._deep[(i, t)].sort(key=lambda entry: len(entry[0]))
        return deep

    def _derive(self, i, t, traj, kept):
        from epde.operators.common.subset_search import derive
        members = set(kept)
        index = {c: n for n, c in enumerate(kept)}
        for kept_s, traj_s, deep in self._deep[(i, t)]:
            if traj_s != traj or not members <= set(kept_s):
                continue
            lists = derive(deep, [index.get(c, -1) for c in kept_s], self.M, self.B)
            if lists is not None:
                return lists
        return None

    def masks(self, i, t, features, finite) -> set:
        """The screened bitmasks over the key's columns ``[X | 1]`` (feature
        ``c`` on bit ``c``, the constant on the top bit): the first
        ``screen_size`` supports of each size 1 .. ``selected_max`` by the
        canonical statistic. ``i`` is the equation's position, ``t`` the
        target's term index, ``features`` the key's feature term indices in
        its column order, ``finite`` the key's ``(key, X, y, sw)`` rows."""
        self.stats['keys'] += 1
        union = self._union(i)
        keys = {key for key, *_ in finite}
        traj = tuple(key for key in union['keys'] if key in keys)
        if (len(traj) != len(keys)
                or not all(union['per_key'][key][1][list(features) + [t]].all() for key in traj)
                or not self._target_is_its_column(i, t, finite)):
            # Not the union's problem -- which cannot happen while
            # Equation.evaluate reads its columns from Term.evaluate: the
            # legacy screen on the key's own data.
            self.stats['legacy'] += 1
            return self._legacy(union, features, finite)
        cols, bits = self._canonical(union, features)
        raw = self._raw(i, traj)
        kept = tuple(c for c in cols if raw[c, c] > 0)      # assemble_from_union's rule
        lists = None
        if not union['duplicate_labels']:
            if not self._deep[(i, t)]:
                whole = self._whole_features(union, t)
                if all(union['per_key'][key][1][whole].all() for key in traj):
                    self._search(i, t, traj, self._canonical(union, whole)[0])
            lists = self._derive(i, t, traj, kept)
            if lists is not None:
                self.stats['derived'] += 1
        if lists is None:
            # Searched on the key's own columns: no stored superset certifies
            # its lists, or two terms share labels (no derivation then).
            self.stats['fallbacks'] += 1
            deep = self._search(i, t, traj, cols, store=not union['duplicate_labels'])
            lists = {k: S[:self.B] for k, (_v, S, _whole) in deep.items()}
        return {sum(1 << bits[kept[c]] for c in support)
                for S in lists.values() for support in S.tolist()}

    def dyad_scores(self, i, t, features, key, subsets, grid_shape, y):
        """``survival.dyad_subset_scores`` of a key's subsets on trajectory
        ``key``, from the call's union slab Grams and memoised over the call
        (``_DYAD_MEMO``), or None when the key is not the union's problem.

        ``subsets`` index the key's ``[X | 1]`` (feature ``c`` -> column ``c``,
        the constant -> ``len(features)``). A subset's score depends on its
        terms, the target and the trajectory only -- the slab partitions are
        the search's (``survival.dyad_t_min_rows``) -- so the subsets of ONE
        union slab Gram per (equation, trajectory, partition, grid axis) are
        sub-blocks of it, bit-identical whichever key asks, and a subset
        scored for one removed set is reused for every other. Each axis is cut
        as storage axis 0 (``survival._axis_layouts``) and the per-term scores
        are combined over the axes as ``survival.dyad_scores`` does. The
        scores differ from the per-key computation only at round-off (the slab
        Grams are formed over the union's columns, the target as one of
        them)."""
        from epde.operators.common import survival
        union = self._union(i)
        if key not in union['columns'] or not self._target_is_its_column(
                i, t, [(key, None, y, None)]) or not union['per_key'][key][1][t]:
            return None
        n_terms = len(union['terms'])
        cols = [tuple(features[j] if j < len(features) else n_terms for j in S) for S in subsets]
        out = np.empty(len(subsets))
        missing = defaultdict(list)
        for n, c in enumerate(cols):
            score = self._dyad_scores.get((i, t, key, c))
            if score is None:
                missing[len(c)].append(n)
            else:
                out[n] = score
                self.stats['dyad_reused'] += 1
        for size, idxs in sorted(missing.items()):
            if size == 0:
                out[idxs] = np.inf
                continue
            min_rows = survival.dyad_t_min_rows(size)
            X, sw = union['columns'][key]
            layouts = survival._axis_layouts(X.shape[0], grid_shape)
            per_axis = []
            for axis, order, shape in layouts:
                if (i, key, min_rows, axis) not in self._dyad_blocks:
                    Xa = X if order is None else X[order]
                    swa = sw if order is None or sw is None else sw[order]
                    parts, G_all, _ = survival.dyad_t_blocks(Xa, np.zeros(Xa.shape[0]), swa, shape,
                                                             Xa.shape[0], min_rows)
                    # Under the design solves (survival._DESIGN_SOLVES) the slab designs too, the target one of their
                    # columns: factored per slab when a solve first reads it, and kept for every later target and
                    # subset.
                    R = (survival._SlabFactors(Xa, swa, parts)
                         if survival._DYAD_SOLVE in survival._DESIGN_SOLVES else None)
                    self._dyad_blocks[(i, key, min_rows, axis)] = (parts, G_all, R)
                parts, G_all, R = self._dyad_blocks[(i, key, min_rows, axis)]
                per_axis.append(survival.dyad_t_scores_from_blocks(parts, G_all, G_all[:, :, t],
                                                                   [list(cols[n]) for n in idxs],
                                                                   None if R is None else (R, t),
                                                                   per_term=len(layouts) > 1))
            if len(layouts) == 1:
                scores = per_axis[0]
            else:
                with np.errstate(over='ignore'):
                    scores = np.sum(np.nan_to_num(survival._combine_axes(np.stack(per_axis))), axis=1)
            for n, score in zip(idxs, scores):
                out[n] = score
                self._dyad_scores[(i, t, key, cols[n])] = float(score)
            self.stats['dyad_scored'] += len(idxs)
        return out

    def _whole_features(self, union, t):
        """Target ``t``'s features with no term removed: every other term
        not related to it (:func:`_related_to_target`), as in
        :func:`_equation_variants` with an empty removed set."""
        terms = union['terms']
        return [j for j, term in enumerate(terms)
                if j != t and not _related_to_target(term, terms[t])]

    def _legacy(self, union, features, finite):
        from epde.operators.common.subset_search import screened_masks
        order = sorted(range(len(features)), key=lambda c: (union['sort_keys'][features[c]], c))
        return screened_masks(np.vstack([X for _key, X, _y, _sw in finite]),
                              np.concatenate([y for _key, _X, y, _sw in finite]),
                              np.concatenate([sw for _key, _X, _y, sw in finite]),
                              self.M, self.B, order)


def _equation_variants(equation, target_term, removed_sigs, coeff_calc, coeff_args,
                       host, host_args, screen_state=None, equation_index=None) -> list:
    """The fitted, non-dominated variants of ONE equation for one target and
    one set of removed terms.

    A deep copy of ``equation`` loses every term whose signature is in
    ``removed_sigs`` (other equations' chosen targets), takes ``target_term``
    as its right part, and has EVERY subset of its remaining terms plus the
    constant fitted (``subset_selection.subset_table`` /
    ``instability_table``: the Discrepancy and Instability objectives' own
    formulas, so the subset search and the Pareto axes cannot disagree).
    Subsets whose fit only reaches the target through amplification
    (``search_space.rps_amplification_cap``) are not candidates, as in
    ``EqRightPartSelector``. The subsets non-dominated on (discrepancy,
    instability) are each materialised as their own fitted, pruned and scored
    copy. No threshold is involved anywhere. Under
    the mode's ``selected_terms_max_number`` (``search_space.system_rps_params``)
    the subsets are those of at most that many columns (the constant counting
    as one) -- every one of them, scored the same way -- and under its ``screen_size``
    only the best of each size by weighted least squares on the trajectories
    pooled (``subset_search``): the one place a subset reaches the Pareto axes
    through another statistic first, and the subsets the rules below refuse
    keep their screen slots.

    ``simplify_equation``'s guards, as deterministic exclusions (nothing is
    rerolled, so the same structure always yields the same variants):
    features related to the target by DIVISION (``_related_to_target``: the
    wave ``u*u_tt`` soak family and its multiples) are removed like another
    equation's target; supports that share a factor with the target or are
    p-th powers of one equation (:func:`_subset_reducible`), and exactly
    rank-deficient supports (:func:`_subset_rank_deficient`), are not
    candidates. The reducibility rule also holds for the support a variant
    keeps once the fit's exact zeros are pruned, its round-off terms not
    counted (:func:`_pruned_to_reducible`): such a variant is not built.

    ``screen_state`` (:class:`_ScreenCallState`, one per
    :func:`_apply_assignment` call) and ``equation_index`` (the equation's
    position in the system) route the screen to the exact engine; without
    them, or under ``subset_search._ENGINE = 'legacy'``, the key is screened
    by :func:`subset_search.screened_masks` on its own data.
    """
    from epde.operators.common.subset_selection import (
        instability_table, mask_of, store_fit, subset_coefficients, subset_table,
        support_scores, trajectory_gram)
    from epde.operators.common import subset_search
    from epde.operators.common.subset_search import screened_masks
    from epde.operators.common.survival import dyad_subset_scores
    from epde.operators.common.objectives import LOSS_NAN_VAL

    position = next(i for i, term in enumerate(equation.structure) if term is target_term)
    work = deepcopy(equation)
    work_target = work.structure[position]
    work.reset_state(True)
    # ``kept`` holds the surviving terms' indices in ``equation.structure``
    # (the exact screen's union columns are keyed by them).
    kept = [j for j, term in enumerate(work.structure)
            if term is work_target
            or (term.factors_labels not in removed_sigs
                and not _related_to_target(term, work_target))]
    work.structure = [work.structure[j] for j in kept]
    work._invalidate_label_cache()
    if len(work.structure) < 2:
        return []
    work.target_idx = next(i for i, term in enumerate(work.structure) if term is work_target)

    cfg = active_config()
    metric = cfg.objectives.instability_metric
    gram_mode = cfg.objectives.gram_mode
    discrepancy = cfg.objectives.discrepancy_metric
    amp_cap = cfg.search_space.rps_amplification_cap
    selected_max = rps_param(cfg.search_space, 'selected_terms_max_number')
    screen_size = rps_param(cfg.search_space, 'screen_size')
    gfuncs = global_var.samples_manager.row_weights(slot_of(work), 'dm')
    shapes = global_var.samples_manager.row_shapes(slot_of(work))

    target, features = work.evaluate()
    if features is None:
        return []
    finite = []
    for key in target.keys():
        X = np.asarray(features[key], dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(target[key], dtype=float).reshape(-1)
        sw = np.asarray(gfuncs[key], dtype=float).reshape(-1)
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            continue
        finite.append((key, X, y, sw))
    if not finite:
        return []
    n_cols = finite[0][1].shape[1] + 1
    # None: every subset, the tables indexed by bitmask. Under the cap, the
    # ascending bitmasks of at most ``selected_max`` columns -- every one, or
    # under the screen the best of each size by weighted least squares on the
    # trajectories pooled -- and the tables aligned with them: one list for
    # every trajectory, so their tables average position by position.
    if selected_max is None:
        if n_cols > MAX_ASSIGNMENT_COLUMNS:
            raise ValueError(
                "system_rps_mode='assignment' enumerates every subset of an equation's "
                f'terms; an equation with {n_cols - 1} non-target terms exceeds the '
                f'{MAX_ASSIGNMENT_COLUMNS}-column limit. Lower equation_terms_max_number.')
        masks = None
    elif screen_size is None:
        count = assignment_subset_count(n_cols, selected_max)
        if count > MAX_ASSIGNMENT_SUBSETS:
            raise ValueError(
                "system_rps_mode='assignment' scores every subset of at most "
                f"{selected_max} of an equation's {n_cols} columns: {count} "
                f'subsets, more than the {MAX_ASSIGNMENT_SUBSETS}-subset limit. Lower '
                "equation_terms_max_number or system_rps_params['selected_terms_max_number'], "
                "or set system_rps_params['screen_size'].")
        masks = _capped_masks(n_cols, selected_max)
    else:
        # Ties at the screen's cut are broken in the order of the terms'
        # labels, so the same structure screens the same subsets whatever the
        # order its terms are held in.
        terms = [term for i, term in enumerate(work.structure) if i != work.target_idx]
        if screen_state is not None and subset_search._ENGINE == 'exact':
            # One ranking for the trajectories pooled, from the call's union
            # Gram, derived from a searched superset where it can be certified.
            masks = sorted(screen_state.masks(equation_index, position,
                                              [j for j in kept if j != position], finite))
        else:
            order = sorted(range(n_cols - 1),
                           key=lambda j: tuple(sorted(repr(lab) for lab in terms[j].factors_labels)))
            # One ranking for the trajectories pooled, so the screen keeps
            # ``screen_size`` of each size however many trajectories there are.
            masks = sorted(screened_masks(np.vstack([X for _key, X, _y, _sw in finite]),
                                          np.concatenate([y for _key, _X, y, _sw in finite]),
                                          np.concatenate([sw for _key, _X, _y, sw in finite]),
                                          selected_max, screen_size, order))
        if len(masks) > MAX_ASSIGNMENT_SUBSETS:
            raise ValueError(
                "system_rps_mode='assignment' screens {0} subsets of at most {1} of an "
                "equation's {2} columns, more than the {3}-subset limit. Lower "
                "system_rps_params['screen_size'].".format(len(masks), selected_max, n_cols,
                                               MAX_ASSIGNMENT_SUBSETS))
    samples = []
    for key, X, y, sw in finite:
        gram = trajectory_gram(None, X, y, sw, shapes[key], gram_mode,
                               work.main_var_to_explain, key)
        disc_tab, amp_tab, n_cols = subset_table(X, y, sw, discrepancy=discrepancy,
                                                 subsets=masks)
        if metric == 'dyad':
            # The same per-subset statistic as instability_table's, from slab
            # Grams built once for all the columns (survival.dyad_subset_scores),
            # or under _DYAD_MEMO from the call's union slab Grams, each subset
            # scored once per call (_ScreenCallState.dyad_scores).
            Xa = np.hstack([X, np.ones((X.shape[0], 1))])
            if masks is None:
                subsets = [[j for j in range(n_cols) if (S >> j) & 1]
                           for S in range(1, 1 << n_cols)]
            else:
                subsets = [[j for j in range(n_cols) if (S >> j) & 1] for S in masks]
            scores = None
            if screen_state is not None and screen_state.dyad_memo:
                scores = screen_state.dyad_scores(equation_index, position,
                                                  [j for j in kept if j != position],
                                                  key, subsets, shapes[key], y)
            if scores is None:
                if screen_state is not None and screen_state.dyad_memo:
                    screen_state.stats['dyad_direct'] += 1
                scores = dyad_subset_scores(Xa, y, sw, shapes[key], subsets)
            inst_tab = np.concatenate([[np.inf], scores]) if masks is None else scores
        else:
            inst_tab = instability_table(metric, X, y, sw, shapes[key], n_cols,
                                         gram_setup=gram, subsets=masks)
        samples.append((key, X, y, sw, gram, disc_tab, amp_tab, inst_tab))
    if not samples:
        return []
    n_cols = samples[0][1].shape[1] + 1
    disc = np.mean([s[5] for s in samples], axis=0)
    amp = np.mean([s[6] for s in samples], axis=0)
    inst = np.mean([s[7] for s in samples], axis=0)
    p = n_cols - 1
    feature_terms = [term for i, term in enumerate(work.structure) if i != work.target_idx]
    X0 = samples[0][1]
    norms = np.linalg.norm(X0, axis=0)
    unit_cols = X0 / np.where(norms > 0, norms, 1.0)
    unit_constant = np.full(X0.shape[0], 1.0 / np.sqrt(X0.shape[0]))

    def admissible(S):
        cols = [j for j in range(p) if (S >> j) & 1]
        with_constant = bool((S >> p) & 1)
        if any(norms[j] == 0 for j in cols):
            return False
        return not (_subset_reducible(work_target, feature_terms, cols, with_constant)
                    or _subset_rank_deficient(unit_cols, unit_constant, cols, with_constant))

    # Position i of the tables holds the subset with bitmask ``mask_at(i)``.
    if masks is None:
        positions, mask_at = range(1, 1 << n_cols), (lambda i: i)
    else:
        positions, mask_at = range(len(masks)), masks.__getitem__
    eligible = [i for i in positions
                if np.isfinite(disc[i]) and np.isfinite(inst[i])
                and (amp_cap is None or amp[i] <= amp_cap)]
    front = _admissible_front(eligible, disc, inst, lambda i: admissible(mask_at(i)))
    if not front:
        return []
    # The denominator counts the empty subset either way, as the dense table does.
    _loop_stats.record('SoEqRPS.assign.equation_front', len(front),
                       1 << n_cols if masks is None else len(masks) + 1)

    def build(S):
        """The fitted, pruned and scored variant of support ``S``; None when the
        fitness host declines it (LOSS_NAN_VAL), or when the support it keeps,
        its exact zeros pruned and its round-off terms not counted, is one
        :func:`admissible` refuses (:func:`_pruned_to_reducible`)."""
        support = mask_of(S, n_cols)
        coefs, sw_weights, vc_scores = [], {}, {}
        for key, X, y, sw, gram, *_ in samples:
            full = subset_coefficients(X, y, sw, support)
            coefs.append(full)
            vc_scores[key], sw_weights[key] = support_scores(gram, full != 0, metric)
        variant = deepcopy(work)
        store_fit(variant, np.mean(np.stack(coefs, axis=1), axis=1),
                  sw_weights, vc_scores)
        coeff_calc.apply(variant, coeff_args)
        variant.remove_zero_terms()
        if _pruned_to_reducible(variant):
            _loop_stats.record('SoEqRPS.assign.pruned_reducible', 1, 1)
            return None
        host.apply(variant, host_args)
        if variant.fitness_value is None or variant.fitness_value >= LOSS_NAN_VAL:
            return None
        variant.right_part_selected = True
        return variant

    disc_attr, inst_attr = (filler.value_attr for filler in host.objectives[:2])
    candidates = []
    for i in front:
        S = mask_at(i)
        cols = [j for j in range(p) if (S >> j) & 1]
        labels = frozenset([work_target.factors_labels]
                           + [feature_terms[j].factors_labels for j in cols])
        candidates.append(_Candidate({disc_attr: float(disc[i]), inst_attr: float(inst[i])},
                                     labels, partial(build, S)))
    return candidates


def _nondominated_systems(variants, value_attrs) -> list:
    """Indices of the system variants no other variant dominates, by the rule
    of ``check_dominance``: the objectives of an equation whose structure
    (``terms_labels``) is the same in both systems are not compared -- the
    same structure is the same equation."""
    m, n = len(variants), len(variants[0])
    vals = np.array([[[getattr(eq, attr) for attr in value_attrs] for eq in variant]
                     for variant in variants], dtype=float)          # (m, n, types)
    ids = np.zeros((m, n), dtype=int)
    for e in range(n):
        lookup = {}
        for k, variant in enumerate(variants):
            ids[k, e] = lookup.setdefault(variant[e].terms_labels, len(lookup))
    keep = []
    for a in range(m):
        compared = (ids != ids[a])[:, :, None]                      # (m, n, 1)
        no_worse = np.all((vals <= vals[a]) | ~compared, axis=(1, 2))
        better = np.any((vals < vals[a]) & compared, axis=(1, 2))
        if not np.any(no_worse & better):
            keep.append(a)
    return keep


def _mark_unassignable(equations, host) -> None:
    """Leave ``equations`` fitted-but-degenerate: zero weights and every
    objective at ``LOSS_NAN_VAL``, the fitness host's own degeneracy stamp.
    The population operators then re-roll them (``regenerate_degenerate_
    equations``) or drop the system, exactly as for a declined fit."""
    from epde.operators.common.objectives import LOSS_NAN_VAL
    for equation in equations:
        var = equation.main_var_to_explain
        equation.reset_state(True)
        candidates = [i for i, term in enumerate(equation.structure) if term.contains_deriv(var)]
        equation.target_idx = candidates[0] if candidates else 0
        equation.weights_internal = np.zeros(len(equation.structure))
        equation.weights_internal_evald = True
        equation.weights_final = np.zeros(len(equation.structure))
        equation.weights_final_evald = True
        for filler in host.objectives:
            setattr(equation, filler.value_attr, LOSS_NAN_VAL)
            setattr(equation, filler.flag_attr, True)
        equation.fitness_value = LOSS_NAN_VAL
        equation.fitness_calculated = True
        equation.aic = None
        equation.aic_calculated = True
        equation.right_part_selected = True


def _apply_assignment(objective, eq_selector, eq_args) -> None:
    """``system_rps_mode='assignment'``: choose the system's targets and
    supports together.

    1. Every equation needs an admissible target, a term with a derivative of
       its own variable (restored if missing). Otherwise the system is not
       calculated: it is marked degenerate for the population operators.
    2. Every assignment of one such term per equation is enumerated (equal
       composite targets excluded). In each equation, the terms equal to
       another equation's chosen target are removed, so no equation carries
       another's target and no two equations can state one law.
    3. Per equation, the non-dominated subsets of its remaining terms
       (:func:`_equation_variants`, cached per target and removed set); the
       system variants of an assignment are their combinations.
    4. Variants already in the search history (:func:`system_history`) are
       skipped; the rest are compared by dominance (:func:`_nondominated_systems`)
       and one non-dominated variant is installed -- uniformly at random when
       there are several (the mode's ``variant_pick='random'``). When every
       variant is already in history they are all compared, and the caller's
       duplicate check decides.
    """
    _, eq_subargs = eq_selector.parse_suboperator_args(arguments=eq_args)
    coeff_calc = eq_selector.suboperators['coeff_calc']
    host = eq_selector.suboperators['fitness_calculation']
    coeff_args = eq_subargs['coeff_calc']
    host_args = eq_subargs['fitness_calculation']

    keys = list(objective.vals.equation_keys)
    equations = list(objective)
    if not all(_ensure_own_derivative(eq) for eq in equations):
        _loop_stats.record('SoEqRPS.assign.no_target', 1, 1)
        _mark_unassignable(equations, host)
        return

    candidates = [[term for term in eq.structure if term.contains_deriv(eq.main_var_to_explain)]
                  for eq in equations]
    space = active_config().search_space
    dyad_memo = _use_dyad_memo(len(equations))
    screen_size = rps_param(space, 'screen_size')
    screen_state = (_ScreenCallState(equations, rps_param(space, 'selected_terms_max_number'),
                                     screen_size, dyad_memo=dyad_memo)
                    if screen_size is not None or dyad_memo else None)
    cache = {}
    assignments = []          # per valid assignment: the per-equation variant lists
    for combo in itertools.product(*candidates):
        sigs = [term.factors_labels for term in combo]
        if len(set(sigs)) < len(sigs):
            continue
        per_equation = []
        for i, (equation, target_term) in enumerate(zip(equations, combo)):
            others = {sig for j, sig in enumerate(sigs) if j != i}
            removed = frozenset(term.factors_labels for term in equation.structure
                                if term is not target_term and term.factors_labels in others)
            cache_key = (i, sigs[i], removed)
            if cache_key not in cache:
                cache[cache_key] = _equation_variants(equation, target_term, removed,
                                                      coeff_calc, coeff_args, host, host_args,
                                                      screen_state=screen_state, equation_index=i)
            per_equation.append(cache[cache_key])
        if all(per_equation):
            assignments.append(per_equation)
    if screen_state is not None and screen_state.stats['keys']:
        # The exact screen's work per call: searches run (supersets included)
        # out of the keys screened, and keys answered from a searched superset.
        _loop_stats.record('SoEqRPS.assign.screen_searches', screen_state.stats['searches'],
                           screen_state.stats['keys'])
        _loop_stats.record('SoEqRPS.assign.screen_derived', screen_state.stats['derived'],
                           screen_state.stats['keys'])

    if not assignments:
        # Not calculated. Only the equations with no fitted variant under any
        # assignment are marked degenerate, so the population operators re-roll
        # just those (regenerate_degenerate_equations); each other equation gets
        # one of its fitted variants, so the system stays scoreable meanwhile.
        # When EVERY equation has some variant, but only under different
        # assignments, mixing them could break the target rule: all are marked.
        _loop_stats.record('SoEqRPS.assign.no_variant', 1, 1)
        fitted = [next((v for (j, *_), cands in cache.items() if j == i
                        for v in (c.materialize() for c in cands) if v is not None), None)
                  for i in range(len(equations))]
        if all(v is not None for v in fitted):
            _mark_unassignable(equations, host)
            return
        for key, equation, variant in zip(keys, equations, fitted):
            if variant is None:
                _mark_unassignable([equation], host)
            else:
                variant.metaparameters = equation.metaparameters
                objective.vals.replace_gene(gene_key=key, value=variant)
        return

    sizes = [int(np.prod([len(options) for options in per_equation]))
             for per_equation in assignments]
    if sum(sizes) <= MAX_SYSTEM_VARIANTS:
        variants = [variant for per_equation in assignments
                    for variant in itertools.product(*per_equation)]
    else:
        # Too many to compare: a uniform sample over ALL target assignments,
        # not the first ones in enumeration order.
        _loop_stats.record('SoEqRPS.assign.sampled', 1, 1)
        weights = np.asarray(sizes, dtype=float) / float(sum(sizes))
        drawn, variants = set(), []
        for _ in range(4 * MAX_SYSTEM_VARIANTS):
            if len(variants) >= MAX_SYSTEM_VARIANTS:
                break
            a = int(np.random.choice(len(assignments), p=weights))
            picks = tuple(int(np.random.randint(len(options))) for options in assignments[a])
            if (a, picks) in drawn:
                continue
            drawn.add((a, picks))
            variants.append(tuple(options[k] for options, k in zip(assignments[a], picks)))

    history = _current_system_history()
    labels = [tuple(eq.terms_labels for eq in variant) for variant in variants]
    fresh = [k for k, label in enumerate(labels) if history is None or label not in history]
    _loop_stats.record('SoEqRPS.assign.variants', len(variants), MAX_SYSTEM_VARIANTS)
    _loop_stats.record('SoEqRPS.assign.fresh', len(fresh), max(len(variants), 1))
    pool = fresh if fresh else list(range(len(variants)))
    value_attrs = [filler.value_attr for filler in host.objectives]
    nondominated = [pool[k] for k in _nondominated_systems([variants[k] for k in pool],
                                                           value_attrs)]
    _loop_stats.record('SoEqRPS.assign.nondominated', len(nondominated), len(pool))
    # Systems are compared on the subset-table values; only the picked variant
    # is built (fitted, pruned, scored by the host). Should the host decline
    # one of its equations, or one restate the others, the pick moves on to
    # another non-dominated variant.
    from epde.operators.common.system_redundancy import redundant_equations
    remaining = list(nondominated)
    while remaining:
        if len(remaining) == 1:
            pick = remaining[0]
        else:
            pick = None
            if rps_param(active_config().search_space, 'variant_pick') == 'sector':
                pick = _sector_pick(variants, remaining, value_attrs)
            if pick is None:
                pick = remaining[np.random.randint(len(remaining))]
        built = [candidate.materialize() for candidate in variants[pick]]
        if all(variant is not None for variant in built) and redundant_equations(built):
            # One equation restates the others (system_redundancy): declined like a host refusal.
            _loop_stats.record('SoEqRPS.assign.redundant_pick', 1, 1)
            remaining.remove(pick)
            continue
        if all(variant is not None for variant in built):
            # Built for this call only (and carrying the sparsity caches the
            # Instability filler reads), so they are installed as they are. The
            # system's equations share its metaparameter dict (SoEq.create); the
            # installed variant keeps the replaced gene's reference.
            for key, equation, variant in zip(keys, equations, built):
                variant.metaparameters = equation.metaparameters
                objective.vals.replace_gene(gene_key=key, value=variant)
            return
        _loop_stats.record('SoEqRPS.assign.declined_pick', 1, 1)
        remaining.remove(pick)
    _mark_unassignable(equations, host)


class SoEqRightPartSelector(CompoundOperator):
    """Chromosome-level RPS that enforces the system's cross-equation rules.

    Target-term rule: no equation may carry another equation's TARGET as a
    whole, standalone term of its active structure (``_carries_target_of``).
    ``du/dx0 = a*u + b*dv/dx0`` beside ``dv/dx0 = ...`` restates the
    v-equation inside the u-equation; such a form is a valid identity, but it
    crowds the true law off the front (LV loses both equations to it).
    Whole-term equality is deliberate. The target as a FACTOR of a product is
    coupling physics and stays legal: continuity's ``v_y`` in ``v*v_y``, the
    double pendulum's ``theta2''`` in ``cos(D)*theta2''``. A factor ban
    makes the double pendulum unreachable at two factors per term. Two
    equations explaining one composite target are flagged too.

    Duplicate-equation guard: no two equations may share an IDENTICAL active
    structure (``_equations_share_active_structure``) -- one law solved for
    two quantities leaves fewer independent equations than variables. Equal
    active sets always carry each other's targets, so the target-term rule
    catches every duplicate first; the guard stays as a backstop.

    A strict superset of another equation's active set always holds that
    equation's target, so the target-term rule repairs it too. Nothing else is
    constrained across equations: a shared non-target coupling term, another
    equation's non-target derivative and another equation's target as a
    factor of a product are all legal. A target without a derivative of its
    own variable reserves nothing, and a declined equation
    (:func:`_rps_declined`) takes no part in either rule.

    Mechanics: a plain per-equation forward pass first, then a bounded
    convergence loop. Each pass visits every pair ``j < k``. For each
    direction, later equation first, an equation carrying the other's target
    has that term rerolled (``_break_target_leak``; the redraw avoids every
    other equation's target) and its right part re-selected. Then a pair that
    still shares its active structure has one shared non-target term of the
    LATER equation rerolled (``_break_equation_duplication``). The loop exits
    at the first pass with no repair (fixed point) or when its pass budget is
    spent; a postcondition then reports any leak or duplicate still left.
    """
    key = 'SoEqRightPartSelector'

    @_loop_stats.timed('SoEqRPS.apply')
    def apply(self, objective, arguments: dict):
        """Run per-equation RPS, then resolve system degeneracies in-place.

        An equation ``EqRightPartSelector.apply`` declines leaves stamped
        fit-degenerate; this method forwards nothing for it -- the population
        operators read the stamp and redraw it, and select the redraw inside
        :func:`resample_scope`, where the forward pass leaves the equations
        that are still selected as they are.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)
        eq_selector = self.suboperators['eq_right_part_selector']
        eq_args = subop_args.get('eq_right_part_selector', arguments)

        if active_config().search_space.system_rps_mode == 'assignment':
            _apply_assignment(objective, eq_selector, eq_args)
            return
        if active_config().search_space.system_rps_mode == 'plain':
            from epde.operators.common.plain_fit import apply_plain
            apply_plain(objective, eq_selector, eq_args)
            return

        equations = list(objective)

        # Forward pass: plain per-equation right-part selection. The
        # cross-equation rules are enforced after it, by the loop below.
        # Inside a resample (``resample_scope``) only the redrawn equations
        # are selected; every other equation keeps the fit it has.
        resample = bool(_RESAMPLE_SCOPE)
        for equation in equations:
            if resample and equation.right_part_selected:
                _loop_stats.record('SoEqRPS.resample_kept_fit', 1, 1)
                continue
            eq_selector.apply(objective=equation, arguments=eq_args)

        # Resolution: break target leaks and identical active structures until
        # a full pass makes no repair or the pass budget is exhausted.
        max_passes = 50
        passes_used = 0
        leaks_seen = False
        duplicates_seen = False
        for _ in range(max_passes):
            passes_used += 1
            any_changes = False
            # Whether THIS pass's scan still found a leak / a duplicate,
            # repaired or not -- the postcondition below reads the last pass's
            # values.
            leaks_seen = False
            duplicates_seen = False
            # Every unordered pair is visited once. The leak predicate is
            # directional, so both directions are checked, the later
            # equation's first; the duplicate predicate is symmetric, and the
            # LATER equation of a duplicate pair is the one repaired: the
            # earlier equations keep the fits the forward pass gave them.
            for k in range(1, len(equations)):
                for j in range(k):
                    for owner, carrier in ((equations[j], equations[k]),
                                           (equations[k], equations[j])):
                        if not _carries_target_of(owner, carrier):
                            continue
                        leaks_seen = True
                        banned = {_reserved_target_signature(eq) for eq in equations
                                  if eq is not carrier} - {None}
                        if not _break_target_leak(carrier,
                                                  _reserved_target_signature(owner),
                                                  banned):
                            continue
                        carrier.right_part_selected = False
                        eq_selector.apply(objective=carrier, arguments=eq_args)
                        _loop_stats.record('SoEqRPS.target_leak_repair', 1, 1)
                        any_changes = True

                    eq_j, eq_k = equations[j], equations[k]
                    if not _equations_share_active_structure(eq_j, eq_k):
                        continue
                    duplicates_seen = True
                    preferred = ({eq_j.target.factors_labels}
                                 if eq_j.target is not None else set())
                    if not _break_equation_duplication(
                            eq_k, eq_j.active_terms_labels,
                            preferred_sigs=preferred):
                        continue
                    # ``simplified`` / ``is_correct_right_part`` are locals of
                    # ``EqRightPartSelector.apply`` now, so re-invoking it is
                    # the whole mechanism -- it re-enters its loop from scratch.
                    eq_k.right_part_selected = False
                    eq_selector.apply(objective=eq_k, arguments=eq_args)
                    _loop_stats.record('SoEqRPS.duplicate_equation_repair', 1, 1)
                    any_changes = True

            if not any_changes:
                break
        _loop_stats.record('SoEqRPS.degeneracy_passes', passes_used, max_passes)

        # POSTCONDITION. The loop exits either at a fixed point or with its
        # pass budget spent, and both can leave a violation standing: the
        # budget can run out (a re-selection can put a leak straight back),
        # and a pass that finds a duplicate it cannot repair makes no change
        # either -- ``_break_equation_duplication`` has no candidate when the
        # shared structure is the target alone -- so the loop reads that as
        # its fixed point and stops. The system would then leave RPS breaking
        # a rule, silently.
        #
        # Rescan only when the LAST pass still saw a violation: a pass that
        # found none has already proved the invariants, so the normal exit adds
        # no scans at all. Either flag triggers BOTH rescans: with three or
        # more equations, a leak repair late in a pass can re-select an
        # equation into a duplicate of one a pair checked earlier in the same
        # pass, which that pass's duplicate scan never sees. Diagnostic, not fatal -- a leaked target or a
        # duplicate law is a quality defect (unlike a duplicate term, which
        # crashes the next operator), and aborting a whole evolutionary run
        # over one offspring would be worse. It reaches a user who asked for
        # search warnings (``init_verbose(show_warnings=True)``) and is always
        # measurable via ``EPDE_LOOP_STATS``.
        if leaks_seen or duplicates_seen:
            leaks = [f'{carrier.main_var_to_explain} carries the target of '
                     f'{owner.main_var_to_explain}'
                     for owner in equations for carrier in equations
                     if owner is not carrier and _carries_target_of(owner, carrier)]
            if leaks:
                _loop_stats.record('SoEqRPS.target_leaks_at_exit',
                                   len(leaks), max_passes)
                warnings.warn(
                    'SoEqRightPartSelector.apply: target-term rule not '
                    f'converged after {passes_used} of {max_passes} passes; '
                    f'{len(leaks)} leak(s) left: ' + '; '.join(leaks[:4]))
        if leaks_seen or duplicates_seen:
            duplicates = [f'{equations[j].main_var_to_explain} and '
                          f'{equations[k].main_var_to_explain}'
                          for k in range(1, len(equations)) for j in range(k)
                          if _equations_share_active_structure(equations[j],
                                                               equations[k])]
            if duplicates:
                _loop_stats.record('SoEqRPS.duplicate_equations_at_exit',
                                   len(duplicates), max_passes)
                warnings.warn(
                    'SoEqRightPartSelector.apply: duplicate-equation guard not '
                    f'converged after {passes_used} of {max_passes} passes; '
                    f'{len(duplicates)} pair(s) still encode one law: '
                    + '; '.join(duplicates[:4]))

    def use_default_tags(self):
        self._tags = {'right part selection', 'chromosome level',
                      'contains suboperators', 'inplace'}
