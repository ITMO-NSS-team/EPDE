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
from epde.evaluators import simple_function_evaluator
from epde.operators.utils.template import CompoundOperator
from epde.decorators import HistoryExtender
from epde.structure.main_structures import Term, Equation
from epde.supplementary import filter_powers
from epde.operators.common.stability import (GramSetup, VaryingCoefSetup)
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
            if global_var.gram_mode == 'vcoef':
                # The only mode that needs its own Gram machinery.
                objective._gram_super = VaryingCoefSetup.precompute_super(
                    Z, sample_weights, grid_shape,
                    main_var=objective.main_var_to_explain)
            elif global_var.gram_mode == 'axis':
                objective._gram_super = GramSetup.precompute_super(
                    Z, sample_weights, grid_shape)
            else:
                # 'chi2' (the default): no windowed super-Gram -- the chi2
                # CD scores work on raw columns. Keep the shared Z so
                # Tier-3 slicing still skips objective.evaluate per
                # candidate target.
                objective._gram_super = {'Z': Z, 'mode': global_var.gram_mode}
            _loop_stats.record('EqRPS.gram_super_built', 1, 1)
        except Exception:
            # Defensive: any unexpected failure (shape mismatch, missing
            # cache) means we silently fall back -- numerics are
            # preserved, only the speedup is lost.
            objective._gram_super = None
            _loop_stats.record('EqRPS.gram_super_skip', 1, 1)

    @_loop_stats.timed('EqRPS.apply')
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
                    # A None fitness means the candidate was DECLINED inside
                    # the host: sparsity zeroed every feature, the fit is
                    # discrepancy-degenerate, or it is an amplified-identity
                    # parasite (guard ratio above rps_amplification_cap --
                    # huge mutually-cancelling coefficients, e.g. the LV
                    # ``Lambda*(du+dv-alpha*u+gamma*v) = d^2u`` form; valid
                    # identity refits at A ~ 1-2 pass untouched). All decline
                    # families share one semantics: the candidate is never
                    # considered. If every eligible target declines,
                    # min_fitness stays inf and the reroll path below fires.
                    fitness = self.suboperators['fitness_calculation'].apply(objective, arguments = subop_args['fitness_calculation'], force_out_of_place = True)
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
                # Every eligible target produced inf fitness for the
                # post-restore structure -- reroll this single equation
                # locally (cheap) and continue the outer loop.
                _loop_stats.record('EqRPS.inf_fitness_regen', 1, 1)
                objective.randomize()
                continue

            objective.weights_internal = weights_internal
            objective.weights_final = weights_final
            objective._cached_sw_weights = sw_weights
            objective._cached_vc_score = vc_score
            objective.weights_internal_evald = True
            objective.weights_final_evald = True
            objective.target_idx = min_idx

            if not self.simplify_equation(objective):
                objective.simplified = True
            if objective.target is not None and objective.target.contains_deriv(objective.main_var_to_explain):
                objective.is_correct_right_part = True

        _loop_stats.record('EqRPS.outer', outer_attempts, outer_max_iter)
        # Drop the super-Gram so a downstream consumer (e.g. fitness
        # recomputation outside the term-sweep) falls back to the
        # per-target ``GramSetup.__init__`` path; the cached super-Gram
        # is only valid for the structure observed during the sweep.
        objective._gram_super = None
        objective.right_part_selected = True
        # Exit guarantee: a valid target MUST leave RPS. The cap-break path
        # (outer loop exhausted) or a structural reroll inside the loop
        # (``objective.randomize()`` on inf-fitness) can leave the identity-
        # tracked target as None; install a deterministic, valid target so the
        # downstream ``Equation.evaluate`` never indexes a stale/None position.
        # Candidates: terms carrying a derivative of the explained variable
        # (the only physically valid right parts); fall back to term 0. Among
        # them, PREFER the first whose sweep-style fit is not declined by the
        # host (a None fit = zeroed / degenerate / amplified-identity, one
        # unified semantics) -- the exit guarantee must not hand a declined
        # candidate a free pass around the sweep. If every candidate is
        # declined, the first deriv-carrying term is installed anyway (a
        # target must leave RPS) and the in-place fitness backstop condemns
        # the amplified fit to LOSS_NAN_VAL (SolverFreeFitness.apply).
        if objective.target is None and objective.structure:
            deriv_idxs = [i for i, term in enumerate(objective.structure)
                          if term.contains_deriv(objective.main_var_to_explain)]
            chosen = None
            if global_var.rps_amplification_cap is not None:
                for cand_idx in deriv_idxs:
                    objective.target_idx = cand_idx
                    cand_fit = self.suboperators['fitness_calculation'].apply(
                        objective, arguments=subop_args['fitness_calculation'],
                        force_out_of_place=True)
                    objective.weights_internal_evald = False
                    objective.weights_final_evald = False
                    if cand_fit is not None:
                        chosen = cand_idx
                        break
            if chosen is None:
                chosen = deriv_idxs[0] if deriv_idxs else 0
            objective.target_idx = chosen
            _loop_stats.record('EqRPS.exit_target_fallback', 1, 1)
        objective.remove_zero_terms()
        # Hard invariant: no duplicate terms may leave RPS. simplify and
        # scrub both regenerate-then-drop, so a surviving duplicate is a
        # logic error to surface HERE -- not one generation later at the
        # crossover/mutation assert (the crash site that "lies").
        _final_sigs = {term.factors_labels for term in objective.structure}
        assert len(_final_sigs) == len(objective.structure), \
            'EqRightPartSelector.apply: duplicate terms survived RPS.'

    def simplify_equation(self, objective: Equation):
        # Get nonzero terms
        tgt = objective.target_idx
        nonzero_terms_mask = np.array([False if weight == 0 else True for weight in objective.weights_internal], dtype=np.int32)
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
        # Scrubbing the target-side member breaks every such family (the
        # partner alone cannot cancel and dies at the refit -- measured
        # under both the chi2 and vcoef keep-rules). Component padding
        # also BLOCKS the common-factor cancellation below; scrubbing it
        # first lets the surviving reduced identity (which DOES share the
        # factor) collapse to its canonical form on a later pass.
        def _soak_related(term_):
            return (_term_divides(term_, objective.target)
                    or _term_divides(objective.target, term_))
        divisors = [term for term in nonzero_terms[:-1]
                    if _soak_related(term)]
        if divisors:
            def _not_soak_related(term_):
                return not _soak_related(term_)
            for term in divisors:
                status = _regen_or_drop_term(
                    objective, term, max_iter=100,
                    stats_name='simplify_equation.divisor_scrub',
                    extra_ok=_not_soak_related)
                if status in ('target', 'floor'):
                    # Cannot scrub without degenerating the equation --
                    # decline and let the outer RPS loop reset/re-select.
                    return False
            try:
                objective.reset_state(reset_right_part=False)
            except TypeError:
                objective.reset_state()
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
                        term.reset_saved_state()
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
                    try:
                        objective.reset_state(reset_right_part=False)
                    except TypeError:
                        objective.reset_state()
                    return True
        common_factors = _common_factors(equation_terms)
        if not common_factors:
            # No symbolic reduction applies -- run the numeric completion of
            # the duplicate-term invariant: exactly linearly dependent
            # nonzero feature columns (the generalization of a duplicate
            # term the label test cannot see, e.g. two symbolically distinct
            # products evaluating to proportional fields) make the active-
            # support OLS refit rank-deficient ("infinite solutions"), so
            # such terms are regenerated like duplicates. The target is
            # never touched: a target in the SPAN of independent features is
            # a perfect fit (a valid identity), not a degeneracy.
            return self._regenerate_dependent_terms(objective,
                                                    nonzero_terms[:-1])

        for common_factor in common_factors:
            # Min power across the matching factor in every nonzero term.
            min_order = np.inf
            for term in nonzero_terms:
                for factor in term.structure:
                    if factor.structural_label_without_power == common_factor:
                        if factor.cache_label[1][0] < min_order:
                            min_order = factor.cache_label[1][0]

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
                term.reset_saved_state()

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
            # empty / non-meaningful: regenerate it; if the pool can't
            # yield a unique, meaningful replacement within the cap, DROP
            # it. A duplicate must never ride out of RPS -- see the exit
            # assert in ``apply``.
            for term in nonzero_terms:
                status = _regen_or_drop_term(
                    objective, term, max_iter=100,
                    stats_name='simplify_equation.replace_term')
                if status in ('target', 'floor'):
                    # Offending term is the RPS target, or dropping would
                    # degenerate the equation -> decline this
                    # simplification and let the outer RPS loop reset and
                    # re-select.
                    return False

            # Structure changed: invalidate stale fitness /
            # weights / AIC caches while leaving RPS to the
            # caller's outer loop.
            try:
                objective.reset_state(reset_right_part=False)
            except TypeError:
                objective.reset_state()
            return True
        return False

    def _regenerate_dependent_terms(self, objective: Equation,
                                    feature_terms) -> bool:
        """Regenerate nonzero non-target terms whose evaluated columns are
        EXACTLY linearly dependent (incl. against the intercept).

        Such columns leave the active-support coefficient solve with
        infinitely many solutions (the observed lv / lorenz failure), so
        they are held to the same regenerate-n-then-drop policy as
        duplicate labels. Strictly machine-exact (``_LIN_DEP_RTOL`` on unit
        columns): near-collinear physics -- including valid analytical
        identities -- is never touched, in keeping with the
        no-collinearity-machinery principle.

        Returns True iff the structure changed; the caller's outer RPS loop
        then re-runs the term sweep and re-checks (bounded by its own
        iteration cap).
        """
        if not feature_terms:
            return False
        cols = [np.asarray(term.evaluate(False, grids=None),
                           dtype=float).reshape(-1)
                for term in feature_terms]
        scan = _flag_dependent_columns(cols)
        if scan is None:
            # Non-finite column: dependence undecidable, mirror the
            # super-Gram skip rather than regenerating on a data problem.
            _loop_stats.record('simplify_equation.lin_dep_skip', 1, 1)
            return False
        flagged_idx, basis = scan
        if not flagged_idx:
            return False

        def _independent(term):
            col = np.asarray(term.evaluate(False, grids=None),
                             dtype=float).reshape(-1)
            if not np.all(np.isfinite(col)):
                return False
            nrm = np.linalg.norm(col)
            if nrm == 0.0:
                return False
            u = col / nrm
            K = np.stack(basis, axis=1)
            coef, *_ = np.linalg.lstsq(K, u, rcond=None)
            return np.linalg.norm(u - K @ coef) >= _LIN_DEP_RTOL

        changed = False
        for i in flagged_idx:
            term = feature_terms[i]
            status = _regen_or_drop_term(
                objective, term, max_iter=100,
                stats_name='simplify_equation.lin_dep',
                extra_ok=_independent)
            if status == 'floor':
                break
            if status in ('regenerated', 'dropped'):
                changed = True
                if status == 'regenerated':
                    # The accepted replacement joins the basis so the
                    # remaining flagged terms are checked against it too.
                    col = np.asarray(term.evaluate(False, grids=None),
                                     dtype=float).reshape(-1)
                    basis.append(col / np.linalg.norm(col))
        if changed:
            try:
                objective.reset_state(reset_right_part=False)
            except TypeError:
                objective.reset_state()
        return changed

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


# Relative projection residual (on unit-normalized columns) below which an
# evaluated term column counts as EXACTLY linearly dependent on the columns
# before it. This is a machine-exactness test for rank deficiency ("infinite
# solutions" in the OLS/Gram solves), NOT a collinearity threshold: exact
# dependence projects to ~eps*conditioning, while even severely near-collinear
# physics (incl. valid analytical identities) sits many orders of magnitude
# above 1e-10.
def _common_factors(equation_terms) -> list:
    """Factors shared by every term of an equation, in a deterministic order.

    ``frozenset`` iteration order over label tuples depends on the
    process-wide string-hash salt, which Python randomizes per process
    unless ``PYTHONHASHSEED`` is set. ``simplify_equation`` walks this list
    while MUTATING the terms -- reducing the order of each common factor and
    dropping factors whose power reaches zero -- so with two or more common
    factors a different order yields a different simplified equation, and the
    search diverges from there.

    That made whole runs irreproducible across processes despite the explicit
    random / numpy / torch seeding: the same seed on the same data reached
    different equations with different costs. Sorting pins the order.

    Elements are ``Factor.structural_label_without_power``, i.e.
    ``(label, quantized_params)`` with numeric params, so they always compare.
    """
    return sorted(frozenset.intersection(*equation_terms))


_LIN_DEP_RTOL = 1e-10


def _flag_dependent_columns(cols, rtol: float = _LIN_DEP_RTOL):
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


def _amplification_ratio(objective: Equation) -> float:
    """Amplification ratio of the CURRENT candidate right-part fit:

        A = sum_j |c_j| * ||col_j||  /  ||target col||

    over the nonzero non-target terms plus the fitted intercept
    (``weights_internal = [*term_coefs, intercept]``, the
    ``LinRegBasedCoeffsEquation`` layout). A ~ 1 when the terms combine
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
    t = np.asarray(objective.structure[tgt].evaluate(False, grids=()),
                   dtype=float).reshape(-1)
    den = np.linalg.norm(t)
    if not np.isfinite(den) or den == 0.0:
        return np.inf
    nonrs = [term for i, term in enumerate(objective.structure) if i != tgt]
    num = 0.0
    for wj, term in zip(w[:len(nonrs)], nonrs):
        if wj == 0.0:
            continue
        col = np.asarray(term.evaluate(False, grids=()),
                         dtype=float).reshape(-1)
        num += abs(wj) * np.linalg.norm(col)
    if len(w) > len(nonrs) and w[-1] != 0.0:
        num += abs(w[-1]) * np.sqrt(t.size)  # intercept column of ones
    return num / den


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


def _regen_or_drop_term(equation: Equation, term, *, max_iter: int = 100,
                        min_terms: int = 2,
                        stats_name: str = 'simplify_equation.regen_or_drop',
                        extra_ok=None) -> str:
    """Make ``equation.structure`` unique w.r.t. ``term`` by regenerating
    ``term`` up to ``max_iter`` times; if it is still empty / non-meaningful
    / a duplicate, DROP it from the structure.

    ``extra_ok`` (optional ``Term -> bool``) extends the acceptability
    predicate beyond label uniqueness -- used by the linear-dependence
    scrub, whose redundant columns are label-unique yet must still be
    regenerated (and dropped on cap-hit, like a persistent duplicate).

    This is the simplify/scrub cap-hit policy -- *regenerate-n-then-drop* --
    deliberately distinct from the *keep-or-revert* ``retry_until_unique``
    policy used by ``Equation.__init__`` and the mutation operators.
    ``max_iter == 0`` means "drop immediately if unacceptable" (no
    regeneration) -- used by the degree-reduction branch and the scrub
    duplicate gate.

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
        if len(signatures) != len(equation.structure):
            return False
        return extra_ok is None or bool(extra_ok(term))

    cap = max_iter if max_iter > 0 else 1
    if _acceptable():
        _loop_stats.record(stats_name, 1, cap)
        return 'ok'

    attempts = 0
    for _ in range(max_iter):
        attempts += 1
        term.randomize()
        term.reset_saved_state()
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
    if len(equation.structure) <= min_terms:
        equation._invalidate_label_cache()
        return 'floor'
    equation.structure = [t for t in equation.structure if t is not drop_term]
    # No manual target_idx decrement: the identity-tracked target auto-tracks
    # the surviving target Term (``drop_term`` is never the target -- the
    # branch above reroutes a target collision to the other duplicate).
    equation._invalidate_label_cache()
    return 'dropped'


def _target_term_in_other_equation(eq_with_target: Equation,
                                   eq_other: Equation):
    """Return ``eq_with_target``'s TARGET-term factor signature iff that
    whole term also appears as a (standalone) term in ``eq_other``'s ACTIVE
    structure -- or, for a DECORATED target, the signature of its
    derivative CORE when that core leaks as a standalone term; otherwise
    ``None``.

    This enforces target-term uniqueness across a system: the explained
    right-part term of one equation may not be carried as a complete term
    by another equation of the same system (the documented Lotka-Volterra
    leak where ``dv/dx0`` rode into the ``u`` equation as its own term).

    It is deliberately a WHOLE-TERM equality test -- ``target.factors_labels``
    is one element of ``eq_other.active_terms_labels`` -- NOT a
    sub-product/divides test: a target derivative appearing only as a
    FACTOR inside a composite coupling term (e.g. continuity's ``v_y``
    inside the v-momentum ``v*v_y`` of true Navier-Stokes) is legitimate
    physics and is left untouched.

    Directional by construction (``eq_with_target``'s target into
    ``eq_other``); the caller scans both orderings of every pair. The
    target term is re-fetched as ``structure[target_idx]`` (never cached
    across rerolls) and guarded exactly as the existing degeneracy path
    (right_part_selection.py:608-612). ACTIVE scope means a zero-weight
    padding copy is ignored; any later reactivation is caught by the next
    per-generation RPS pass.
    """
    tgt = eq_with_target.target
    if tgt is None:
        return None
    target_sig = tgt.factors_labels
    if target_sig in eq_other.active_terms_labels:
        return target_sig
    # DERIV-CORE extension: a DECORATED target must not dodge the ban by
    # wearing a costume -- a ``dv/dx0 * cos(2t)`` target reserves
    # ``dv/dx0`` exactly as a bare ``dv/dx0`` target would. The core is
    # the target's derivative factors taken as a standalone term
    # (whole-term equality against it, same convention as above -- the
    # core as a FACTOR of a composite coupling term stays legal).
    # Measured on lv: the only identity-family members that STRICTLY
    # DOMINATE the truth (the sum identity and its d/dt lift) carry the
    # other equation's target core as a standalone term; every
    # substituted variant (v_tt for v_t, u*v_t for v_t) measurably fails
    # to dominate. For a PURE target the core equals the whole term and
    # this clause adds nothing.
    core_sig = frozenset(
        f.structural_label for f in tgt.structure
        if f.is_deriv and f.deriv_code != [None, ]
        and f.evaluator._evaluator == simple_function_evaluator)
    if core_sig and core_sig != target_sig \
            and core_sig in eq_other.active_terms_labels:
        return core_sig
    return None


def _wrap_term_with_factor(equation: Equation, term: Term, banned_sigs,
                           max_tries: int = 20) -> bool:
    """Demote ``term`` (a leaked standalone copy of another equation's
    target) into a FACTOR of a composite term: multiply it by one extra
    pool factor instead of destroying it.

    The whole-term uniqueness invariant allows another equation's target
    as a factor inside a composite coupling term, just not as a separate
    term. The leaked standalone copy is often the small-angle/stepping-stone
    form of legitimate coupling physics (e.g. ``th2''`` inside the ``th1``
    double-pendulum equation, whose true form is ``cos(D)*th2''``), so
    wrapping preserves that information where a reroll erases it.

    Accepts the wrap iff the new signature (a) differs from the original,
    (b) is outside ``banned_sigs`` and (c) duplicates no other term of the
    equation. Restores the original structure and returns False when the
    per-term factor cap is already reached or no acceptable wrap was drawn
    -- the caller then falls back to the randomize repair.
    """
    # The authoritative factor cap is the equation-level metaparameter
    # (evolution-built terms mirror it, but e.g. translated equations
    # leave the Term attribute at its constructor default of 1).
    try:
        cap = equation.metaparameters['max_factors_in_term']['value']
    except (AttributeError, KeyError, TypeError):
        cap = term.max_factors_in_term
    if isinstance(cap, dict):
        cap = max(cap['factors_num'])
    if len(term.structure) >= int(cap):
        return False
    original = term.structure
    original_sig = term.factors_labels
    other_sigs = {t.factors_labels for t in equation.structure if t is not term}
    attempts = 0
    for _ in range(max_tries):
        attempts += 1
        try:
            _, factor = term.pool.create(label=None, create_meaningful=False)
        except ValueError:
            break
        # filter_powers clones via copy_for_power_update, so ``original``
        # stays intact for the restore path below.
        term.structure = filter_powers(list(original) + [factor])
        term.reset_saved_state()
        sig = term.factors_labels
        if sig != original_sig and sig not in banned_sigs and sig not in other_sigs:
            _loop_stats.record('break_equation_duplication.wrap', attempts, max_tries)
            return True
    term.structure = original
    term.reset_saved_state()
    _loop_stats.record('break_equation_duplication.wrap_fail', attempts, max_tries)
    return False


def _break_equation_duplication(equation: Equation, shared_sigs, *,
                                preferred_sigs=(), max_iter: int = 2000) -> bool:
    """Break a system-level degeneracy by repairing ONE non-target term of
    ``equation`` whose factor signature belongs to ``shared_sigs`` (the
    active structure this equation shares with another equation of the
    system). Repair prefers demoting the term to a factor of a composite
    term (:func:`_wrap_term_with_factor`); only when that fails is the
    term randomized away.

    The term matching one of ``preferred_sigs`` (typically the other
    equation's target signature) is chosen first, so the rerolled equation
    moves away from "the other equation's explained quantity" before
    touching genuinely shared coupling terms. The randomize loop demands
    that the replacement (a) leaves ``shared_sigs`` and (b) does not
    duplicate another term; on cap-hit a surviving duplicate is dropped via
    the ``_regen_or_drop_term`` drop policy (a still-shared-but-unique term
    is tolerated -- the caller's convergence loop re-checks).

    Returns True if the structure changed; cached fitness/weight state is
    reset on the way out so the caller can re-run right-part selection.
    """
    candidates = [term for idx, term in enumerate(equation.structure)
                  if idx != getattr(equation, 'target_idx', None)
                  and term.factors_labels in shared_sigs]
    if not candidates:
        # The shared structure is carried entirely by the target term (a
        # single-term law explained from both sides). Nothing safe to
        # randomize here; the caller's pass cap tolerates the leftover.
        return False

    preferred = [t for t in candidates if t.factors_labels in preferred_sigs]
    term = preferred[0] if preferred else candidates[0]

    # Demote-to-factor repair first: keep the leaked target alive as a
    # factor of a composite coupling term (legal under the whole-term
    # rule) instead of rerolling it into unrelated structure. Falls back
    # to the randomize path when the wrap cannot produce a unique term.
    if _wrap_term_with_factor(equation, term, set(shared_sigs)):
        try:
            equation.reset_state(reset_right_part=False)
        except TypeError:
            equation.reset_state()
        return True

    attempts = 0
    for _ in range(max_iter):
        attempts += 1
        term.randomize()
        term.reset_saved_state()
        signatures = {t.factors_labels for t in equation.structure}
        duplicate = len(signatures) != len(equation.structure)
        if term.factors_labels not in shared_sigs and not duplicate:
            break
    _loop_stats.record('break_equation_duplication', attempts, max_iter)
    # Cap-hit may leave the rerolled term as a DUPLICATE -- a duplicate must
    # never ride out of RPS, so drop it (regenerate attempts already spent).
    _regen_or_drop_term(equation, term, max_iter=0,
                        stats_name='break_equation_duplication.drop')

    try:
        equation.reset_state(reset_right_part=False)
    except TypeError:
        equation.reset_state()
    return True


class SoEqRightPartSelector(CompoundOperator):
    """Chromosome-level RPS that prevents system-level degeneracy.

    Invariant: no equation's TARGET term -- nor, for decorated targets,
    the target's derivative CORE -- may appear as a whole (standalone)
    term in ANY other equation of the system: the explained right-part
    term of one equation is reserved to that equation, costume or no
    costume (``dv/dx0 * cos(2t)`` reserves ``dv/dx0``). This is a
    whole-term equality rule, NOT a sub-product one: a target derivative
    appearing only as a FACTOR inside a composite coupling term of
    another equation (e.g. continuity's ``v_y`` inside the v-momentum
    convective term ``v*v_y`` of Navier-Stokes) is legitimate physics and
    is left untouched.

    Cross-equation FACTOR sharing is otherwise allowed: an equation for
    ``v`` may carry ``du/dx0`` inside a composite term even when the
    equation for ``u`` explains ``du/dx0`` -- only a bare standalone
    ``du/dx0`` term in the ``v`` equation is forbidden.

    This subsumes the old "no two equations share an identical active
    structure" guard: two equations that collapse onto the same law (e.g.
    both Navier-Stokes velocity equations becoming continuity) necessarily
    have DIFFERENT targets, so each carries the other's target as a
    standalone term and is broken here. (The only case it cannot catch --
    two full duplicates sharing the identical target -- requires a composite
    target carrying both equations' main-var derivatives, which is
    unreachable for the studied systems.)

    Mechanics: a plain per-equation forward pass first, then a bounded
    convergence loop that, each pass, rerolls any equation carrying another
    equation's target term as a whole term (via
    ``_break_equation_duplication`` + right-part re-selection). The loop
    exits at the first pass with no changes (fixed point).
    """
    key = 'SoEqRightPartSelector'

    @_loop_stats.timed('SoEqRPS.apply')
    def apply(self, objective, arguments: dict):
        """Run per-equation RPS, then resolve system degeneracies in-place.

        Failures inside ``EqRightPartSelector.apply`` are handled locally
        via per-equation ``objective.randomize()`` -- this method has no
        regen signal to forward.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)
        eq_selector = self.suboperators['eq_right_part_selector']
        eq_args = subop_args.get('eq_right_part_selector', arguments)

        equations = list(objective)

        # Forward pass: plain per-equation right-part selection. No
        # cross-equation scrubbing -- shared terms are legitimate coupling.
        for equation in equations:
            eq_selector.apply(objective=equation, arguments=eq_args)

        # Degeneracy resolution: enforce target-term uniqueness until a full
        # pass makes no change or the pass budget is exhausted.
        max_passes = 50
        passes_used = 0
        for _ in range(max_passes):
            passes_used += 1
            any_changes = False
            # Target-term uniqueness: no equation's TARGET term may appear
            # as a whole (standalone) term in ANOTHER equation of the
            # system. Directional -- scan every ordered pair (i -> j) and,
            # when eq i's target rides in eq j, reroll eq j's offending copy
            # via the same _break_equation_duplication drop-or-reroll
            # primitive, then re-select eq j's right part. A target
            # derivative that appears only as a FACTOR inside a composite
            # coupling term of eq j (e.g. continuity's ``v_y`` in ``v*v_y``)
            # is left untouched -- this is whole-term equality, not
            # sub-product. If the only match is eq j's own target,
            # _break_equation_duplication finds no rerollable candidate and
            # returns False, so the pass tolerates it (cannot reroll a
            # target).
            for i in range(len(equations)):
                for j in range(len(equations)):
                    if i == j:
                        continue
                    eq_i, eq_j = equations[i], equations[j]
                    leak_sig = _target_term_in_other_equation(eq_i, eq_j)
                    if leak_sig is None:
                        continue
                    changed = _break_equation_duplication(
                        eq_j, {leak_sig}, preferred_sigs={leak_sig})
                    if not changed:
                        continue
                    eq_j.right_part_selected = False
                    eq_j.simplified = False
                    eq_j.is_correct_right_part = False
                    eq_selector.apply(objective=eq_j, arguments=eq_args)
                    _loop_stats.record('SoEqRPS.target_leak_repair', 1, 1)
                    any_changes = True

            if not any_changes:
                break
        _loop_stats.record('SoEqRPS.degeneracy_passes', passes_used, max_passes)

    def use_default_tags(self):
        self._tags = {'right part selection', 'chromosome level',
                      'contains suboperators', 'inplace'}
