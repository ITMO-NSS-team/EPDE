#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The repair-mode contract of ``EqRightPartSelector``: right-part selection
USES ONLY ITS INPUT.

The operator used to reroll a candidate until something was accepted -- the
whole equation when every target was declined, a term the ratio-fit or
linear-dependence scrub disliked, a random derivative term after a
cancellation stripped the target's -- and at its outer cap it let out whatever
state it had reached. Its output was therefore not a function of the
candidate, and under a strict keep rule the population collapsed onto the few
forms the rerolls reached. Pinned here:

* the equation that leaves is built from the terms that came in -- on real
  fits over random candidates, on a cancellation that reaches the true law,
  and on a scripted ratio-fit scrub;
* a declined candidate leaves with the structure it came in with: a
  cancellation that would leave the target with no meaningful factor, or a
  feature with none at the two-term floor, and a scrub -- the ratio-fit or
  the linear-dependence one -- that would drop below two terms are declined
  before anything is rewritten, and the label comes back declined again;
* a candidate with no right part leaves DECLINED (stamped fit-degenerate), the
  stamp stands through the chromosome fitness the strategy wires -- solver-free
  and solver-based -- and the CALLER's regeneration redraws it: up to
  ``RESAMPLE_BUDGET`` times, fitting only the redrawn equation, and not at all
  in a system with no degenerate equation; a fit the host stamped, and in
  ``'assignment'`` / ``'plain'`` mode an unassigned equation (the same
  all-zero stamp), share ``max_passes`` instead; the single-objective search
  does the same through ``ResamplingRightPartSelector``, which never redraws
  a stamped fit;
* a cancellation that would leave NO term with a derivative of the variable
  declines, before any reduction, instead of handing the next pass to
  ``restore_property``; one that strips only the target's derivative lets the
  next pass re-select among the reduced terms;
* at the outer cap nothing leaves as an accepted form: not a derivative-free
  target, not a form ``simplify_equation`` would still rewrite -- and the
  capped candidate leaves with the structure it came in with, though earlier
  passes rewrote it, so it comes back to the same passes and is declined at
  the cap again; that is the structure after the entry validator, so a
  derivative term the validator drew stays and is not drawn again; below the
  cap, reading that structure changes nothing;
* a declined equation takes no part in the system's cross-equation rules;
* ``amplification_ratio`` is 0.0 when every weight and the intercept are 0.
"""

import copy
import warnings

import numpy as np
import pytest

import epde.operators.common.right_part_selection as rps
from conftest import using_config
from test_lin_dep_scrub_multisample import FakeTerm, N0, N1, _cols, _independent_pair, _run
from test_rps_single_equation_paths import (  # noqa: F401
    DU, DV, U_TOKEN, V_TOKEN, _assert_declined, _selector, _stub_fit, _terms,
    _translated_u_equation, _u_equation, no_draws, uv_pool)

U_DU = f'{U_TOKEN} * {DU}'
V_DU = f'{V_TOKEN} * {DU}'
D2U = 'd^2u/dx0^2{power: 1.0}'
X = 'x_0{power: 1.0, dim: 0.0}'
SIN = 'sin{power: 1.0, freq: 2.0, dim: 0.0}'


def _stub_fit_by_target(monkeypatch, accepted):
    """Fit/score stand-in: unit weights on every feature, a score for a target
    whose name is in ``accepted`` (read at call time) and None -- the host's
    decline -- for any other. Returns the names of the targets it was asked
    to fit, in order."""
    calls = []

    def fake_fit(self, objective, subop_args):
        calls.append(objective.target.name)
        weights = np.ones(len(objective.structure))
        objective.weights_internal = weights
        objective.weights_final = weights.copy()
        objective.weights_internal_evald = True
        objective.weights_final_evald = True
        objective._cached_sw_weights = None
        objective._cached_vc_score = None
        return 0.5 if objective.target.name in accepted else None

    monkeypatch.setattr(rps.EqRightPartSelector, '_fit_and_score', fake_fit)
    return calls


def _stub_fit_scored(monkeypatch, scores, zero=()):
    """Fit/score stand-in: unit weights on every feature but those named in
    ``zero``, a zero intercept, and the score ``scores`` gives the target's
    name -- None, the host's decline, for a name it lacks. Returns the names
    of the targets it was asked to fit, in order."""
    calls = []

    def fake_fit(self, objective, subop_args):
        target = objective.target
        calls.append(target.name)
        weights = np.array([0.0 if term.name in zero else 1.0
                            for term in objective.structure if term is not target]
                           + [0.0])
        objective.weights_internal = weights
        objective.weights_final = weights.copy()
        objective.weights_internal_evald = True
        objective.weights_final_evald = True
        objective._cached_sw_weights = None
        objective._cached_vc_score = None
        return scores.get(target.name)

    monkeypatch.setattr(rps.EqRightPartSelector, '_fit_and_score', fake_fit)
    return calls


def _refuse_restore(monkeypatch):
    """``restore_property`` may not run after entry."""
    from epde.structure.main_structures import Equation

    def refuse_restore(self, **kwargs):
        raise AssertionError('restore_property was called after entry')

    monkeypatch.setattr(Equation, 'restore_property', refuse_restore)


def _apply(eq, selector=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        (selector or _selector()).apply(objective=eq, arguments={})
    return [str(w.message) for w in caught]


def _snapshot(eq):
    """Each term's identity and factor labels (power dropped)."""
    return {id(term): set(term.factors_labels_without_power) for term in eq.structure}


def _objects(eq):
    """The Term objects of ``eq`` in order, each with its Factor objects."""
    return [(term, list(term.structure)) for term in eq.structure]


def _same_objects(eq, objects):
    """``eq`` holds exactly ``objects`` (``_objects``), by identity: ``==``
    on a Term or a Factor compares labels."""
    now = _objects(eq)
    return len(now) == len(objects) and all(
        term is was and len(factors) == len(had)
        and all(factor is old for factor, old in zip(factors, had))
        for (term, factors), (was, had) in zip(now, objects))


def _random_state():
    """numpy's global random state, comparable with ``==``."""
    _, keys, pos, has_gauss, cached = np.random.get_state()
    return keys.tobytes(), pos, has_gauss, cached


# ---------------------------------------------------------------------------
# The output is built from the input: real fits.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def probe_pool():
    """``u = sin 2t + 0.3 t`` with ``x_0`` up to power 2: the data on which
    repair mode rerolled every candidate into ``du/dx0 = c*u`` under the
    default keep rule. 80 points: the fourth-order FD boundary leaves 72
    rows, above the instability metrics' row floors."""
    import epde
    from epde.interface.prepared_tokens import GridTokens
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False},
                             equation_terms_max_number=3, equation_factors_max_number=2)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0.0, 4.0, 80)
    _, domain = search.createDomain(t, ID=0)
    _, trajectory = search.createTrajectory({'u': np.sin(2 * t) + 0.3 * t}, domain,
                                            cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(1,),
                       additional_tokens=[GridTokens(['x_0'], max_power=2)])
    return search.pool


@pytest.fixture(scope='module')
def second_order_probe_pool():
    """The probe data with ``d^2u/dx0^2`` in the pool, whose law is
    ``u'' = -4*u + 1.2*x_0``, and ``sin``/``cos`` at frequency 2: a family
    that is not meaningful, so a term holding nothing else is one the grammar
    does not allow. The fixtures of this module share the global caches, so
    each one's tests run together: this one's sit between ``probe_pool``'s
    and ``uv_pool``'s."""
    import epde
    from epde.interface.prepared_tokens import GridTokens, TrigonometricTokens
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False},
                             equation_terms_max_number=4, equation_factors_max_number=2)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0.0, 4.0, 60)
    _, domain = search.createDomain(t, ID=0)
    _, trajectory = search.createTrajectory({'u': np.sin(2 * t) + 0.3 * t}, domain,
                                            cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(2,),
                       additional_tokens=[GridTokens(['x_0'], max_power=2),
                                          TrigonometricTokens(freq_values=[2.0],
                                                              dimensionality=0)])
    return search.pool


def _probe_equation(pool, text):
    """The u-equation ``text``, translated on a probe pool."""
    from epde.interface.equation_translator import translate_equation
    eq = translate_equation({'u': text}, pool, all_vars=['u']).vals['u']
    eq.main_var_to_explain = 'u'
    return eq


def _real_selector(objectives=None):
    """The live per-equation selector: the configured keep rule, the
    coefficient refit and a solver-free host -- discrepancy + instability, or
    ``objectives`` (primary first)."""
    from epde.interface.search_config import active_config
    from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
    from epde.operators.common.fitness import SolverFreeFitness
    from epde.operators.common.objectives import Discrepancy, Instability
    from epde.operators.common.sparsity import build_sparsity_operator
    from epde.operators.utils.template import add_base_param_to_operator
    if objectives is None:
        objectives = [Discrepancy(), Instability()]
    host = SolverFreeFitness(['penalty_coeff'], objectives=objectives, primary=objectives[0])
    add_base_param_to_operator(host, {})
    cfg = active_config().objectives
    selector = rps.EqRightPartSelector()
    selector.set_suboperators({'fitness_calculation': host,
                               'sparsity': build_sparsity_operator(cfg.sparsity_cls,
                                                                   cfg.sparsity_kwargs),
                               'coeff_calc': LinRegBasedCoeffsEquation()})
    return selector


_META = {'sparsity': {'optimizable': True, 'value': 1.},
         'max_terms_number': {'optimizable': False, 'value': 3},
         'max_factors_in_term': {'optimizable': False, 'value': 2}}


class TestTheOutputIsBuiltFromTheInput:

    def test_random_candidates_on_real_fits(self, probe_pool, monkeypatch):
        """Twenty random candidates through the live selector. Each leaves
        either declined or fitted, and in both cases built from its own
        terms: the same objects, each holding a subset of its factors (the
        only rewrites are drops and lowered powers). The one term that may be
        added is the entry restore's, when no term carried a derivative of u.
        Nothing is redrawn: no ``Equation.randomize``, and no
        ``Term.randomize`` outside that restore."""
        from epde.structure.main_structures import Equation, Term
        from epde.operators.multiobjective.moeadd_specific import _equation_is_degenerate
        selector = _real_selector()
        state = {'rps': False, 'restore': False}
        draws, restored = [], []
        real_restore, real_draw = Equation.restore_property, Term.randomize

        def spy_restore(self, **kwargs):
            state['restore'] = True
            before = {id(t) for t in self.structure}
            try:
                return real_restore(self, **kwargs)
            finally:
                state['restore'] = False
                restored.extend(id(t) for t in self.structure if id(t) not in before)

        def spy_draw(self, *args, **kwargs):
            if state['rps'] and not state['restore']:
                draws.append(self)
            return real_draw(self, *args, **kwargs)

        def no_reroll(self):
            raise AssertionError('right-part selection rerolled the equation')

        candidates = []
        for seed in range(20):
            np.random.seed(seed)
            candidates.append(Equation(probe_pool, [], 'u',
                                       metaparameters=copy.deepcopy(_META)))
        monkeypatch.setattr(Equation, 'restore_property', spy_restore)
        monkeypatch.setattr(Term, 'randomize', spy_draw)
        monkeypatch.setattr(Equation, 'randomize', no_reroll)

        declined = placed = 0
        for seed, eq in enumerate(candidates):
            np.random.seed(100 + seed)
            entry = _snapshot(eq)
            del restored[:]
            state['rps'] = True
            try:
                _apply(eq, selector)
            finally:
                state['rps'] = False
            exit_ = _snapshot(eq)
            assert set(exit_) <= set(entry) | set(restored), seed
            for key, labels in exit_.items():
                if key in entry:
                    assert labels <= entry[key], (seed, labels, entry[key])
            if _equation_is_degenerate(eq):
                declined += 1
                _assert_declined(eq)
            else:
                placed += 1
                assert eq.target.contains_deriv('u')
                # Converged: there is nothing left for simplify to rewrite.
                assert selector.simplify_equation(eq) is False
        assert draws == []
        # The fixture exercises both verdicts: the declined candidates are the
        # ones the operator used to reroll.
        assert declined and placed, (declined, placed)

    def test_a_cancellation_reaches_the_true_law(self, second_order_probe_pool,
                                                 monkeypatch, no_draws):
        """``x_0*u' = a*u*u' + b*u'*u''`` on real fits. Every term shares
        ``u'``, so cancelling it strips the derivative from whichever target
        won -- but ``u'*u''`` sheds it into ``u''``, a term the sweep could not
        take as a target before (two derivatives of u) and can now. So the
        cancellation is no decline: the next pass selects ``u''`` and the
        candidate leaves as the true law ``u'' = a'*u + b'*x_0``, on the same
        Term objects, with no draw and no restore. A decline whenever the
        target alone sheds its derivative refused it."""
        eq = _probe_equation(second_order_probe_pool,
                             f'0.5 * {U_TOKEN} * {DU} + 0.3 * {D2U} * {DU} + 0.0 = {X} * {DU}')
        before = list(eq.structure)
        _refuse_restore(monkeypatch)
        _apply(eq, _real_selector())

        assert not rps._rps_declined(eq)
        assert eq.target.name == D2U
        assert sorted(t.name for t in eq.structure) == sorted([U_TOKEN, X, D2U])
        assert all(any(t is b for b in before) for t in eq.structure)
        assert all(w != 0 for w in eq.weights_final[:-1])

    def test_below_the_cap_reading_the_entry_structure_changes_nothing(
            self, second_order_probe_pool, monkeypatch, no_draws):
        """What the outer cap puts back is read on entry
        (``_structure_snapshot``), on every candidate; below the cap nothing
        reads it back. Reading it draws nothing and writes nothing, so the
        cancellation above, on real fits, leaves bit for bit as it does with
        no entry read -- the operator before the cap restored anything: the
        same terms and target, the same weights and scores, and the same
        random state after it."""
        text = f'0.5 * {U_TOKEN} * {DU} + 0.3 * {D2U} * {DU} + 0.0 = {X} * {DU}'
        runs = []
        for read_entry in (True, False):
            if not read_entry:
                monkeypatch.setattr(rps, '_structure_snapshot', lambda equation: None)
            eq = _probe_equation(second_order_probe_pool, text)
            selector = _real_selector()
            np.random.seed(0)
            _apply(eq, selector)
            runs.append(([t.name for t in eq.structure], eq.target.name,
                         np.asarray(eq.weights_internal).tobytes(),
                         np.asarray(eq.weights_final).tobytes(),
                         eq.fitness_value, eq.coefficients_stability, _random_state()))

        assert runs[0][1] == D2U                  # placed, past the cancellation
        assert runs[0] == runs[1]

    def test_a_target_left_without_a_meaningful_factor_declines_unreduced(
            self, second_order_probe_pool, monkeypatch, no_draws):
        """``sin*u' = c*u*u' + 0*u''``. Cancelling ``u'`` would leave the
        target ``sin``, which has no meaningful factor -- a term the grammar
        does not allow -- while the zero-weight ``u''`` keeps a derivative in
        the structure. Declined before anything is reduced, so the candidate
        leaves with its own structure, and coming back it is declined again.
        It used to be reduced to ``[u, u'', sin]`` first and declined; that
        label came back placed, as ``u'' = c*sin``."""
        eq = _probe_equation(second_order_probe_pool,
                             f'0.5 * {U_DU} + 0.0 * {D2U} + 0.0 = {SIN} * {DU}')
        before = [(t, t.name) for t in eq.structure]
        _refuse_restore(monkeypatch)
        _stub_fit_scored(monkeypatch, {f'{SIN} * {DU}': 0.1, U_DU: 0.3, D2U: 0.5},
                         zero={D2U})
        for _ in range(2):
            eq.right_part_selected = False
            _apply(eq)
            _assert_declined(eq)
            assert [(t, t.name) for t in eq.structure] == before

    @pytest.mark.parametrize('text, zero', [
        (f'0.5 * {U_TOKEN} * {SIN} + 0.0 = {U_DU}', ()),
        (f'0.5 * {U_TOKEN} * {SIN} + 0.0 * {SIN} + 0.0 = {U_DU}', (SIN,)),
    ], ids=['two terms', 'beside the copy the reduction collides with'])
    def test_a_feature_left_without_a_meaningful_factor_at_the_floor_declines_unreduced(
            self, second_order_probe_pool, monkeypatch, no_draws, text, zero):
        """``u*u' = c*u*sin``. Cancelling ``u`` would leave the feature
        ``sin``, which has no meaningful factor and cannot be dropped from
        two terms. A zero-weight ``sin`` beside it does not help: the reduced
        feature duplicates it, and that copy goes first. Declined before
        anything is reduced; it used to be reduced to ``u' = c*sin`` first,
        a label that comes back placed."""
        eq = _probe_equation(second_order_probe_pool, text)
        before = [(t, t.name) for t in eq.structure]
        _refuse_restore(monkeypatch)
        _stub_fit_scored(monkeypatch, {U_DU: 0.3}, zero=set(zero))
        _apply(eq)

        _assert_declined(eq)
        assert [(t, t.name) for t in eq.structure] == before

    def test_a_feature_left_without_a_meaningful_factor_above_the_floor_is_dropped(
            self, second_order_probe_pool, monkeypatch, no_draws):
        """The control: ``u*u' = a*u*sin + b*u*x_0`` loses the feature
        ``sin`` and keeps two terms, so the reduction runs, on the same Term
        objects, and the next pass places ``u' = c*x_0``."""
        eq = _probe_equation(second_order_probe_pool,
                             f'0.5 * {U_TOKEN} * {SIN} + 0.3 * {U_TOKEN} * {X} + 0.0 = {U_DU}')
        before = list(eq.structure)
        _refuse_restore(monkeypatch)
        calls = _stub_fit_scored(monkeypatch, {U_DU: 0.3, DU: 0.3})
        _apply(eq)

        assert not rps._rps_declined(eq)
        assert calls == [U_DU, DU]
        assert eq.target.name == DU
        assert sorted(t.name for t in eq.structure) == sorted([X, DU])
        assert all(any(t is b for b in before) for t in eq.structure)

    def test_the_ratio_fit_scrub_drops(self, uv_pool, monkeypatch, no_draws):
        """``u*du`` is a multiple of the target ``du``: the scrub drops it and
        the selection converges on what is left of the input -- where it used
        to redraw ``u*du`` until no relative of the target came up."""
        eq = _translated_u_equation(uv_pool, f'0.5 * {V_TOKEN} + 0.3 * {U_DU} + 0.0 = {DU}')
        before = list(eq.structure)
        calls = _stub_fit_by_target(monkeypatch, {DU})
        _apply(eq)

        assert eq.target.name == DU and not rps._rps_declined(eq)
        assert [t.name for t in eq.structure] == [V_TOKEN, DU]
        assert all(any(t is b for b in before) for t in eq.structure)
        # Two sweeps: the scrub's structure change, then convergence.
        assert calls == [U_DU, DU, DU]

    def test_a_scrub_below_two_terms_declines_unreduced(self, uv_pool, monkeypatch,
                                                        no_draws):
        """``u*du = a*u + b*du``: both features divide the target, and
        dropping both would leave it alone. Declined before anything is
        dropped -- the scrub used to drop ``u``, refuse ``du`` at the floor,
        and leave ``[du, u*du]`` declined, a label that is not the
        candidate's."""
        eq = _translated_u_equation(uv_pool, f'0.5 * {U_TOKEN} + 0.3 * {DU} + 0.0 = {U_DU}')
        before = [(t, t.name) for t in eq.structure]
        _stub_fit_by_target(monkeypatch, {U_DU})
        _apply(eq)

        _assert_declined(eq)
        assert [(t, t.name) for t in eq.structure] == before


# ---------------------------------------------------------------------------
# Declined, and redrawn by the caller.
# ---------------------------------------------------------------------------

def _fitness_condition(system):
    """The strategies' chromosome-fitness condition: the host runs on a
    system with any equation unscored -- and then on EVERY equation."""
    return not getattr(system, 'fitness_calculated')


class TestADeclinedCandidate:
    """A declined equation beside a healthy one, on real fits, through the
    chromosome fitness the MOEA/DD strategy wires. The condition sits on the
    SYSTEM, so the host runs because the healthy equation is unscored, and
    reaches the declined one too: its stamp has to stand there, or the
    declined equation reads as fitted and is placed."""

    @staticmethod
    def _system(pool, u_text=f'0.5 * {U_DU} + 0.0 = {DU}'):
        """``du = c*u*du`` beside ``dv = c*u``. The u-equation has no right
        part: whichever of its two terms is the target, the other is a
        multiple or a component of it, which the ratio-fit scrub cannot drop
        from two terms -- and a zero weight on it is the host's decline. The
        v-equation fits. ``u_text`` replaces the u-equation."""
        from epde.interface.equation_translator import translate_equation
        system = translate_equation({'u': u_text,
                                     'v': f'-0.9 * {U_TOKEN} + 0.0 = {DV}'},
                                    pool, all_vars=['u', 'v'])
        for var in ('u', 'v'):
            system.vals[var].main_var_to_explain = var
            system.vals[var].right_part_selected = False
        return system

    @staticmethod
    def _script_the_redraw(monkeypatch, pool):
        """``Equation.randomize`` redraws to ``du = c*v``, which fits;
        ``Term.randomize`` is refused. Returns the redrawn equations, in
        order, and the redraw's terms."""
        from epde.structure.main_structures import Equation, Term
        redraws, redrawn = [], _terms(pool, f'0.4 * {V_TOKEN} + 0.0 = {DU}')

        def scripted_redraw(self):
            redraws.append(self)
            self.reset_state(True)
            self.structure = list(redrawn)
            self._invalidate_label_cache()

        def refuse(self, *args, **kwargs):
            raise AssertionError('a term was redrawn')

        monkeypatch.setattr(Equation, 'randomize', scripted_redraw)
        monkeypatch.setattr(Term, 'randomize', refuse)
        return redraws, redrawn

    @staticmethod
    def _system_selector(selector):
        sys_selector = rps.SoEqRightPartSelector()
        sys_selector.set_suboperators({'eq_right_part_selector': selector})
        return sys_selector

    def test_comes_back_degenerate_and_the_caller_redraws_it(self, uv_pool, monkeypatch):
        """Solver-free wiring: the sweep's host is the chromosome fitness,
        mapped from gene to system level. The u-equation leaves the selector
        stamped, the v-equation fitted but unscored, so the mapped host runs
        and scores BOTH -- its in-place pass stamps the declined one again
        (all-zero term weights are degenerate). ``regenerate_degenerate_
        equations`` -- not the selector -- then redraws the u-equation, once,
        and the redraw is selected and scored; the v-equation keeps its
        terms."""
        from epde.operators.common.objectives import LOSS_NAN_VAL
        from epde.operators.multiobjective.moeadd_specific import (
            has_degenerate_equation, regenerate_degenerate_equations)
        from epde.operators.utils.operator_mappers import map_operator_between_levels

        selector = _real_selector()
        host = selector.suboperators['fitness_calculation']
        fitness = map_operator_between_levels(host, 'gene level', 'chromosome level',
                                              objective_condition=_fitness_condition)
        sys_selector = self._system_selector(selector)
        scored = []
        real_score = host.apply

        def spy(objective, arguments, force_out_of_place=False):
            if not force_out_of_place:
                scored.append(objective.main_var_to_explain)
            return real_score(objective, arguments, force_out_of_place=force_out_of_place)

        monkeypatch.setattr(host, 'apply', spy)
        system = self._system(uv_pool)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        redraws, redrawn = self._script_the_redraw(monkeypatch, uv_pool)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sys_selector.apply(system, {})
        assert redraws == [], 'the selector rerolled the candidate itself'
        _assert_declined(eq_u)
        assert not rps._rps_declined(eq_v) and eq_v.target.name == DV
        assert not eq_v.fitness_calculated
        v_terms = [id(t) for t in eq_v.structure]

        fitness.apply(system, {})
        assert sorted(scored) == ['u', 'v']
        _assert_declined(eq_u)
        assert eq_v.fitness_calculated and eq_v.fitness_value < LOSS_NAN_VAL
        assert has_degenerate_equation(system)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            regenerate_degenerate_equations(system, sys_selector, fitness, {}, {})
        assert redraws == [eq_u]
        assert not has_degenerate_equation(system)
        assert eq_u.target.name == DU and not rps._rps_declined(eq_u)
        assert {id(t) for t in eq_u.structure} == {id(t) for t in redrawn}
        assert eq_u.fitness_calculated and eq_u.fitness_value < LOSS_NAN_VAL
        assert [id(t) for t in eq_v.structure] == v_terms

    def test_the_solver_host_keeps_the_stamp(self, uv_pool, monkeypatch):
        """Solver wiring (``use_solver=True``): the sweep scores with a
        lightweight solver-free host, the chromosome fitness is
        ``SolverBasedFitness`` under the same condition. It used to solve the
        system and write a finite error over the declined equation's stamp,
        so the declined equation read as fitted and was placed. It does not
        solve a system holding one: the stamp stands and the v-equation waits
        unscored until the caller's redraw makes the system solvable, which is
        then solved once and scored. The solve itself is a stand-in returning
        a zero field; the scoring after it is the host's own."""
        import torch
        from epde.operators.common.fitness import SolverBasedFitness
        from epde.operators.common.objectives import Discrepancy, Instability, LOSS_NAN_VAL
        from epde.operators.multiobjective.moeadd_specific import (
            _equation_is_degenerate, has_degenerate_equation,
            regenerate_degenerate_equations)
        from epde.operators.utils.operator_mappers import OperatorCondition

        solves = []

        def fake_solve(system, domain_key, grids, boundary_conditions, use_fourier):
            solves.append(system)
            return 0.0, lambda grid: torch.zeros(grid.shape[0], 2)

        adapter = type('Adapter', (), {'solve_epde_system': staticmethod(fake_solve)})()
        monkeypatch.setattr(SolverBasedFitness, 'set_adapter',
                            lambda self, net=None, pretrained_net=None:
                            setattr(self, 'adapter', adapter))
        primary = Discrepancy('solver_l2')
        solver_host = SolverBasedFitness(['penalty_coeff', 'pinn_loss_mult'],
                                         objectives=[primary, Instability()], primary=primary)
        solver_host.params = {'penalty_coeff': 0.2, 'pinn_loss_mult': 1.0}
        fitness = OperatorCondition(solver_host, _fitness_condition)
        sys_selector = self._system_selector(
            _real_selector(objectives=[Discrepancy('l2_relative')]))
        system = self._system(uv_pool)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        redraws, redrawn = self._script_the_redraw(monkeypatch, uv_pool)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sys_selector.apply(system, {})
        assert rps._rps_declined(eq_u) and not rps._rps_declined(eq_v)
        assert not eq_v.fitness_calculated

        fitness.apply(system, {})
        assert solves == []
        assert rps._rps_declined(eq_u) and _equation_is_degenerate(eq_u)
        assert eq_u.fitness_value == LOSS_NAN_VAL
        assert not eq_v.fitness_calculated
        assert has_degenerate_equation(system)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            regenerate_degenerate_equations(system, sys_selector, fitness, {}, {})
        assert redraws == [eq_u] and solves == [system]
        assert not has_degenerate_equation(system)
        assert {id(t) for t in eq_u.structure} == {id(t) for t in redrawn}
        for eq in (eq_u, eq_v):
            assert eq.fitness_calculated and eq.fitness_value < LOSS_NAN_VAL


DECLINED_U = f'0.5 * {U_DU} + 0.0 = {DU}'      # no right part: declined
FITTING_U = f'0.4 * {V_TOKEN} + 0.0 = {DU}'     # fits
#: Besides its target only the v-equation's target, which ``'assignment'`` and
#: ``'plain'`` mode remove: left unassigned there.
UNASSIGNED_U = f'0.5 * {DV} + 0.0 = {DU}'


def _fitted_state(eq):
    """What a fit and its scores leave on ``eq``, comparable with ``==``."""
    return ([id(t) for t in eq.structure], eq.target_idx,
            np.asarray(eq.weights_internal).tobytes(),
            np.asarray(eq.weights_final).tobytes(),
            eq.fitness_value, eq.coefficients_stability, eq.fitness_calculated)


class TestTheResample:
    """``regenerate_degenerate_equations`` redraws an equation right-part
    selection declined up to ``RESAMPLE_BUDGET`` times -- the redraw the
    selector used to make in place -- and any other degenerate equation
    within ``max_passes``, and re-fits nothing else of the system. A system
    with no degenerate equation it leaves untouched. On real
    fits: the u-equation of ``TestADeclinedCandidate._system`` beside the
    v-equation that fits; ``Equation.randomize`` is scripted."""

    @staticmethod
    def _wiring():
        """The live selector, the system selector over it and the chromosome
        fitness the MOEA/DD strategy wires (solver-free)."""
        from epde.operators.utils.operator_mappers import map_operator_between_levels
        selector = _real_selector()
        fitness = map_operator_between_levels(selector.suboperators['fitness_calculation'],
                                              'gene level', 'chromosome level',
                                              objective_condition=_fitness_condition)
        return selector, TestADeclinedCandidate._system_selector(selector), fitness

    @staticmethod
    def _script_redraws(monkeypatch, pool, texts):
        """The k-th ``Equation.randomize`` installs fresh terms of
        ``texts[k]`` (the last text for every later call); ``Term.randomize``
        is refused. Returns the redrawn equations, in order."""
        from epde.structure.main_structures import Equation, Term
        redraws = []

        def scripted_redraw(self):
            text = texts[min(len(redraws), len(texts) - 1)]
            redraws.append(self)
            self.reset_state(True)
            self.structure = _terms(pool, text)
            self._invalidate_label_cache()

        def refuse(self, *args, **kwargs):
            raise AssertionError('a term was redrawn')

        monkeypatch.setattr(Equation, 'randomize', scripted_redraw)
        monkeypatch.setattr(Term, 'randomize', refuse)
        return redraws

    @staticmethod
    def _spy_fits(monkeypatch):
        """The variables of the equations the term sweep fits, in order."""
        fits = []
        real_fit = rps.EqRightPartSelector._fit_and_score

        def spy(self, objective, subop_args):
            fits.append(objective.main_var_to_explain)
            return real_fit(self, objective, subop_args)

        monkeypatch.setattr(rps.EqRightPartSelector, '_fit_and_score', spy)
        return fits

    @staticmethod
    def _spy_in_place_scores(monkeypatch, selector):
        """The variables of the equations the host scores in place."""
        host = selector.suboperators['fitness_calculation']
        scored, real_score = [], host.apply

        def spy(objective, arguments, force_out_of_place=False):
            if not force_out_of_place:
                scored.append(objective.main_var_to_explain)
            return real_score(objective, arguments, force_out_of_place=force_out_of_place)

        monkeypatch.setattr(host, 'apply', spy)
        return scored

    @staticmethod
    def _spy_stats(monkeypatch):
        import epde.operators.multiobjective.moeadd_specific as ms
        records, stock = [], ms._loop_stats.record

        def spy(site, iters, cap):
            records.append((site, iters, cap))
            stock(site, iters, cap)

        monkeypatch.setattr(ms._loop_stats, 'record', spy)
        return records

    @staticmethod
    def _selected_and_scored(system, sys_selector, fitness):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sys_selector.apply(system, {})
            fitness.apply(system, {})

    @staticmethod
    def _regenerate(system, sys_selector, fitness, **kwargs):
        from epde.operators.multiobjective.moeadd_specific import (
            regenerate_degenerate_equations)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return regenerate_degenerate_equations(system, sys_selector, fitness,
                                                   {}, {}, **kwargs)

    def test_redraws_that_all_decline_spend_exactly_the_budget(self, uv_pool, monkeypatch):
        """Every redraw is declined again: the u-equation is redrawn exactly
        ``budget`` times and left stamped, for the caller to drop. Each
        declined redraw is stamped by the selector and the v-equation is
        already scored, so the chromosome fitness scores nothing, and the
        v-equation keeps its fit throughout."""
        from epde.operators.multiobjective.moeadd_specific import has_degenerate_equation
        selector, sys_selector, fitness = self._wiring()
        system = TestADeclinedCandidate._system(uv_pool)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        self._selected_and_scored(system, sys_selector, fitness)
        _assert_declined(eq_u)
        v_fit = _fitted_state(eq_v)

        redraws = self._script_redraws(monkeypatch, uv_pool, [DECLINED_U])
        scored = self._spy_in_place_scores(monkeypatch, selector)
        records = self._spy_stats(monkeypatch)
        assert self._regenerate(system, sys_selector, fitness, budget=4) is False

        assert redraws == [eq_u] * 4
        _assert_declined(eq_u)
        assert has_degenerate_equation(system)
        assert scored == []
        assert _fitted_state(eq_v) == v_fit
        assert ('regenerate_degenerate_equations.redraws', 4, 4) in records
        assert ('regenerate_degenerate_equations.exhausted', 1, 1) in records

    def test_the_default_budget_is_the_module_constant(self, uv_pool, monkeypatch):
        import epde.operators.multiobjective.moeadd_specific as ms
        selector, sys_selector, fitness = self._wiring()
        system = TestADeclinedCandidate._system(uv_pool)
        self._selected_and_scored(system, sys_selector, fitness)
        redraws = self._script_redraws(monkeypatch, uv_pool, [DECLINED_U])
        monkeypatch.setattr(ms, 'RESAMPLE_BUDGET', 5)
        assert self._regenerate(system, sys_selector, fitness) is False
        assert len(redraws) == 5

    def test_a_redraw_accepted_within_the_budget_is_placed(self, uv_pool, monkeypatch):
        """Two redraws declined, the third fits: three redraws, the system
        leaves with no degenerate equation, scored."""
        from epde.operators.common.objectives import LOSS_NAN_VAL
        from epde.operators.multiobjective.moeadd_specific import has_degenerate_equation
        selector, sys_selector, fitness = self._wiring()
        system = TestADeclinedCandidate._system(uv_pool)
        eq_u = system.vals['u']
        self._selected_and_scored(system, sys_selector, fitness)
        redraws = self._script_redraws(monkeypatch, uv_pool,
                                       [DECLINED_U, DECLINED_U, FITTING_U])
        records = self._spy_stats(monkeypatch)
        assert self._regenerate(system, sys_selector, fitness, budget=4) is True

        assert redraws == [eq_u] * 3
        assert not has_degenerate_equation(system)
        assert eq_u.fitness_calculated and eq_u.fitness_value < LOSS_NAN_VAL
        assert ('regenerate_degenerate_equations.redraws', 3, 4) in records
        assert ('regenerate_degenerate_equations.resampled', 1, 1) in records

    def test_a_fit_the_host_stamps_keeps_the_pass_budget(self, uv_pool, monkeypatch):
        """A degenerate equation selection did NOT decline -- the fit was
        placed and the fitness host stamped it (here every time; a failed
        solve does that to a whole system) -- shares ``max_passes`` redraw
        passes, as before the decline budget: each pass can cost a solve."""
        from epde.operators.common.objectives import LOSS_NAN_VAL
        selector, sys_selector, fitness = self._wiring()
        system = TestADeclinedCandidate._system(uv_pool)
        eq_u = system.vals['u']
        redraws = self._script_redraws(monkeypatch, uv_pool, [FITTING_U])
        eq_u.randomize()

        class StampingFitness:
            """The chromosome fitness, then the stamp on the u-equation."""
            def apply(self, objective, arguments):
                fitness.apply(objective, arguments)
                objective.vals['u'].fitness_value = LOSS_NAN_VAL

        stamping = StampingFitness()
        self._selected_and_scored(system, sys_selector, stamping)
        assert np.any(eq_u.weights_internal)         # a fit, not a decline
        del redraws[:]
        records = self._spy_stats(monkeypatch)
        assert self._regenerate(system, sys_selector, stamping, max_passes=2,
                                budget=10) is False
        assert redraws == [eq_u] * 2
        assert ('regenerate_degenerate_equations.passes', 2, 2) in records
        assert not any(site == 'regenerate_degenerate_equations.redraws'
                       for site, *_ in records)

    @pytest.mark.parametrize('mode', ['assignment', 'plain'])
    def test_an_unassigned_equation_keeps_the_pass_budget(self, uv_pool, monkeypatch,
                                                          mode):
        """In ``'assignment'`` and ``'plain'`` mode the system-level choice
        leaves an equation it cannot fit unassigned (``_mark_unassignable``):
        here the u-equation's one term besides its target is the v-equation's
        target, which both modes remove. Its stamp is a decline's -- every
        weight zero -- but nothing was declined, and the decline budget is
        ``'repair'`` mode's (``test_redraws_that_all_decline_spend_exactly_
        the_budget``): the equation shares ``max_passes`` redraw passes, as
        before that budget."""
        from epde.operators.multiobjective.moeadd_specific import (
            _declined_by_selection, has_degenerate_equation)
        with using_config(system_rps_mode=mode):
            selector, sys_selector, fitness = self._wiring()
            system = TestADeclinedCandidate._system(uv_pool, UNASSIGNED_U)
            eq_u = system.vals['u']
            self._selected_and_scored(system, sys_selector, fitness)
            assert _declined_by_selection(eq_u)

            redraws = self._script_redraws(monkeypatch, uv_pool, [UNASSIGNED_U])
            records = self._spy_stats(monkeypatch)
            assert self._regenerate(system, sys_selector, fitness, max_passes=2,
                                    budget=5) is False
        assert redraws == [eq_u] * 2
        assert has_degenerate_equation(system)
        assert ('regenerate_degenerate_equations.passes', 2, 2) in records
        assert not any(site == 'regenerate_degenerate_equations.redraws'
                       for site, *_ in records)

    def test_a_stamped_fit_is_refitted_alone(self, uv_pool, monkeypatch):
        """The resample keeps the fits of the equations still fitted for a
        fit the host stamped as for a decline: the u-equation fits, the host
        stamps it once, and its redraw is the only equation the term sweep
        fits. The v-equation keeps its fit, bit for bit -- the regeneration
        before the decline budget selected it again, on its pruned
        structure."""
        from epde.operators.common.objectives import LOSS_NAN_VAL
        selector, sys_selector, fitness = self._wiring()
        system = TestADeclinedCandidate._system(uv_pool, FITTING_U)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        stamped = []

        class StampingOnce:
            """The chromosome fitness, then -- the first time only -- the
            stamp on the u-equation."""
            def apply(self, objective, arguments):
                fitness.apply(objective, arguments)
                if not stamped:
                    stamped.append(objective.vals['u'])
                    objective.vals['u'].fitness_value = LOSS_NAN_VAL

        stamping = StampingOnce()
        self._selected_and_scored(system, sys_selector, stamping)
        assert stamped == [eq_u] and np.any(eq_u.weights_internal)   # not a decline
        v_fit = _fitted_state(eq_v)

        redraws = self._script_redraws(monkeypatch, uv_pool, [FITTING_U])
        fits = self._spy_fits(monkeypatch)
        assert self._regenerate(system, sys_selector, stamping) is True
        assert redraws == [eq_u]
        assert fits and set(fits) == {'u'}
        assert _fitted_state(eq_v) == v_fit

    def test_the_fitted_equations_are_not_fitted_again(self, uv_pool, monkeypatch):
        """Only the redrawn u-equation goes through the term sweep; the
        v-equation keeps its fit. The chromosome fitness re-scores it in
        place (the condition sits on the system), and as the host is a pure
        scorer its values come back bit for bit."""
        selector, sys_selector, fitness = self._wiring()
        system = TestADeclinedCandidate._system(uv_pool)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        self._selected_and_scored(system, sys_selector, fitness)
        v_fit = _fitted_state(eq_v)

        redraws = self._script_redraws(monkeypatch, uv_pool, [FITTING_U])
        fits = self._spy_fits(monkeypatch)
        assert self._regenerate(system, sys_selector, fitness) is True

        assert redraws == [eq_u]
        assert fits and set(fits) == {'u'}
        assert _fitted_state(eq_v) == v_fit

    def test_outside_a_resample_every_equation_is_selected_again(self, uv_pool,
                                                                 monkeypatch):
        """The scope changes the resample only: the ordinary system selection
        (a mutated offspring, a new candidate) still selects every equation of
        the system, fitted or not."""
        selector, sys_selector, fitness = self._wiring()
        system = TestADeclinedCandidate._system(uv_pool)
        self._script_redraws(monkeypatch, uv_pool, [FITTING_U])
        system.vals['u'].randomize()
        self._selected_and_scored(system, sys_selector, fitness)

        fits = self._spy_fits(monkeypatch)
        for scoped in (False, True):
            del fits[:]
            system.vals['u'].right_part_selected = False
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if scoped:
                    with rps.resample_scope():
                        sys_selector.apply(system, {})
                else:
                    sys_selector.apply(system, {})
            assert set(fits) == ({'u'} if scoped else {'u', 'v'}), scoped

    def test_a_system_with_nothing_degenerate_is_untouched(self, uv_pool, monkeypatch):
        """No declines: no draw, no selection, no scoring -- the random
        stream and every fit are what the selection and fitness left."""
        selector, sys_selector, fitness = self._wiring()
        system = TestADeclinedCandidate._system(uv_pool)
        self._script_redraws(monkeypatch, uv_pool, [FITTING_U])
        system.vals['u'].randomize()
        self._selected_and_scored(system, sys_selector, fitness)
        before = [_fitted_state(eq) for eq in system]
        calls = []

        class Spy:
            def __init__(self, name, inner):
                self.name, self.inner = name, inner

            def apply(self, objective, arguments):
                calls.append(self.name)
                return self.inner.apply(objective, arguments)

        np.random.seed(7)
        state = _random_state()
        assert self._regenerate(system, Spy('rps', sys_selector),
                                Spy('fitness', fitness)) is True
        assert calls == []
        assert _random_state() == state
        assert [_fitted_state(eq) for eq in system] == before

    def test_the_single_objective_selector_redraws_what_it_declines(self, uv_pool,
                                                                    monkeypatch):
        """``so_specific.ResamplingRightPartSelector``: the system selector
        declines the u-equation, the candidate is scored and handed to the
        shared resample, and the redraw is placed."""
        from epde.operators.multiobjective.moeadd_specific import has_degenerate_equation
        from epde.operators.singleobjective.so_specific import ResamplingRightPartSelector
        selector, sys_selector, fitness = self._wiring()
        resampling = ResamplingRightPartSelector()
        resampling.set_suboperators({'right_part_selector': sys_selector,
                                     'chromosome_fitness': fitness})
        system = TestADeclinedCandidate._system(uv_pool)
        redraws = self._script_redraws(monkeypatch, uv_pool, [FITTING_U])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            resampling.apply(system, {})
        assert redraws == [system.vals['u']]
        assert not has_degenerate_equation(system)

    def test_the_single_objective_selector_never_redraws_a_stamped_fit(self, uv_pool,
                                                                       monkeypatch):
        """``max_passes=0``: beside the declined u-equation the fitness host
        stamps the v-equation's fit. The single-objective search never
        regenerated a stamped fit, so it stays, and with it the stamp: the
        candidate sorts last whatever the u-equation's redraws give. It is
        given up at once, with no redraw, and the loop stats record it
        exhausted."""
        from epde.operators.common.objectives import LOSS_NAN_VAL
        from epde.operators.singleobjective.so_specific import ResamplingRightPartSelector
        from epde.optimizers.single_criterion.supplementary import is_stamped
        selector, sys_selector, fitness = self._wiring()
        real_apply = fitness.apply

        def stamping_apply(objective, arguments):
            """The chromosome fitness, then the stamp on the v-equation."""
            real_apply(objective, arguments)
            objective.vals['v'].fitness_value = LOSS_NAN_VAL

        monkeypatch.setattr(fitness, 'apply', stamping_apply)
        resampling = ResamplingRightPartSelector()
        resampling.set_suboperators({'right_part_selector': sys_selector,
                                     'chromosome_fitness': fitness})
        system = TestADeclinedCandidate._system(uv_pool)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        redraws = self._script_redraws(monkeypatch, uv_pool, [FITTING_U])
        records = self._spy_stats(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            resampling.apply(system, {})

        assert redraws == []
        _assert_declined(eq_u)
        assert np.any(eq_v.weights_internal)          # a fit, not a decline
        assert is_stamped(system)
        assert ('regenerate_degenerate_equations.exhausted', 1, 1) in records

    def test_the_single_objective_selector_without_a_decline_is_the_selector(
            self, uv_pool, monkeypatch):
        """No decline: bit for bit what the bare system selector leaves, the
        random stream included, and the candidate is left unscored for the
        population's fitness block."""
        from epde.operators.singleobjective.so_specific import ResamplingRightPartSelector
        selector, sys_selector, fitness = self._wiring()
        resampling = ResamplingRightPartSelector()
        resampling.set_suboperators({'right_part_selector': sys_selector,
                                     'chromosome_fitness': fitness})
        system = TestADeclinedCandidate._system(uv_pool)
        self._script_redraws(monkeypatch, uv_pool, [FITTING_U])
        system.vals['u'].randomize()
        runs = []
        for operator in (sys_selector, resampling):
            candidate = copy.deepcopy(system)
            np.random.seed(11)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                operator.apply(candidate, {})
            runs.append(([[t.name for t in eq.structure] for eq in candidate],
                         [np.asarray(eq.weights_final).tobytes() for eq in candidate],
                         [eq.fitness_calculated for eq in candidate], _random_state()))
        assert runs[0] == runs[1]
        assert runs[1][2] == [False, False]


# ---------------------------------------------------------------------------
# Common-factor cancellation never leaves the structure without a derivative.
# ---------------------------------------------------------------------------

class TestCancellation:

    def test_stripping_every_derivative_declines(self, uv_pool, monkeypatch, no_draws):
        """``u*du = c*v*du`` shares ``du`` with its target at the lowest power:
        cancelling it leaves ``u = c*v``, and no term with a derivative of u.
        Declined before anything is reduced -- it used to be reduced, and the
        next pass handed to ``restore_property`` for a random derivative
        term."""
        eq = _translated_u_equation(uv_pool, f'0.5 * {V_DU} + 0.0 = {U_DU}')
        before = [(t, t.name) for t in eq.structure]
        _refuse_restore(monkeypatch)
        _stub_fit_by_target(monkeypatch, {U_DU})
        _apply(eq)

        _assert_declined(eq)
        assert [(t, t.name) for t in eq.structure] == before

    def test_a_cancellation_keeping_the_derivative_still_runs(self, uv_pool, monkeypatch,
                                                              no_draws):
        """The control: ``u*du = c*u*v`` sheds the common ``u`` into
        ``du = c*v``, a right part, on the same Term objects."""
        eq = _translated_u_equation(uv_pool, f'0.5 * {U_TOKEN} * {V_TOKEN} + 0.0 = {U_DU}')
        before = list(eq.structure)
        _stub_fit_by_target(monkeypatch, {U_DU, DU})
        _apply(eq)

        assert not rps._rps_declined(eq)
        assert eq.target.name == DU
        assert sorted(t.name for t in eq.structure) == sorted([V_TOKEN, DU])
        assert {id(t) for t in eq.structure} == {id(t) for t in before}

    def test_a_target_shedding_its_derivative_beside_another_is_reselected(
            self, uv_pool, monkeypatch, no_draws):
        """``u*du = c*v*du`` again, now with a zero-weight ``du`` beside it.
        Cancelling ``du`` strips it from both fitted terms, but ``du`` itself
        is untouched and still carries one: no decline and no restore. The
        next pass selects ``du`` among the reduced terms -- the same Term
        objects -- as the old code did without drawing."""
        eq = _translated_u_equation(uv_pool,
                                    f'0.5 * {V_DU} + 0.0 * {DU} + 0.0 = {U_DU}')
        before = list(eq.structure)
        _refuse_restore(monkeypatch)
        # The sweep's verdicts: ``u*du`` beats ``du`` while both stand, and
        # ``du`` is the only feature the support leaves out.
        calls = _stub_fit_scored(monkeypatch, {U_DU: 0.3, DU: 0.5}, zero={DU})
        _apply(eq)

        assert not rps._rps_declined(eq)
        assert calls == [V_DU, DU, U_DU, DU]
        assert eq.target.name == DU
        assert sorted(t.name for t in eq.structure) == sorted([V_TOKEN, U_TOKEN, DU])
        assert {id(t) for t in eq.structure} == {id(t) for t in before}


# ---------------------------------------------------------------------------
# The linear-dependence scrub declines before it drops.
# ---------------------------------------------------------------------------

class TestTheLinearDependenceScrub:
    """``_drop_dependent_terms`` on the stand-ins of
    test_lin_dep_scrub_multisample.py, whose columns are exactly dependent by
    construction, in both trajectories."""

    def test_flags_below_two_terms_decline_undropped(self):
        """``c`` and ``2c`` beside the target: both columns are constant, in
        the intercept's span, so both are flagged, and dropping both would
        leave the target alone. Declined before anything is dropped -- the
        scrub used to drop ``c``, refuse ``2c`` at the floor and leave
        ``[2c, target]`` declined, a label that is not the candidate's."""
        const = FakeTerm('c', _cols(np.full(N0, 2.0), np.full(N1, 2.0)))
        double = FakeTerm('2c', _cols(np.full(N0, 4.0), np.full(N1, 4.0)))
        target = FakeTerm('target', _independent_pair(99))
        labels = [term.factors_labels for term in (const, double, target)]

        verdict, equation = _run([const, double], target)

        assert verdict == rps.DECLINED
        assert len(equation.structure) == 3
        assert all(term is was for term, was in zip(equation.structure, (const, double, target)))
        assert [term.factors_labels for term in equation.structure] == labels
        assert equation.resets == 0

    def test_flags_above_two_terms_are_dropped(self):
        """The control: ``a``, ``2a`` and ``c`` beside the target. ``2a`` lies
        in the span of the intercept and ``a``, ``c`` in the intercept's:
        both go, which leaves ``a`` and the target, the same objects."""
        free = FakeTerm('a', _independent_pair(3))
        double = FakeTerm('2a', {key: 2.0 * col for key, col in free.evaluate().items()})
        const = FakeTerm('c', _cols(np.full(N0, 2.0), np.full(N1, 2.0)))
        target = FakeTerm('target', _independent_pair(99))

        verdict, equation = _run([free, double, const], target)

        assert verdict is True
        assert len(equation.structure) == 2
        assert equation.structure[0] is free and equation.structure[1] is target
        assert equation.resets == 1


# ---------------------------------------------------------------------------
# The outer cap.
# ---------------------------------------------------------------------------

class TestTheOuterCap:

    def test_a_form_still_being_rewritten_is_declined(self, uv_pool, monkeypatch, no_draws):
        """``simplify_equation`` reports a rewrite on every pass, so the cap is
        reached: the candidate is declined -- never let out as it stood, the
        state the cap break used to accept."""
        monkeypatch.setattr(rps, 'RPS_OUTER_MAX_ITER', 3)
        passes = []

        def endless_rewrite(self, objective):
            passes.append(objective.target.name)
            objective.reset_for_structure_change()
            return True

        monkeypatch.setattr(rps.EqRightPartSelector, 'simplify_equation', endless_rewrite)
        eq = _translated_u_equation(uv_pool, f'0.5 * {V_TOKEN} + 0.0 = {DU}')
        _stub_fit_by_target(monkeypatch, {DU})
        messages = _apply(eq)

        assert passes == [DU] * 3
        assert any('declining the candidate' in m for m in messages), messages
        _assert_declined(eq)
        assert eq.target.contains_deriv('u')

    def test_a_derivative_free_target_never_leaves_accepted(self, uv_pool, monkeypatch,
                                                            no_draws):
        """The cap break used to let a target with no derivative of u out with
        finite objectives (``u = 0.1075 x_0^2``). A pass that strips the
        derivative now ends declined, whichever way the next pass goes -- and
        with the derivative back in the target: the in-place strip is undone,
        as the cap declines the candidate as it came in."""
        monkeypatch.setattr(rps, 'RPS_OUTER_MAX_ITER', 1)

        def stripping_rewrite(self, objective):
            target = objective.target
            target.structure = [f for f in target.structure
                                if not f.cache_label[0].startswith('du/')]
            target.resetSavedState()
            objective.reset_for_structure_change()
            return True

        monkeypatch.setattr(rps.EqRightPartSelector, 'simplify_equation', stripping_rewrite)
        eq = _translated_u_equation(uv_pool, f'0.5 * {V_TOKEN} + 0.0 = {U_DU}')
        before = _objects(eq)
        _stub_fit_by_target(monkeypatch, {U_DU})
        _apply(eq)

        _assert_declined(eq)
        assert _same_objects(eq, before)
        assert eq.target.name == U_DU and eq.target.contains_deriv('u')

    def test_a_capped_candidate_is_declined_as_it_came_in(self, uv_pool, monkeypatch,
                                                         no_draws):
        """``u*du = a*u*v + b*du`` converges in three passes: the ratio-fit
        scrub drops ``du`` (a component of the target), the cancellation
        sheds ``u`` IN PLACE (``u*v -> v``, ``u*du -> du``, the power of each
        ``u`` lowered to 0) and ``du = c*v`` is placed. With a two-pass
        budget the cap declines it after both rewrites, with the structure
        it came in with: the same Term and Factor objects, the powers
        restored, the entry label. It used to be declined as ``[v, du]``, a
        label that came back with a fresh budget and was placed as
        ``du = c*v``; the entry label comes back to the same two rewrites and
        is declined at the cap again. With the full budget it converges, on
        the same Term objects."""
        budget = rps.RPS_OUTER_MAX_ITER
        monkeypatch.setattr(rps, 'RPS_OUTER_MAX_ITER', 2)
        eq = _translated_u_equation(uv_pool,
                                    f'0.5 * {U_TOKEN} * {V_TOKEN} + 0.3 * {DU} + 0.0 = {U_DU}')
        before, names, label = _objects(eq), [t.name for t in eq.structure], eq.terms_labels
        _refuse_restore(monkeypatch)
        # The sweep's verdicts: ``u*du`` beats ``du`` while both stand.
        calls = _stub_fit_scored(monkeypatch, {U_DU: 0.3, DU: 0.5})
        for _ in range(2):
            eq.right_part_selected = False
            del calls[:]
            messages = _apply(eq)

            # Two passes, each a rewrite: the scrub, then the cancellation.
            assert calls == [DU, U_DU, U_DU]
            assert any('declining the candidate' in m for m in messages), messages
            _assert_declined(eq)
            assert _same_objects(eq, before)
            assert [t.name for t in eq.structure] == names
            assert eq.terms_labels == label

        monkeypatch.setattr(rps, 'RPS_OUTER_MAX_ITER', budget)
        eq.right_part_selected = False
        _apply(eq)
        assert not rps._rps_declined(eq)
        assert eq.target.name == DU
        assert sorted(t.name for t in eq.structure) == sorted([V_TOKEN, DU])
        assert all(any(t is term for term, _ in before) for t in eq.structure)

    def test_the_cap_keeps_the_term_the_entry_validator_drew(self, uv_pool, monkeypatch):
        """``u*v = c*u`` has no derivative of u, so the entry validator draws
        a term with one and installs it -- the one draw right-part selection
        makes. Every pass then rewrites (a stand-in chain: ``u`` shed from
        ``u*v`` in place, then the term ``u`` dropped), and with a two-pass
        budget the cap declines the candidate. It is put back as the
        validator left it: the same Term and Factor objects and label, the
        drawn term its placeholder target, after both visits. So the label
        comes back to the same passes and is declined at the cap again, with
        no second validator call and no draw. Put back as it was called, with
        no derivative term, it would come back to the validator, which would
        draw anew on every visit -- and one of the draws could be placed. The
        term rewritten in place leaves with no saved evaluation of its
        rewritten form."""
        from epde.structure.main_structures import Equation, Term
        monkeypatch.setattr(rps, 'RPS_OUTER_MAX_ITER', 2)
        eq = _translated_u_equation(uv_pool, f'0.5 * {U_TOKEN} * {V_TOKEN} + 0.0 = {U_TOKEN}')
        # Room for a third term: the validator adds its draw, replacing none.
        eq.metaparameters['max_terms_number']['value'] = 3
        uv_term, u_term = eq.structure
        validated, draws, passes = [], [], []
        # What the class holds now, so a wrapper already around the validator
        # stays in the call chain.
        real_restore, real_draw = Equation.restore_property, Term.randomize

        def spy_restore(self, **kwargs):
            real_restore(self, **kwargs)
            validated.append((_objects(self), self.terms_labels, len(draws)))

        def spy_draw(self, *args, **kwargs):
            draws.append(self)
            return real_draw(self, *args, **kwargs)

        def rewrite(self, objective):
            passes.append(objective.target)
            if len(uv_term.structure) == 2:
                # ``u`` shed from ``u*v`` in place, as the cancellation does.
                uv_term.structure = [f for f in uv_term.structure if f.variable != 'u']
                uv_term.resetSavedState()
            else:
                # The term ``u`` dropped. This pass evaluated the reduced
                # ``u*v``: its evaluation saved under the reduced label, as
                # ``Term.evaluate`` saves it on a cache miss.
                uv_term.saved[False], uv_term.saved_as[False] = True, uv_term.factors_labels
                objective.structure = [t for t in objective.structure if t is not u_term]
                objective._invalidate_label_cache()
            objective.reset_for_structure_change()
            return True

        monkeypatch.setattr(Equation, 'restore_property', spy_restore)
        monkeypatch.setattr(Term, 'randomize', spy_draw)
        monkeypatch.setattr(rps.EqRightPartSelector, 'simplify_equation', rewrite)
        _stub_fit(monkeypatch, {'now': False})
        np.random.seed(0)
        for _ in range(2):
            eq.right_part_selected = False
            del passes[:]
            messages = _apply(eq)

            # One validator call, on the first visit, and no draw outside it.
            assert len(validated) == 1
            objects, label, drawn = validated[0]
            assert len(draws) == drawn
            assert len(objects) == 3
            assert objects[0][0] is uv_term and objects[1][0] is u_term
            installed = objects[2][0]
            # Two passes, each a rewrite, both on the drawn target.
            assert len(passes) == 2 and all(t is installed for t in passes)
            assert any('declining the candidate' in m for m in messages), messages
            _assert_declined(eq)
            assert _same_objects(eq, objects) and eq.terms_labels == label
            assert eq.target is installed and installed.contains_deriv('u')
            assert uv_term.saved_as[False] in (None, uv_term.factors_labels)

    def test_a_form_simplify_would_rewrite_is_never_placed(self, uv_pool, monkeypatch,
                                                           no_draws):
        """With a one-pass budget the ratio-fit scrub's rewrite cannot be
        selected again: the candidate is declined, not placed carrying the
        multiple of its target (``5.715 du/dx0 = du/dx0*x_0^2``)."""
        monkeypatch.setattr(rps, 'RPS_OUTER_MAX_ITER', 1)
        eq = _translated_u_equation(uv_pool, f'0.5 * {V_TOKEN} + 0.3 * {U_DU} + 0.0 = {DU}')
        _stub_fit_by_target(monkeypatch, {DU})
        _apply(eq)

        _assert_declined(eq)


# ---------------------------------------------------------------------------
# A declined equation and the system's rules.
# ---------------------------------------------------------------------------

class TestTheSystemRules:

    def test_a_declined_equation_reserves_and_carries_nothing(self, uv_pool):
        """The u-equation is declined with the placeholder target ``du``; the
        v-equation holds ``du`` as an active term. Neither direction is a
        leak, and the two are no duplicate law: the placeholder must not
        reroll a healthy equation's term ahead of the redraw it waits for."""
        from epde.interface.equation_translator import translate_equation
        system = translate_equation({'u': f'0.5 * {V_TOKEN} + 0.0 = {DU}',
                                     'v': f'0.5 * {DU} + 0.0 = {DV}'},
                                    uv_pool, all_vars=['u', 'v'])
        eq_u, eq_v = system.vals['u'], system.vals['v']
        for eq, var in ((eq_u, 'u'), (eq_v, 'v')):
            eq.main_var_to_explain = var
            w = np.array([0.5, 0.0])
            eq.weights_internal, eq.weights_final = w, w.copy()
            eq.weights_internal_evald = eq.weights_final_evald = True
            eq.target_idx = len(eq.structure) - 1
        assert rps._carries_target_of(eq_u, eq_v)          # the healthy baseline

        selector = _selector()
        selector._decline(eq_u, 'test', 1)
        assert rps._rps_declined(eq_u) and not rps._rps_declined(eq_v)
        assert eq_u.target.name == DU
        assert rps._reserved_target_signature(eq_u) is None
        assert not rps._carries_target_of(eq_u, eq_v)
        assert not rps._carries_target_of(eq_v, eq_u)
        assert not rps._equations_share_active_structure(eq_u, eq_v)


# ---------------------------------------------------------------------------
# amplification_ratio on a fit with nothing in it.
# ---------------------------------------------------------------------------

class TestAmplificationRatio:

    def test_all_zero_weights_and_intercept_give_zero(self, uv_pool):
        """No nonzero term and no intercept: nothing reaches the target, so the
        ratio is the empty sum, 0.0 -- it used to raise ``AssertionError
        ('Mismatching keys of dicts inputs into dictDot')``."""
        eq = _u_equation(uv_pool, f'0.0 * {V_TOKEN} + 0.0 = {DU}', [0.0, 0.0])
        assert rps.amplification_ratio(eq) == 0.0

    def test_the_nonzero_cases_are_unchanged(self, uv_pool):
        """A term, and the intercept alone, keep the ``|c|*||col|| / ||target||``
        rule, the intercept's column the ones-vector."""
        eq = _u_equation(uv_pool, f'0.5 * {V_TOKEN} + 0.0 = {DU}', [0.5, 0.0])
        target = {k: np.asarray(v, dtype=float).reshape(-1)
                  for k, v in eq.target.evaluate().items()}
        feature = {k: np.asarray(v, dtype=float).reshape(-1)
                   for k, v in eq.structure[0].evaluate().items()}
        expected = np.mean([0.5 * np.linalg.norm(feature[k]) / np.linalg.norm(target[k])
                            for k in target])
        assert rps.amplification_ratio(eq) == pytest.approx(expected, rel=1e-12)

        eq = _u_equation(uv_pool, f'0.0 * {V_TOKEN} + 0.0 = {DU}', [0.0, 0.25])
        expected = np.mean([0.25 * np.sqrt(target[k].size) / np.linalg.norm(target[k])
                            for k in target])
        assert rps.amplification_ratio(eq) == pytest.approx(expected, rel=1e-12)
