#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The duplicate-equation guard of ``SoEqRightPartSelector``.

Its one cross-equation invariant: no two equations of a system may share an
IDENTICAL active structure (target plus nonzero-weight terms, powers kept) --
that is one law rearranged, and the system is left with fewer independent
equations than variables. The later equation of such a pair has one shared
term rerolled and its right part re-selected.

Nothing else is constrained across equations. A strict superset is coupling,
and so is one equation carrying another's TARGET as a standalone term: the
stricter target-term uniqueness rule that used to ban that is retired, and a
negative pin below keeps it from creeping back.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import epde.operators.common.right_part_selection as rps
from epde.operators.common.right_part_selection import (
    _break_equation_duplication, _equations_share_active_structure)

# The translator gives every term one factor at most; regenerated terms need
# the live two-factor budget the searches run with.
_LIVE_FACTORS = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
DU = 'du/dx0{power: 1.0}'
DV = 'dv/dx0{power: 1.0}'
U_TOKEN = 'u{power: 1.0}'
V_TOKEN = 'v{power: 1.0}'


def _search_pool(fields):
    import epde
    search = epde.EpdeSearch(use_solver=False,
                             verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 4 * np.pi, 200)
    _, domain = search.createDomain(t, boundary_width=10, ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory(
        {var: make(t) for var, make in fields.items()}, domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1)
    return search.pool


@pytest.fixture(scope='module')
def uv_pool():
    return _search_pool({'u': np.sin,
                         'v': lambda t: np.cos(t) + 0.3 * np.sin(2 * t)})


@pytest.fixture(scope='module')
def uvw_pool():
    """Three variables, for the translator's per-equation variable binding."""
    return _search_pool({'u': np.sin,
                         'v': lambda t: np.cos(t) + 0.3 * np.sin(2 * t),
                         'w': lambda t: np.sin(3 * t) + 0.5})


def _system(pool, texts, weights):
    """``texts`` translated on ``pool`` and pinned to the fitted state the
    forward RPS pass leaves: target last, the given weights evaluated, the
    live factor budget."""
    from epde.interface.equation_translator import translate_equation
    system = translate_equation(texts, pool, all_vars=list(texts))
    for var in texts:
        eq = system.vals[var]
        eq.main_var_to_explain = var
        eq.metaparameters['max_factors_in_term']['value'] = dict(_LIVE_FACTORS)
        for term in eq.structure:
            term.max_factors_in_term = dict(_LIVE_FACTORS)
        w = np.array(weights[var], dtype=float)
        eq.weights_internal = w
        eq.weights_final = w.copy()
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        eq.target_idx = len(eq.structure) - 1
    return system


def _soeq_selector():
    """The chromosome-level selector with a per-equation selector that
    records its calls and changes nothing, so the guard's repair is the only
    thing acting. Its ``apply`` takes no extra keyword: a call that still
    handed the per-equation selection a system ban would raise here."""
    from epde.operators.common.right_part_selection import SoEqRightPartSelector
    from epde.operators.utils.template import CompoundOperator

    class RecordingSelector(CompoundOperator):
        key = 'RecordingSelector'

        def __init__(self):
            super().__init__()
            self.calls = []

        def apply(self, objective, arguments):
            self.calls.append(objective.main_var_to_explain)

        def use_default_tags(self):
            self._tags = {'custom level', 'no suboperators', 'inplace'}

    inner = RecordingSelector()
    selector = SoEqRightPartSelector()
    selector.set_suboperators({'eq_right_part_selector': inner})
    return selector, inner


def _run(selector, system):
    """Apply ``selector`` and return the warning messages it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        selector.apply(system, {})
    return [str(w.message) for w in caught]


def _scripted_term_draws(monkeypatch, pool, labels):
    """``Term.randomize`` draws ``labels`` in turn (the last one repeats) and
    records which term it was called on."""
    from epde.structure.main_structures import Term
    made = []

    def scripted_randomize(self, *args, **kwargs):
        made.append(self)
        label = labels[min(len(made), len(labels)) - 1]
        self.structure = [pool.create(label=label)[1]]

    monkeypatch.setattr(Term, 'randomize', scripted_randomize)
    return made


def _names(eq):
    return [term.name for term in eq.structure]


# ---------------------------------------------------------------------------
# The predicate.
# ---------------------------------------------------------------------------

class TestThePredicate:

    @pytest.mark.parametrize('texts, weights, shared', [
        # The same law solved for two quantities.
        ({'u': f'0.5 * {DV} + 0.0 = {DU}', 'v': f'2.0 * {DU} + 0.0 = {DV}'},
         {'u': [0.5, 0.0], 'v': [2.0, 0.0]}, True),
        # u's active set is a strict superset of v's: coupling, not a duplicate.
        ({'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
          'v': f'2.0 * {DU} + 0.0 = {DV}'},
         {'u': [0.5, 0.3, 0.0], 'v': [2.0, 0.0]}, False),
        # A zero-weight padding term is not part of the fitted law.
        ({'u': f'0.5 * {DV} + 0.0 * {V_TOKEN} + 0.0 = {DU}',
          'v': f'2.0 * {DU} + 0.0 = {DV}'},
         {'u': [0.5, 0.0, 0.0], 'v': [2.0, 0.0]}, True),
        # Powers are kept: u^2 and u are different terms.
        ({'u': f'0.5 * u{{power: 2.0}} + 0.0 = {DU}',
          'v': f'0.7 * {U_TOKEN} + 0.0 = {DU}'},
         {'u': [0.5, 0.0], 'v': [0.7, 0.0]}, False),
    ], ids=['identical', 'superset', 'zero-weight-padding', 'power-distinct'])
    def test_exact_equality_of_the_active_structures(self, uv_pool, texts,
                                                     weights, shared):
        system = _system(uv_pool, texts, weights)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        assert _equations_share_active_structure(eq_u, eq_v) is shared
        assert _equations_share_active_structure(eq_v, eq_u) is shared

    def test_an_empty_active_structure_is_never_a_duplicate(self):
        empty = SimpleNamespace(active_terms_labels=frozenset())
        assert _equations_share_active_structure(empty, empty) is False


# ---------------------------------------------------------------------------
# The repair inside SoEqRightPartSelector.
# ---------------------------------------------------------------------------

class TestTheGuard:

    def test_only_the_later_equation_is_repaired(self, uv_pool, monkeypatch):
        """``0.5 dv/dx0 = du/dx0`` and ``2 du/dx0 = dv/dx0`` are one law. The
        v-equation's shared feature is rerolled and v alone is re-selected."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.0 = {DU}',
                          'v': f'2.0 * {DU} + 0.0 = {DV}'},
                         {'u': [0.5, 0.0], 'v': [2.0, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        u_before = _names(eq_u)
        v_feature, v_target = eq_v.structure
        made = _scripted_term_draws(monkeypatch, uv_pool, ['u'])

        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert made == [v_feature]
        assert v_feature.name == U_TOKEN and eq_v.target is v_target
        assert _names(eq_u) == u_before
        # Forward pass on both equations, then one re-selection of the repaired one.
        assert inner.calls == ['u', 'v', 'v']
        assert not _equations_share_active_structure(eq_u, eq_v)
        assert not any('duplicate-equation guard' in m for m in messages), messages

    def test_a_redraw_back_onto_the_shared_signature_is_rejected(self, uv_pool,
                                                                 monkeypatch):
        """The acceptance rule. The first redraw reproduces the rerolled term's
        own original ``du/dx0`` -- unique within the v-equation, so only the
        "leave ``shared_sigs``" half of the rule can refuse it, and accepting
        it would hand back the very duplicate being broken. It is redrawn, and
        the repair still costs one re-selection."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.0 = {DU}',
                          'v': f'2.0 * {DU} + 0.0 = {DV}'},
                         {'u': [0.5, 0.0], 'v': [2.0, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        v_feature, v_target = eq_v.structure
        assert v_feature.factors_labels in eq_u.active_terms_labels
        made = _scripted_term_draws(monkeypatch, uv_pool, ['du/dx0', 'u'])

        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert made == [v_feature, v_feature]
        assert v_feature.name == U_TOKEN and eq_v.target is v_target
        assert inner.calls == ['u', 'v', 'v']
        assert not _equations_share_active_structure(eq_u, eq_v)
        assert not any('duplicate-equation guard' in m for m in messages), messages

    def test_a_repair_drops_the_stale_fit_but_keeps_the_target(self, uv_pool,
                                                              monkeypatch):
        """After a True return the v-equation's weights describe a term set it
        no longer has, so they must be gone before the caller re-selects --
        the mirror of the no-change case below, where the fit stays. The
        installed target survives (``reset_for_structure_change``)."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.0 = {DU}',
                          'v': f'2.0 * {DU} + 0.0 = {DV}'},
                         {'u': [0.5, 0.0], 'v': [2.0, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        v_target = eq_v.target
        _scripted_term_draws(monkeypatch, uv_pool, ['u'])

        assert _break_equation_duplication(
            eq_v, eq_u.active_terms_labels,
            preferred_sigs={eq_u.target.factors_labels}) is True
        assert not eq_v.weights_internal_evald
        assert not eq_v.weights_final_evald
        assert eq_v.target is v_target

    def test_the_term_carrying_the_earlier_target_goes_first(self, uv_pool,
                                                            monkeypatch):
        """Both of the v-equation's features are shared; the one carrying the
        u-equation's target ``du/dx0`` is rerolled although ``v`` comes first
        in structure order."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
                          'v': f'0.4 * {V_TOKEN} + 2.0 * {DU} + 0.0 = {DV}'},
                         {'u': [0.5, 0.3, 0.0], 'v': [0.4, 2.0, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        assert _equations_share_active_structure(eq_u, eq_v)
        v_plain, v_du, _ = eq_v.structure
        made = _scripted_term_draws(monkeypatch, uv_pool, ['u'])

        selector, inner = _soeq_selector()
        _run(selector, system)

        assert made == [v_du]
        assert v_plain.name == V_TOKEN and v_du.name == U_TOKEN
        assert inner.calls == ['u', 'v', 'v']

    @pytest.mark.parametrize('texts, weights', [
        # u carries the v-equation's target dv/dx0 as a standalone term.
        ({'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
         {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]}),
        # Both equations explain one composite target, with different features.
        ({'u': f'0.5 * {U_TOKEN} + 0.0 = {DU} * {DV}',
          'v': f'0.7 * {V_TOKEN} + 0.0 = {DU} * {DV}'},
         {'u': [0.5, 0.0], 'v': [0.7, 0.0]}),
    ], ids=['target-as-standalone-term', 'shared-composite-target'])
    def test_sharing_without_an_identical_law_is_not_repaired(
            self, uv_pool, monkeypatch, texts, weights):
        """NEGATIVE pin. The retired target-term rule rerolled exactly these;
        the guard must leave them alone, since the active sets differ."""
        from epde.structure.main_structures import Term
        system = _system(uv_pool, texts, weights)
        before = {var: _names(system.vals[var]) for var in texts}
        monkeypatch.setattr(Term, 'randomize',
                            lambda self, *a, **k: pytest.fail('repaired a legal system'))

        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert inner.calls == ['u', 'v']
        assert {var: _names(system.vals[var]) for var in texts} == before
        assert not any('duplicate-equation guard' in m for m in messages), messages

    def test_a_clean_system_takes_one_pass_and_no_rescan(self, uv_pool,
                                                        monkeypatch):
        """Two equations = one pair. A pass that finds no duplicate is the
        fixed point and has already proved the invariant, so the predicate is
        evaluated exactly once."""
        system = _system(uv_pool,
                         {'u': f'0.3 * {V_TOKEN} + 0.0 = {DU}',
                          'v': f'0.7 * {U_TOKEN} + 0.0 = {DV}'},
                         {'u': [0.3, 0.0], 'v': [0.7, 0.0]})
        stock = rps._equations_share_active_structure
        scans = []

        def counting(eq_a, eq_b):
            scans.append((eq_a.main_var_to_explain, eq_b.main_var_to_explain))
            return stock(eq_a, eq_b)

        monkeypatch.setattr(rps, '_equations_share_active_structure', counting)
        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert scans == [('u', 'v')]
        assert inner.calls == ['u', 'v']
        assert not any('duplicate-equation guard' in m for m in messages), messages


class TestAnUnrepairableDuplicate:
    """``du/dx0*dv/dx0 = const`` twice: the shared structure is carried by the
    target alone, so there is no non-target term to reroll."""

    @staticmethod
    def _target_only_pair(pool):
        shared = f'{DU} * {DV}'
        return _system(pool,
                       {'u': f'0.0 * {U_TOKEN} + 0.0 = {shared}',
                        'v': f'0.0 * {V_TOKEN} + 0.0 = {shared}'},
                       {'u': [0.0, 0.0], 'v': [0.0, 0.0]})

    def test_the_repair_reports_no_change(self, uv_pool, monkeypatch):
        from epde.structure.main_structures import Term
        system = self._target_only_pair(uv_pool)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        assert _equations_share_active_structure(eq_u, eq_v)
        before = _names(eq_v)
        monkeypatch.setattr(Term, 'randomize',
                            lambda self, *a, **k: pytest.fail('rerolled a term'))

        assert _break_equation_duplication(
            eq_v, eq_u.active_terms_labels,
            preferred_sigs={eq_u.target.factors_labels}) is False
        assert _names(eq_v) == before
        assert eq_v.weights_internal_evald, 'nothing changed, so the fit stays'

    def test_the_postcondition_says_so(self, uv_pool):
        """No change reads as the loop's fixed point, so the loop alone would
        stop at pass 1 believing it converged. The duplicate must be said out
        loud."""
        system = self._target_only_pair(uv_pool)
        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert inner.calls == ['u', 'v']
        assert any('duplicate-equation guard not converged' in m
                   and 'u and v' in m for m in messages), messages


# ---------------------------------------------------------------------------
# The repair primitive keeps the structure duplicate-free.
# ---------------------------------------------------------------------------

class TestTheRerollNeverLeavesADuplicate:

    def test_a_cap_hit_on_a_two_term_equation_redraws_for_uniqueness(
            self, uv_pool, monkeypatch):
        """The post-loop drop refuses to go below two terms, so a duplicate left
        by a cap-hit would ride out of the repair -- and the next
        ``EqRightPartSelector.apply`` asserts on duplicates."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.0 = {DU}',
                          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
                         {'u': [0.5, 0.0], 'v': [0.7, 0.0]})
        eq_u = system.vals['u']
        shared = eq_u.structure[0]
        # Duplicate the target until the loop gives up, then a free term.
        made = _scripted_term_draws(monkeypatch, uv_pool, ['du/dx0'] * 5 + ['v'])

        assert _break_equation_duplication(eq_u, {shared.factors_labels},
                                           preferred_sigs={shared.factors_labels},
                                           max_iter=5) is True
        assert len(made) == 6                      # 5 rejected, then the redraw
        assert shared.name == V_TOKEN
        signatures = {t.factors_labels for t in eq_u.structure}
        assert len(signatures) == len(eq_u.structure)


# ---------------------------------------------------------------------------
# Not the guard's, but it builds the systems these tests run on.
# ---------------------------------------------------------------------------

class TestTheTranslatedSystemKnowsItsVariables:

    def test_each_translated_equation_explains_its_own_variable(self, uvw_pool):
        """translate_equation's dict overload used to rebind its loop key in the
        sparsity loop, so every equation came back explaining all_vars[-1]
        ('w'); SolverBasedFitness then observed w for all three."""
        from epde.interface.equation_translator import translate_equation
        system = translate_equation(
            {'u': f'0.5 * {V_TOKEN} + 0.0 = {DU}',
             'v': f'0.7 * {U_TOKEN} + 0.0 = {DV}',
             'w': f'0.2 * {U_TOKEN} + 0.0 = dw/dx0{{power: 1.0}}'},
            uvw_pool, all_vars=['u', 'v', 'w'])
        assert {var: system.vals[var].main_var_to_explain
                for var in system.vars_to_describe} == {'u': 'u', 'v': 'v', 'w': 'w'}
