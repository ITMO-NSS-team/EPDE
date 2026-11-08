#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The target-term rule of ``SoEqRightPartSelector``.

No equation may carry another equation's TARGET as a whole, standalone term of
its active structure. ``du/dx0 = a*u + b*dv/dx0`` beside an equation for
``dv/dx0`` restates the v-equation inside the u-equation: a valid identity, but
one that crowds the true law off the front (LV loses both equations to it in
the best-subset harness). The carrier has that term rerolled and its right part
re-selected.

Whole-term equality is the point. The target as a FACTOR of a product is
coupling physics: continuity's ``v_y`` in ``v*v_y``, the double pendulum's
``theta2''`` in ``cos(D)*theta2''``. A factor ban makes the double pendulum
unreachable at two factors per term, so a negative pin keeps it out.
"""

import warnings
from types import SimpleNamespace

import pytest

import epde.operators.common.right_part_selection as rps
from epde.operators.common.right_part_selection import _carries_target_of
from epde.operators.multiobjective.moeadd_specific import _debug_assert_rps_unique

from test_duplicate_equation_guard import (DU, DV, U_TOKEN, V_TOKEN,  # noqa: F401
                                           _names, _run, _scripted_term_draws,
                                           _soeq_selector, _system,
                                           uv_pool, uvw_pool)

DW = 'dw/dx0{power: 1.0}'
TERM_RULE_WARNING = 'target-term rule not converged'


# ---------------------------------------------------------------------------
# The predicate.
# ---------------------------------------------------------------------------

class TestThePredicate:

    @pytest.mark.parametrize('texts, weights, u_carries_v, v_carries_u', [
        # u carries v's target as a term of its own.
        ({'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
         {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]}, True, False),
        # v's target inside a product: coupling, not a leak.
        ({'u': f'0.5 * {V_TOKEN} * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
         {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]}, False, False),
        # Powers are kept: (dv/dx0)^2 is not the term dv/dx0.
        ({'u': f'0.5 * dv/dx0{{power: 2.0}} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
         {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]}, False, False),
        # A zero-weight padding term is not part of the fitted law.
        ({'u': f'0.0 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
         {'u': [0.0, 0.3, 0.0], 'v': [0.7, 0.0]}, False, False),
        # One composite target explained twice: each carries the other's.
        ({'u': f'0.5 * {U_TOKEN} + 0.0 = {DU} * {DV}',
          'v': f'0.7 * {V_TOKEN} + 0.0 = {DU} * {DV}'},
         {'u': [0.5, 0.0], 'v': [0.7, 0.0]}, True, True),
        # One law solved for two quantities: every duplicate is a leak.
        ({'u': f'0.5 * {DV} + 0.0 = {DU}', 'v': f'2.0 * {DU} + 0.0 = {DV}'},
         {'u': [0.5, 0.0], 'v': [2.0, 0.0]}, True, True),
    ], ids=['standalone', 'as-a-factor', 'other-power', 'zero-weight-padding',
            'shared-composite-target', 'identical-law'])
    def test_whole_term_equality_in_the_active_structure(
            self, uv_pool, texts, weights, u_carries_v, v_carries_u):
        system = _system(uv_pool, texts, weights)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        assert _carries_target_of(eq_v, eq_u) is u_carries_v
        assert _carries_target_of(eq_u, eq_v) is v_carries_u

    def test_an_equation_without_a_target_leaks_nothing(self):
        owner = SimpleNamespace(target=None)
        carrier = SimpleNamespace(active_terms_labels=frozenset({frozenset()}))
        assert _carries_target_of(owner, carrier) is False

    def test_a_target_without_its_own_derivative_reserves_nothing(self, uv_pool,
                                                                  monkeypatch):
        """A target without a derivative of its own variable (here ``u`` for
        the v-equation) is a placeholder: ``_mark_unassignable`` installs
        term 0 as the target of an equation where no term holds such a
        derivative. Reserving it would reroll ``u`` out of the u-equation,
        where it is a true term."""
        from epde.structure.main_structures import Term
        system = _system(uv_pool,
                         {'u': f'0.3 * {U_TOKEN} + 0.0 = {DU}',
                          'v': f'0.7 * {DV} + 0.0 = {U_TOKEN}'},
                         {'u': [0.3, 0.0], 'v': [0.7, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        assert eq_v.target.factors_labels in eq_u.active_terms_labels
        assert rps._reserved_target_signature(eq_v) is None
        assert _carries_target_of(eq_v, eq_u) is False
        monkeypatch.setattr(Term, 'randomize',
                            lambda self, *a, **k: pytest.fail('rerolled a true term'))

        selector, inner = _soeq_selector()
        _run(selector, system)
        assert inner.calls == ['u', 'v']


# ---------------------------------------------------------------------------
# The repair inside SoEqRightPartSelector.
# ---------------------------------------------------------------------------

def _messages_about_the_rule(messages):
    return [m for m in messages if TERM_RULE_WARNING in m]


def _counting_predicate(monkeypatch):
    """Record every (owner, carrier) check the operator makes, in order."""
    stock = rps._carries_target_of
    checks = []

    def counting(owner, carrier):
        checks.append((owner.main_var_to_explain, carrier.main_var_to_explain))
        return stock(owner, carrier)

    monkeypatch.setattr(rps, '_carries_target_of', counting)
    return checks


class TestTheRepair:

    def test_only_the_carrier_is_repaired(self, uv_pool, monkeypatch):
        """u carries ``dv/dx0``; that term alone is rerolled, u keeps its
        target and is re-selected, and v is left alone."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
                          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
                         {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        u_leak, _, u_target = eq_u.structure
        v_before = _names(eq_v)
        made = _scripted_term_draws(monkeypatch, uv_pool, ['u'])

        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert made == [u_leak]
        assert u_leak.name == U_TOKEN and eq_u.target is u_target
        assert not eq_u.weights_internal_evald, 'the stale fit is dropped'
        assert _names(eq_v) == v_before
        assert inner.calls == ['u', 'v', 'u']
        assert not _carries_target_of(eq_v, eq_u)
        assert not _messages_about_the_rule(messages), messages

    @pytest.mark.parametrize('first_draw', ['dv/dx0', 'dw/dx0'],
                             ids=['back-onto-the-leak', 'onto-a-third-target'])
    def test_a_redraw_onto_any_other_target_is_rejected(self, uvw_pool,
                                                        monkeypatch, first_draw):
        """Both first draws are unique within the u-equation, so only the ban
        can refuse them: the leaked target itself, and the target of a third
        equation, which would only move the leak."""
        system = _system(uvw_pool,
                         {'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
                          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}',
                          'w': f'0.2 * {U_TOKEN} + 0.0 = {DW}'},
                         {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0], 'w': [0.2, 0.0]})
        eq_u = system.vals['u']
        u_leak = eq_u.structure[0]
        made = _scripted_term_draws(monkeypatch, uvw_pool, [first_draw, 'u'])

        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert made == [u_leak, u_leak]
        assert u_leak.name == U_TOKEN
        assert inner.calls == ['u', 'v', 'w', 'u']
        assert not any(_debug_assert_rps_unique(system))
        assert not _messages_about_the_rule(messages), messages

    def test_a_mutual_leak_is_repaired_in_both_equations(self, uv_pool,
                                                         monkeypatch):
        """The LV shape: each equation carries the other's target. The pair
        is checked later equation first, so v is repaired, then u."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.3 * {U_TOKEN} + 0.0 = {DU}',
                          'v': f'0.4 * {DU} + 0.7 * {V_TOKEN} + 0.0 = {DV}'},
                         {'u': [0.5, 0.3, 0.0], 'v': [0.4, 0.7, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        u_leak, v_leak = eq_u.structure[0], eq_v.structure[0]
        made = _scripted_term_draws(monkeypatch, uv_pool, ['u', 'v'])
        checks = _counting_predicate(monkeypatch)

        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert made == [v_leak, u_leak]
        assert v_leak.name == U_TOKEN and u_leak.name == V_TOKEN
        assert inner.calls == ['u', 'v', 'v', 'u']
        # Both directions repaired in pass 1, a clean pass 2, and no rescan.
        assert checks == [('u', 'v'), ('v', 'u')] * 2
        assert not any(_debug_assert_rps_unique(system))
        assert not _messages_about_the_rule(messages), messages

    def test_a_leak_made_by_a_re_selection_is_caught_next_pass(self, uv_pool,
                                                              monkeypatch):
        """Pass 1 repairs u, and u's re-selection moves its target onto the
        rerolled term ``du/dx0*v`` -- which v holds as a coupling term. That
        pair was already checked this pass, so pass 2 must catch the new
        leak, reading u's NEW target; pass 3 is clean."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
                          'v': f'0.4 * {DU} * {V_TOKEN} + 0.7 * {V_TOKEN} + 0.0 = {DV}'},
                         {'u': [0.5, 0.3, 0.0], 'v': [0.4, 0.7, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        u_leak, v_coupling = eq_u.structure[0], eq_v.structure[0]
        made = _scripted_term_draws(monkeypatch, uv_pool, ['du/dx0*v', 'u'])
        checks = _counting_predicate(monkeypatch)

        def retarget(eq, call_no):
            # u's first re-selection explains the two-factor term.
            if eq.main_var_to_explain == 'u' and call_no == 2:
                eq.target_idx = next(i for i, t in enumerate(eq.structure)
                                     if len(t.structure) == 2)

        selector, inner = _soeq_selector(on_apply=retarget)
        messages = _run(selector, system)

        assert made == [u_leak, v_coupling]
        assert eq_u.target is u_leak and v_coupling.name == U_TOKEN
        assert inner.calls == ['u', 'v', 'u', 'v']
        assert checks == [('u', 'v'), ('v', 'u')] * 3
        assert not any(_debug_assert_rps_unique(system))
        assert not any('not converged' in m for m in messages), messages

    @pytest.mark.parametrize('weights', [[0.5, 0.0], [0.0, 0.0]],
                             ids=['with-features', 'target-only'])
    def test_a_composite_target_explained_twice_rerolls_the_later_target(
            self, uv_pool, monkeypatch, weights):
        """``du/dx0*dv/dx0`` explains both equations. The leaked term is the
        v-equation's own target, so the target is rerolled; the Term object
        stays the installed target for the re-selection. The target-only
        variant is the duplicate the guard alone cannot repair."""
        shared = f'{DU} * {DV}'
        system = _system(uv_pool,
                         {'u': f'0.5 * {U_TOKEN} + 0.0 = {shared}',
                          'v': f'0.5 * {V_TOKEN} + 0.0 = {shared}'},
                         {'u': list(weights), 'v': list(weights)})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        v_target = eq_v.target
        made = _scripted_term_draws(monkeypatch, uv_pool, ['dv/dx0'])

        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert made == [v_target]
        assert eq_v.target is v_target and v_target.name == DV
        assert inner.calls == ['u', 'v', 'v']
        assert not any(_debug_assert_rps_unique(system))
        assert not any('not converged' in m for m in messages), messages

    def test_the_target_as_a_factor_is_left_alone(self, uv_pool, monkeypatch):
        """NEGATIVE pin (the NS / double-pendulum shape). A factor ban would
        reroll this coupling term; the rule must not."""
        from epde.structure.main_structures import Term
        texts = {'u': f'0.5 * {V_TOKEN} * {DV} + 0.3 * {U_TOKEN} + 0.0 = {DU}',
                 'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'}
        system = _system(uv_pool, texts, {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]})
        before = {var: _names(system.vals[var]) for var in texts}
        monkeypatch.setattr(Term, 'randomize',
                            lambda self, *a, **k: pytest.fail('repaired a legal system'))

        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert inner.calls == ['u', 'v']
        assert {var: _names(system.vals[var]) for var in texts} == before
        assert not any(_debug_assert_rps_unique(system))
        assert not _messages_about_the_rule(messages), messages

    def test_a_clean_system_checks_each_direction_once(self, uv_pool,
                                                       monkeypatch):
        """A pass that finds no leak is the fixed point and has already proved
        the invariant: one check per direction, and no rescan."""
        system = _system(uv_pool,
                         {'u': f'0.3 * {V_TOKEN} + 0.0 = {DU}',
                          'v': f'0.7 * {U_TOKEN} + 0.0 = {DV}'},
                         {'u': [0.3, 0.0], 'v': [0.7, 0.0]})
        checks = _counting_predicate(monkeypatch)
        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert checks == [('u', 'v'), ('v', 'u')]
        assert inner.calls == ['u', 'v']
        assert not _messages_about_the_rule(messages), messages

    def test_a_leak_that_keeps_coming_back_is_reported(self, uv_pool,
                                                       monkeypatch):
        """A repair that the re-selection undoes every time (simulated: the
        repair changes nothing) spends the pass budget. The postcondition
        must say which equation still carries whose target."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
                          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
                         {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]})
        monkeypatch.setattr(rps, '_break_target_leak', lambda *a, **k: True)

        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert inner.calls == ['u', 'v'] + ['u'] * 50
        reports = _messages_about_the_rule(messages)
        assert len(reports) == 1, messages
        assert 'after 50 of 50 passes' in reports[0]
        # Exactly the one leak: an equation never counts as carrying its own target.
        assert reports[0].endswith('1 leak(s) left: u carries the target of v'), reports[0]

    def test_a_budget_exit_after_leaks_also_rescans_for_duplicates(self, uv_pool,
                                                                 monkeypatch):
        """With three or more equations a leak repair late in the last pass
        can make a duplicate of a pair that pass already checked, so its
        duplicate scan saw nothing. Simulated: the in-loop duplicate checks
        all say no, the rescan finds one. The leak alone must trigger the
        duplicate rescan."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
                          'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
                         {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]})
        monkeypatch.setattr(rps, '_break_target_leak', lambda *a, **k: True)
        dup_checks = []

        def duplicate_only_at_exit(eq_a, eq_b):
            dup_checks.append(1)
            return len(dup_checks) > 50          # one pair per pass, 50 passes

        monkeypatch.setattr(rps, '_equations_share_active_structure',
                            duplicate_only_at_exit)
        selector, inner = _soeq_selector()
        messages = _run(selector, system)

        assert len(dup_checks) == 51
        assert any('duplicate-equation guard not converged' in m for m in messages), messages


# ---------------------------------------------------------------------------
# The repair primitive on a target term.
# ---------------------------------------------------------------------------

class TestRerollingATarget:

    def test_a_cap_hit_drops_the_duplicated_feature_and_keeps_the_target(
            self, uv_pool, monkeypatch):
        """The leaked term is v's own target, and every redraw duplicates the
        feature ``v``. The target is never dropped, so the drop policy removes
        the duplicated FEATURE; the Term object stays the installed target and
        the stale fit goes."""
        system = _system(uv_pool,
                         {'u': f'0.5 * {U_TOKEN} + 0.0 = {DU} * {DV}',
                          'v': f'0.5 * {V_TOKEN} + 0.3 * {U_TOKEN} + 0.0 = {DU} * {DV}'},
                         {'u': [0.5, 0.0], 'v': [0.5, 0.3, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        v_feature, u_feature, v_target = eq_v.structure
        made = _scripted_term_draws(monkeypatch, uv_pool, ['v'])

        assert rps._break_target_leak(eq_v, eq_u.target.factors_labels,
                                      {eq_u.target.factors_labels}, max_iter=3) is True

        assert made == [v_target] * 3
        assert [id(t) for t in eq_v.structure] == [id(u_feature), id(v_target)]
        assert eq_v.target is v_target and v_target.name == V_TOKEN
        assert not eq_v.weights_internal_evald and not eq_v.weights_final_evald


# ---------------------------------------------------------------------------
# The debug check reports what the operator repairs.
# ---------------------------------------------------------------------------

class TestTheDebugCheck:

    @pytest.mark.parametrize('u_text, flags', [
        (f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}', [True, False]),
        (f'0.5 * {V_TOKEN} * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}', [False, False]),
    ], ids=['standalone', 'as-a-factor'])
    def test_it_flags_the_carrier_only(self, uv_pool, u_text, flags):
        system = _system(uv_pool,
                         {'u': u_text, 'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
                         {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]})
        assert _debug_assert_rps_unique(system) == flags
