#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single-equation recovery paths of ``EqRightPartSelector.apply``.

Three paths fire only when the fit keeps failing, so a live search hardly ever
shows them; each is driven here by stubbing the fit/score step:

* the INF-FITNESS REROLL -- every eligible target is declined, so the whole
  equation is rerolled and the outer loop goes round again;
* the RESTORE GIVE-UP -- ``restore_property`` never supplies a derivative of
  the explained variable, so after ``inner_max_iter`` attempts the operator
  warns and rerolls the equation;
* the EXIT GUARANTEE -- the outer loop hits its cap with the target orphaned
  by the last reroll, and a valid target must still leave RPS.

They used to be pinned only as the ban-free CONTROLS of the retired
cross-equation target rule's scrub tests (``test_target_core_ban.py``). With
that rule gone, these are the only behavioural tests of the paths themselves.
"""

import warnings

import numpy as np
import pytest

from conftest import using_config

# The translator gives every term one factor at most; regenerated terms need
# the live two-factor budget the searches run with.
_LIVE_FACTORS = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
DU = 'du/dx0{power: 1.0}'
DV = 'dv/dx0{power: 1.0}'
U_TOKEN = 'u{power: 1.0}'
V_TOKEN = 'v{power: 1.0}'


@pytest.fixture(scope='module')
def uv_pool():
    import epde
    search = epde.EpdeSearch(use_solver=False,
                             verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 4 * np.pi, 200)
    _, domain = search.createDomain(t, boundary_width=10, ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory(
        {'u': np.sin(t), 'v': np.cos(t) + 0.3 * np.sin(2 * t)}, domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1)
    return search.pool


def _translated_u_equation(pool, text):
    """The u-equation of ``text`` beside a fixed v-equation, translated on
    ``pool`` and given the live factor budget. The v-equation is there only
    because the translator builds a system; nothing below reads it."""
    from epde.interface.equation_translator import translate_equation
    system = translate_equation(
        {'u': text, 'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'}, pool, all_vars=['u', 'v'])
    eq = system.vals['u']
    eq.main_var_to_explain = 'u'
    eq.metaparameters['max_factors_in_term']['value'] = dict(_LIVE_FACTORS)
    for term in eq.structure:
        term.max_factors_in_term = dict(_LIVE_FACTORS)
    return eq


def _u_equation(pool, text, weights):
    """``_translated_u_equation`` pinned to the fitted state a previous
    selection leaves: target last, ``weights`` evaluated."""
    eq = _translated_u_equation(pool, text)
    w = np.array(weights, dtype=float)
    eq.weights_internal = w
    eq.weights_final = w.copy()
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    eq.target_idx = len(eq.structure) - 1
    return eq


def _terms(pool, text):
    """Fresh ``Term`` objects for the u-equation ``text``, on the live factor
    budget -- what a scripted reroll installs."""
    return list(_translated_u_equation(pool, text).structure)


def _stub_fit(monkeypatch, failing):
    """Stand in for the fit/score step of the term sweep: install the unit
    weight layout ``apply`` expects and return a score -- or ``None`` while
    ``failing['now']`` holds, which is how the real operator reports "no
    usable fit" and is what triggers the inf-fitness reroll. Returns the list
    the stub appends each call's target index to."""
    from epde.operators.common.right_part_selection import EqRightPartSelector
    calls = []

    def fake_fit(self, objective, subop_args):
        calls.append(objective.target_idx)
        weights = np.ones(len(objective.structure))
        objective.weights_internal = weights
        objective.weights_final = weights.copy()
        objective.weights_internal_evald = True
        objective.weights_final_evald = True
        objective._cached_sw_weights = None
        objective._cached_vc_score = None
        return None if failing['now'] else 0.5

    monkeypatch.setattr(EqRightPartSelector, '_fit_and_score', fake_fit)
    return calls


def _apply(eq):
    """Run a bare ``EqRightPartSelector`` on ``eq`` (``_fit_and_score`` is
    stubbed, so it needs no suboperators); return the warning messages."""
    from epde.operators.common.right_part_selection import EqRightPartSelector
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        EqRightPartSelector().apply(objective=eq, arguments={})
    return [str(w.message) for w in caught]


# The reroll every test below scripts. It carries ``dv/dx0`` -- the
# v-equation's target -- as a standalone term: the retired cross-equation rule
# scrubbed exactly that, so it also pins that nothing single-equation does any
# more.
REROLLED = f'0.5 * {DV} + 0.3 * {U_TOKEN} + 0.0 = {DU}'


class TestTheInfFitnessReroll:

    def test_the_rerolled_structure_leaves_rps(self, uv_pool, monkeypatch):
        """Every eligible target scoring ``None`` rerolls the whole equation;
        the next outer iteration selects on the rerolled structure, and every
        one of its terms -- ``dv/dx0`` included -- leaves RPS."""
        from epde.structure.main_structures import Equation
        eq_u = _u_equation(uv_pool, f'0.5 * {V_TOKEN} + 0.0 = {DU}', [0.5, 0.0])
        failing = {'now': True}
        _stub_fit(monkeypatch, failing)
        rerolled = []
        real_randomize = Equation.randomize

        def scripted_randomize(self):
            real_randomize(self)
            self.structure = _terms(uv_pool, REROLLED)
            self._invalidate_label_cache()
            rerolled.append(list(self.structure))
            failing['now'] = False        # the reroll "worked"; the sweep scores

        monkeypatch.setattr(Equation, 'randomize', scripted_randomize)
        messages = _apply(eq_u)

        assert len(rerolled) == 1, 'the inf-fitness reroll fired %d times' % len(rerolled)
        # The very term objects the reroll installed, none redrawn or dropped.
        assert {id(t) for t in eq_u.structure} == {id(t) for t in rerolled[0]}
        assert {t.name for t in eq_u.structure} == {DV, U_TOKEN, DU}
        assert eq_u.target is not None and eq_u.target.name == DU
        assert not any('outer loop did not converge' in m for m in messages), messages


class TestTheRestoreGiveUp:

    def test_the_give_up_warns_and_rerolls(self, uv_pool, monkeypatch):
        """``restore_property`` never manages to introduce a derivative of u,
        so after ``inner_max_iter`` attempts the operator says so and rerolls
        the equation wholesale; selection then runs on the reroll."""
        from epde.structure.main_structures import Equation
        eq_u = _u_equation(uv_pool, f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
                           [0.5, 0.3, 0.0])
        # No derivative of u anywhere, so the restore loop must run.
        eq_u.structure = [t for t in eq_u.structure if not t.contains_deriv('u')]
        eq_u._target_term = None
        eq_u._invalidate_label_cache()
        assert eq_u.structure, 'the fixture left nothing to start from'
        _stub_fit(monkeypatch, {'now': False})
        restores, rerolled = [], []

        def failing_restore(self, **kwargs):
            restores.append(None)        # injects nothing: the property is never met

        def scripted_randomize(self):
            self.structure = _terms(uv_pool, REROLLED)
            self._target_term = None
            self._invalidate_label_cache()
            rerolled.append(list(self.structure))

        monkeypatch.setattr(Equation, 'restore_property', failing_restore)
        monkeypatch.setattr(Equation, 'randomize', scripted_randomize)
        messages = _apply(eq_u)

        assert restores, 'the restore loop never ran'
        assert len(rerolled) == 1, 'the give-up reroll fired %d times' % len(rerolled)
        give_ups = [m for m in messages
                    if 'restore_property failed to introduce a deriv' in m]
        assert len(give_ups) == 1, messages
        assert "'u'" in give_ups[0]
        assert {id(t) for t in eq_u.structure} == {id(t) for t in rerolled[0]}
        assert eq_u.target is not None and eq_u.target.name == DU


class TestTheExitGuarantee:

    # Two eligible targets, the composite the FIRST of them, so structure order
    # takes it. Neither end of the structure is eligible: ``v`` (term 0, the
    # "no eligible term at all" fallback) and ``u`` (the translator's target
    # slot) carry no derivative of u -- so a fallback taking term 0, the last
    # eligible term or the old target position each picks something else.
    _EXIT_TEXT = (f'0.5 * {V_TOKEN} + 0.3 * {DU} * {DV} + 0.2 * {DU} + 0.0 '
                  f'= {U_TOKEN}')

    @pytest.mark.parametrize('amplification_cap', [100.0, None],
                             ids=['probe-declines-every-candidate', 'no-probe'])
    def test_a_derivative_carrying_target_leaves_through_the_fallback(
            self, uv_pool, monkeypatch, amplification_cap):
        """Every candidate is declined, so the outer loop rerolls until its
        cap and breaks with the identity-tracked target orphaned by the last
        reroll. The exit fallback must install the first term carrying a
        derivative of u -- whether its probe ran and declined every candidate
        (default cap) or was switched off -- and leave the equation fitted."""
        import epde.operators.common.right_part_selection as rps
        from epde.structure.main_structures import Equation
        eq_u = _u_equation(uv_pool, f'0.5 * {V_TOKEN} + 0.0 = {DU}', [0.5, 0.0])
        fits = _stub_fit(monkeypatch, {'now': True})     # every candidate declined
        text = self._EXIT_TEXT
        rerolled = []

        def scripted_randomize(self):
            self.structure = _terms(uv_pool, text)
            self._target_term = None
            self._invalidate_label_cache()
            rerolled.append(list(self.structure))

        monkeypatch.setattr(Equation, 'randomize', scripted_randomize)
        stock_record = rps._loop_stats.record
        sites = []

        def spy_record(site, iters, cap):
            sites.append(site)
            stock_record(site, iters, cap)

        monkeypatch.setattr(rps._loop_stats, 'record', spy_record)
        with using_config(rps_amplification_cap=amplification_cap):
            messages = _apply(eq_u)

        assert any('outer loop did not converge' in m for m in messages), messages
        assert sites.count('EqRPS.exit_target_fallback') == 1, 'the fallback never fired'
        last = rerolled[-1]
        assert {id(t) for t in eq_u.structure} == {id(t) for t in last}
        first_deriv = next(t for t in last if t.contains_deriv('u'))
        assert eq_u.target is first_deriv, eq_u.target and eq_u.target.name
        assert eq_u.target.name == f'{DU} * {DV}'
        # EXIT CONTRACT: fitted for the target actually installed. The last
        # candidate either the sweep or the probe tried is the later ``du/dx0``,
        # so only the contract's refit makes the final fit the composite's.
        assert fits[-1] == eq_u.target_idx
        assert eq_u.weights_internal_evald and eq_u.weights_final_evald
        assert eq_u.right_part_selected
