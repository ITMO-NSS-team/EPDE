#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single-equation failure paths of ``EqRightPartSelector.apply``.

Right-part selection uses only its input: a candidate it cannot fit leaves
DECLINED -- stamped fit-degenerate for the population operators to redraw --
instead of being rerolled inside the operator. Three paths fire only when the
fit keeps failing, so a live search hardly ever shows them; each is driven
here by stubbing the fit/score step:

* ALL TARGETS DECLINED -- every eligible target is declined, so the equation
  leaves declined (it used to be rerolled wholesale and the outer loop went
  round again);
* the ENTRY RESTORE -- no term carries a derivative of the explained variable,
  so ``restore_property`` (a validator) is called ONCE; a restore that
  installs nothing leaves the candidate declined (the operator used to retry
  it 100 times, warn and reroll the equation);
* no EXIT FALLBACK -- the cap break that used to install a fallback target
  after rerolls is gone with the rerolls; a candidate whose every target is
  declined leaves declined, never fitted for a target the sweep refused.

They used to be pinned only as the ban-free CONTROLS of the retired
derivative-core / wrap version of the cross-equation target rule
(``test_target_core_ban.py``). The repair contract as a whole -- output built
from the input terms, the scrubs dropping, the outer cap -- is pinned in
``test_rps_repair_contract.py``.
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
    budget."""
    return list(_translated_u_equation(pool, text).structure)


def _stub_fit(monkeypatch, failing):
    """Stand in for the fit/score step of the term sweep: install the unit
    weight layout ``apply`` expects and return a score -- or ``None`` while
    ``failing['now']`` holds, which is how the real operator reports "no
    usable fit". Returns the list the stub appends each call's target index
    to."""
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


@pytest.fixture
def no_draws(monkeypatch):
    """Right-part selection draws nothing: neither the equation nor a term may
    be rerolled inside it."""
    from epde.structure.main_structures import Equation, Term

    def refuse(self, *args, **kwargs):
        raise AssertionError('right-part selection redrew %s' % type(self).__name__)

    monkeypatch.setattr(Equation, 'randomize', refuse)
    monkeypatch.setattr(Term, 'randomize', refuse)


def _selector():
    """An ``EqRightPartSelector`` whose only wired suboperator is the host a
    declined candidate is stamped by (``_fit_and_score`` is stubbed)."""
    from epde.operators.common.fitness import SolverFreeFitness
    from epde.operators.common.objectives import Discrepancy, Instability
    from epde.operators.common.right_part_selection import EqRightPartSelector
    disc = Discrepancy()
    host = SolverFreeFitness(['penalty_coeff'], objectives=[disc, Instability()],
                             primary=disc)
    host.params = {'penalty_coeff': 0.2}
    selector = EqRightPartSelector()
    selector.set_suboperators({'fitness_calculation': host})
    return selector


def _apply(eq):
    """Run the selector on ``eq``; return the warning messages."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _selector().apply(objective=eq, arguments={})
    return [str(w.message) for w in caught]


def _assert_declined(eq):
    """The verdict ``_decline`` leaves: every objective stamped, the stamp
    read by the population operators, zero weights for an installed target."""
    from epde.operators.common.objectives import LOSS_NAN_VAL
    from epde.operators.common.right_part_selection import _rps_declined
    from epde.operators.multiobjective.moeadd_specific import _equation_is_degenerate
    assert _rps_declined(eq) and _equation_is_degenerate(eq)
    assert eq.fitness_value == LOSS_NAN_VAL and eq.fitness_calculated
    assert eq.coefficients_stability == LOSS_NAN_VAL
    assert eq.right_part_selected and eq.target is not None
    assert eq.weights_internal_evald and eq.weights_final_evald
    assert not np.any(eq.weights_internal) and not np.any(eq.weights_final)


class TestAllTargetsDeclined:

    def test_the_candidate_leaves_declined_and_unredrawn(self, uv_pool, monkeypatch,
                                                         no_draws):
        """Every eligible target scoring ``None`` declines the candidate: its
        terms leave as they came in, unpruned, and the one sweep is all the
        fitting it gets."""
        eq_u = _u_equation(uv_pool, f'0.5 * {V_TOKEN} + 0.3 * {DV} + 0.0 = {DU}',
                           [0.5, 0.3, 0.0])
        before = list(eq_u.structure)
        fits = _stub_fit(monkeypatch, {'now': True})
        messages = _apply(eq_u)

        _assert_declined(eq_u)
        assert [id(t) for t in eq_u.structure] == [id(t) for t in before]
        assert eq_u.target.name == DU
        assert len(fits) == 1, 'one sweep over the one eligible target'
        assert not any('did not converge' in m for m in messages), messages


class TestTheEntryRestore:

    def test_a_restored_term_is_the_only_addition(self, uv_pool, monkeypatch, no_draws):
        """No term carries a derivative of u, so ONE ``restore_property`` call
        installs one; the sweep selects it and nothing else is added."""
        from epde.structure.main_structures import Equation
        eq_u = _u_equation(uv_pool, f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
                           [0.5, 0.3, 0.0])
        eq_u.structure = [t for t in eq_u.structure if not t.contains_deriv('u')]
        eq_u._target_term = None
        eq_u._invalidate_label_cache()
        entry = list(eq_u.structure)
        restores = []

        def scripted_restore(self, **kwargs):
            restores.append(kwargs)
            self.structure.append(_terms(uv_pool, f'0.5 * {V_TOKEN} + 0.0 = {DU}')[-1])
            self._invalidate_label_cache()

        monkeypatch.setattr(Equation, 'restore_property', scripted_restore)
        _stub_fit(monkeypatch, {'now': False})
        _apply(eq_u)

        assert restores == [{'mandatory_family': False, 'deriv': True}]
        assert eq_u.target is not None and eq_u.target.name == DU
        assert [id(t) for t in eq_u.structure[:-1]] == [id(t) for t in entry]
        assert eq_u.right_part_selected and not np.isnan(eq_u.weights_final).any()

    def test_a_restore_that_installs_nothing_declines(self, uv_pool, monkeypatch, no_draws):
        """``restore_property`` is a validator, called once: when it leaves the
        candidate without a derivative of u, the sweep has no eligible target
        and the candidate is declined -- no retry loop, no reroll."""
        from epde.structure.main_structures import Equation
        eq_u = _u_equation(uv_pool, f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
                           [0.5, 0.3, 0.0])
        eq_u.structure = [t for t in eq_u.structure if not t.contains_deriv('u')]
        eq_u._target_term = None
        eq_u._invalidate_label_cache()
        restores = []

        def failing_restore(self, **kwargs):
            restores.append(None)        # injects nothing: the property is never met

        monkeypatch.setattr(Equation, 'restore_property', failing_restore)
        fits = _stub_fit(monkeypatch, {'now': False})
        messages = _apply(eq_u)

        assert len(restores) == 1
        assert fits == []
        _assert_declined(eq_u)
        assert not any('restore_property failed' in m for m in messages), messages


class TestNoExitFallback:

    # Two eligible targets, the composite the FIRST of them. Neither end of the
    # structure is eligible: ``v`` (term 0) and ``u`` (the translator's target
    # slot) carry no derivative of u.
    _EXIT_TEXT = (f'0.5 * {V_TOKEN} + 0.3 * {DU} * {DV} + 0.2 * {DU} + 0.0 '
                  f'= {U_TOKEN}')

    @pytest.mark.parametrize('amplification_cap', [100.0, None],
                             ids=['default-cap', 'no-cap'])
    def test_every_target_declined_leaves_declined(self, uv_pool, monkeypatch, no_draws,
                                                   amplification_cap):
        """Every candidate is declined. The candidate leaves declined after one
        sweep -- no outer-loop retries, no fallback probe, no exit refit -- with
        the first derivative-carrying term as its placeholder target, which
        the stamp keeps off every front."""
        import epde.operators.common.right_part_selection as rps
        eq_u = _translated_u_equation(uv_pool, self._EXIT_TEXT)
        before = list(eq_u.structure)
        fits = _stub_fit(monkeypatch, {'now': True})
        stock_record = rps._loop_stats.record
        sites = []

        def spy_record(site, iters, cap):
            sites.append(site)
            stock_record(site, iters, cap)

        monkeypatch.setattr(rps._loop_stats, 'record', spy_record)
        with using_config(rps_amplification_cap=amplification_cap):
            messages = _apply(eq_u)

        assert not any('did not converge' in m for m in messages), messages
        assert 'EqRPS.exit_target_fallback' not in sites
        assert sites.count('EqRPS.declined.all_targets_declined') == 1
        assert [id(t) for t in eq_u.structure] == [id(t) for t in before]
        # One sweep: the two eligible targets, in structure order.
        assert [eq_u.structure[i].name for i in fits] == [f'{DU} * {DV}', DU]
        _assert_declined(eq_u)
        assert eq_u.target.name == f'{DU} * {DV}'
