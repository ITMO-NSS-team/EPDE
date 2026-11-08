#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``_term_divides`` -- the ratio-fit scrub predicate in
``simplify_equation``: a feature term divides the target iff every one of
its factors appears in the target with at least the same power. Divisor
features make the ``Lambda * (identity)`` FD-error soak representable
(the wave trap ``u*u_tt = 3.75*u_tt - 0.15*u_xx + 0.04*u*u_xx``)."""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.operators.common.right_part_selection import (
    _related_to_target, _term_divides)


def _factor(label, power):
    return SimpleNamespace(
        structural_label_without_power=(label, ()),
        cache_label=(label, (power,)))


def _term(*factors):
    return SimpleNamespace(structure=list(factors))


class TestTermDivides:

    def test_derivative_core_divides_composite_target(self):
        # The wave trap: u_tt divides u * u_tt.
        u_tt = _term(_factor('d^2u/dx0^2', 1.0))
        target = _term(_factor('u', 1.0), _factor('d^2u/dx0^2', 1.0))
        assert _term_divides(u_tt, target)

    def test_plain_variable_divides_composite_target(self):
        u = _term(_factor('u', 1.0))
        target = _term(_factor('u', 1.0), _factor('d^2u/dx0^2', 1.0))
        assert _term_divides(u, target)

    def test_non_component_does_not_divide(self):
        # u_xx shares no factor with u * u_tt -- the padding term that
        # must NOT be classified as a divisor (the keep-rule owns it).
        u_xx = _term(_factor('d^2u/dx1^2', 1.0))
        target = _term(_factor('u', 1.0), _factor('d^2u/dx0^2', 1.0))
        assert not _term_divides(u_xx, target)

    def test_power_containment_is_directional(self):
        target_u1 = _term(_factor('u', 1.0), _factor('d^2u/dx0^2', 1.0))
        target_u2 = _term(_factor('u', 2.0), _factor('d^2u/dx0^2', 1.0))
        u2 = _term(_factor('u', 2.0))
        assert not _term_divides(u2, target_u1)   # u^2 does not divide u*u_tt
        assert _term_divides(u2, target_u2)       # u^2 divides u^2*u_tt

    def test_target_divides_itself(self):
        # Degenerate self-case (never reachable live: a feature equal to
        # the target is a banned duplicate) -- documents the convention.
        target = _term(_factor('u', 1.0), _factor('d^2u/dx0^2', 1.0))
        assert _term_divides(target, target)

    def test_multiple_of_target_detected_in_reverse_direction(self):
        # The hitchhiker shape: u_t * u_tt rides on the canonical target
        # u_tt -- the scrub tests BOTH directions, and this is the
        # reverse one (target divides the term).
        target = _term(_factor('d^2u/dx0^2', 1.0))
        hitchhiker = _term(_factor('du/dx0', 1.0),
                           _factor('d^2u/dx0^2', 1.0))
        assert _term_divides(target, hitchhiker)
        assert not _term_divides(hitchhiker, target)

    def test_partner_of_multiple_is_not_related(self):
        # The hitchhiker's cancelling partner u_t * u_xx is related to
        # the target u_tt in NEITHER direction -- it must survive the
        # scrub and die at the refit instead.
        target = _term(_factor('d^2u/dx0^2', 1.0))
        partner = _term(_factor('du/dx0', 1.0),
                        _factor('d^2u/dx1^2', 1.0))
        assert not _term_divides(partner, target)
        assert not _term_divides(target, partner)


# ---------------------------------------------------------------------------
# The scrub inside ``simplify_equation``, on a real pool.
# ---------------------------------------------------------------------------

# The translator gives every term one factor at most; regenerated terms need
# the live two-factor budget the searches run with.
_LIVE_FACTORS = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}


@pytest.fixture(scope='module')
def wave_pool():
    """A 2-D wave field (``u_tt = 0.04 u_xx``) with derivatives to second
    order in both axes. Several modes: a single one makes every
    finite-difference derivative column exactly proportional to ``u`` or its
    quadrature, and the linear-dependence scrub would then regenerate the very
    terms these tests watch -- passing them without the divisor scrub."""
    import epde
    search = epde.EpdeSearch(use_solver=False,
                             verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 1.0, 41)
    x = np.linspace(0.0, 1.0, 41)
    grids = np.meshgrid(t, x, indexing='ij')
    t_, x_ = grids
    data = (np.sin(2 * np.pi * (x_ - 0.2 * t_))
            + 0.5 * np.cos(6 * np.pi * (x_ + 0.2 * t_))
            + 0.3 * np.sin(4 * np.pi * (x_ - 0.2 * t_) + 1.0))
    _, domain = search.createDomain((grids[0], grids[1]), boundary_width=(4, 4), ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': data}, domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(2, 2), data_fun_pow=1)
    return search.pool


def _equation(pool, text, weights):
    """``text`` translated on ``pool`` and pinned to ``weights`` -- the state
    the term sweep hands ``simplify_equation``. The target is the last term."""
    from epde.interface.equation_translator import translate_equation
    eq = translate_equation(text, pool, all_vars=['u']).vals['u']
    eq.main_var_to_explain = 'u'
    for term in eq.structure:
        term.max_factors_in_term = dict(_LIVE_FACTORS)
    eq.metaparameters['max_factors_in_term']['value'] = dict(_LIVE_FACTORS)
    w = np.array(weights, dtype=float)
    eq.weights_internal = w
    eq.weights_final = w.copy()
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    eq.target_idx = len(eq.structure) - 1
    return eq


def _simplify(eq):
    from epde.operators.common.right_part_selection import EqRightPartSelector
    return EqRightPartSelector().simplify_equation(eq)


def _related_features(eq):
    return [term.name for term in eq.structure
            if term is not eq.target and _related_to_target(term, eq.target)]


def _no_draw(self, *args, **kwargs):
    """``Term.randomize`` stand-in: right-part selection draws no terms."""
    raise AssertionError('simplify_equation drew a term')


U_XX = 'd^2u/dx1^2{power: 1.0}'
U_TT = 'd^2u/dx0^2{power: 1.0}'
U_T = 'du/dx0{power: 1.0}'
U_X = 'du/dx1{power: 1.0}'
U = 'u{power: 1.0}'


class TestTheScrubInSimplify:

    def test_the_true_law_is_left_alone(self, wave_pool):
        eq = _equation(wave_pool, f'0.04 * {U_XX} + 0.0 = {U_TT}', [0.04, 0.0])
        before = [term.name for term in eq.structure]
        assert _simplify(eq) is False
        assert [term.name for term in eq.structure] == before

    @pytest.mark.parametrize('text, weights, related', [
        # The hitchhiker: u_t * u_tt is a MULTIPLE of the target u_tt.
        (f'0.04 * {U_XX} + -0.0275 * {U_T} * {U_XX} + 0.687 * {U_T} * {U_TT}'
         f' + 0.0 = {U_TT}', [0.04, -0.0275, 0.687, 0.0], f'{U_T} * {U_TT}'),
        # The padded composite: u_tt is a COMPONENT of the target u * u_tt.
        (f'-0.15 * {U_XX} + 3.75 * {U_TT} + 0.04 * {U} * {U_XX}'
         f' + 0.0 = {U_TT} * {U}', [-0.15, 3.75, 0.04, 0.0], U_TT),
    ], ids=['multiple', 'component'])
    def test_a_related_feature_is_dropped_and_nothing_else(
            self, wave_pool, monkeypatch, text, weights, related):
        """The related feature leaves the structure; nothing is drawn in its
        place (right-part selection uses only its input)."""
        from epde.structure.main_structures import Term
        monkeypatch.setattr(Term, 'randomize', _no_draw)
        eq = _equation(wave_pool, text, weights)
        target = eq.target
        untouched = [term for term in eq.structure if term.name != related]
        assert _related_features(eq) == [related]

        assert _simplify(eq) is True
        assert eq.target is target
        assert _related_features(eq) == []
        # Exactly the input minus the related term: the same objects with the
        # same structure, in order.
        assert [id(t) for t in eq.structure] == [id(t) for t in untouched]
        assert [t.name for t in eq.structure] == [t.name for t in untouched]
        # The structure changed, so the fit is gone.
        assert not eq.weights_internal_evald and not eq.weights_final_evald

    def test_a_reduction_drops_the_dead_copy_and_keeps_the_carrier(self, wave_pool):
        """``u * u_xx`` sheds the common ``u`` and collides with a zero-weight
        ``u_xx`` already in the structure. The dead copy goes; the reduced
        carrier -- the term that holds the identity -- stays."""
        eq = _equation(wave_pool,
                       f'-0.15 * {U} * {U_XX} + 0.0 * {U_XX} + 0.0 * {U_X}'
                       f' + 0.0 = {U} * {U_TT}', [-0.15, 0.0, 0.0, 0.0])
        carrier, dead_copy, bystander, target = eq.structure

        assert _simplify(eq) is True
        assert eq.target is target and target.name == U_TT
        assert any(t is carrier for t in eq.structure) and carrier.name == U_XX
        assert not any(t is dead_copy for t in eq.structure)
        assert any(t is bystander for t in eq.structure)

    def test_every_related_term_is_dropped_with_its_weight_slot(
            self, wave_pool, monkeypatch):
        """Both multiples of the target are dropped, and the weight vector
        loses the matching slot each time instead of drifting out of step with
        the structure."""
        import epde.operators.common.right_part_selection as rps
        from epde.structure.main_structures import Term
        monkeypatch.setattr(Term, 'randomize', _no_draw)
        eq = _equation(wave_pool,
                       f'0.04 * {U_XX} + 0.5 * {U_T} * {U_TT} + 0.7 * {U} * {U_TT}'
                       f' + 0.0 * {U_X} + 0.0 = {U_TT}', [0.04, 0.5, 0.7, 0.0, 0.0])
        target = eq.target
        layouts = []
        stock_drop = rps._drop_term

        def spy_drop(equation, term, **kwargs):
            status = stock_drop(equation, term, **kwargs)
            layouts.append((status, len(equation.structure),
                            len(equation._weights_internal), len(equation._weights_final)))
            return status

        monkeypatch.setattr(rps, '_drop_term', spy_drop)
        assert _simplify(eq) is True
        assert [status for status, *_ in layouts] == ['dropped', 'dropped']
        assert all(n == wi == wf for _, n, wi, wf in layouts)
        assert eq.target is target
        assert [t.name for t in eq.structure] == [U_XX, U_X, U_TT]

    def test_a_scrub_at_the_floor_declines(self, wave_pool, monkeypatch):
        """Two terms cannot lose one: the candidate is the soak alone (the
        target and a multiple of it), so it is declined -- untouched, with
        nothing drawn in the related term's place."""
        from epde.operators.common.right_part_selection import DECLINED
        from epde.structure.main_structures import Term
        monkeypatch.setattr(Term, 'randomize', _no_draw)
        eq = _equation(wave_pool, f'0.5 * {U_T} * {U_TT} + 0.0 = {U_TT}', [0.5, 0.0])
        before = list(eq.structure)
        names = [t.name for t in before]

        assert _simplify(eq) == DECLINED
        assert [id(t) for t in eq.structure] == [id(t) for t in before]
        assert [t.name for t in eq.structure] == names
