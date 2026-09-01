#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``_target_term_in_other_equation`` with the DERIV-CORE extension: a
decorated target (``dv/dx0 * cos``) reserves its derivative core exactly
as a bare ``dv/dx0`` target would -- the lv coupled-junk mechanism, where
the cos-costumed v-target unlocked standalone ``v_t`` for the u-equation's
dominating sum/differentiated identities."""

from types import SimpleNamespace

from epde.evaluators import simple_function_evaluator
from epde.operators.common.right_part_selection import (
    _target_term_in_other_equation)


def _deriv_factor(label):
    return SimpleNamespace(
        structural_label=(label, (1,)),
        is_deriv=True, deriv_code=[0],
        evaluator=SimpleNamespace(_evaluator=simple_function_evaluator))


def _plain_factor(label):
    return SimpleNamespace(
        structural_label=(label, (1,)),
        is_deriv=False, deriv_code=[None, ],
        evaluator=SimpleNamespace(_evaluator=simple_function_evaluator))


def _term(*factors):
    return SimpleNamespace(
        structure=list(factors),
        factors_labels=frozenset(f.structural_label for f in factors))


def _eq_with_target(target_term):
    return SimpleNamespace(target=target_term)


def _eq_other(*terms):
    return SimpleNamespace(
        active_terms_labels=frozenset(t.factors_labels for t in terms))


V_T = _deriv_factor('dv/dx0')
COS = _plain_factor('cos')
U = _plain_factor('u')


class TestTargetCoreBan:

    def test_pure_target_whole_term_leak_still_caught(self):
        tgt = _term(_deriv_factor('dv/dx0'))
        other = _eq_other(_term(_deriv_factor('dv/dx0')), _term(U))
        assert (_target_term_in_other_equation(_eq_with_target(tgt), other)
                == tgt.factors_labels)

    def test_decorated_target_core_leak_caught(self):
        # v-eq target dv/dx0 * cos; u-eq carries standalone dv/dx0 --
        # the lv junk pair's enabling leak.
        tgt = _term(V_T, COS)
        core = _term(_deriv_factor('dv/dx0'))
        other = _eq_other(core, _term(U))
        assert (_target_term_in_other_equation(_eq_with_target(tgt), other)
                == core.factors_labels)

    def test_core_as_factor_of_composite_stays_legal(self):
        # dv/dx0 only inside u * dv/dx0 (a coupling term): NOT a leak --
        # the whole-term-only compromise (the NS v*v_y case).
        tgt = _term(V_T, COS)
        other = _eq_other(_term(U, _deriv_factor('dv/dx0')), _term(U))
        assert _target_term_in_other_equation(
            _eq_with_target(tgt), other) is None

    def test_unrelated_derivative_not_banned(self):
        # d^2v/dx0^2 standalone is NOT the core of dv/dx0 * cos.
        tgt = _term(V_T, COS)
        other = _eq_other(_term(_deriv_factor('d^2v/dx0^2')), _term(U))
        assert _target_term_in_other_equation(
            _eq_with_target(tgt), other) is None

    def test_no_target_returns_none(self):
        other = _eq_other(_term(U))
        assert _target_term_in_other_equation(
            SimpleNamespace(target=None), other) is None
