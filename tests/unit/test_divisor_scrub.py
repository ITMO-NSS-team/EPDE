#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``_term_divides`` -- the ratio-fit scrub predicate in
``simplify_equation``: a feature term divides the target iff every one of
its factors appears in the target with at least the same power. Divisor
features make the ``Lambda * (identity)`` FD-error soak representable
(the wave trap ``u*u_tt = 3.75*u_tt - 0.15*u_xx + 0.04*u*u_xx``)."""

from types import SimpleNamespace

from epde.operators.common.right_part_selection import _term_divides


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
