"""Unit tests for the equation-matching used to score discovered equations."""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks', 'golem'))

import metrics  # noqa: E402

WAVE = '0.04 * d^2u/dx1^2{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}'


def test_identical_equation_matches():
    ok, error = metrics.compare(WAVE, WAVE)
    assert ok and error == pytest.approx(0.0, abs=1e-9)


def test_sides_may_be_swapped():
    """``a = b`` and ``b = a`` are the same equation."""
    flipped = '25.0 * d^2u/dx0^2{power: 1.0} + 0.0 = d^2u/dx1^2{power: 1.0}'
    ok, error = metrics.compare(flipped, WAVE)
    assert ok
    assert error < 1e-6


def test_coefficient_drift_is_reported_not_rejected():
    drifted = '0.0402 * d^2u/dx1^2{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}'
    ok, error = metrics.compare(drifted, WAVE)
    assert ok
    assert 0 < error < 0.01


def test_extra_term_is_a_miss():
    extra = ('0.04 * d^2u/dx1^2{power: 1.0} + 0.3 * du/dx0{power: 1.0} '
             '+ 0.0 = d^2u/dx0^2{power: 1.0}')
    ok, _ = metrics.compare(extra, WAVE)
    assert not ok


def test_numerically_zero_terms_are_dropped():
    padded = ('0.04 * d^2u/dx1^2{power: 1.0} + 0.0 * du/dx0{power: 1.0} '
              '+ 0.0 = d^2u/dx0^2{power: 1.0}')
    ok, _ = metrics.compare(padded, WAVE)
    assert ok


def test_common_factor_only_matches_in_loose_mode():
    """The true law times a redundant token is the same PDE."""
    scaled = ('0.04 * u{power: 3.0} * d^2u/dx1^2{power: 1.0} + 0.0 '
              '= u{power: 3.0} * d^2u/dx0^2{power: 1.0}')
    assert not metrics.compare(scaled, WAVE)[0]
    assert metrics.compare(scaled, WAVE, up_to_factor=True)[0]


def test_frequency_parameter_is_not_part_of_identity():
    a = '1.0 * sin{power: 1.0, freq: 1.9999999, dim: 0.0} + 0.0 = du/dx0{power: 1.0}'
    b = '1.0 * sin{power: 1.0, freq: 2.0000001, dim: 0.0} + 0.0 = du/dx0{power: 1.0}'
    assert metrics.compare(a, b)[0]


def test_equation_lines_splits_a_system():
    text = (' / 1.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}\n'
            ' \\ 2.0 * v{power: 1.0} + 0.0 = dv/dx0{power: 1.0}\n'
            "{'max_terms_number': 5}")
    lines = metrics.equation_lines(text)
    assert len(lines) == 2
    assert lines[0].startswith('1.0 * u')
    assert lines[1].startswith('2.0 * v')


def test_snapshot_hits_finds_the_reference():
    snapshot = [{'text_form': 'junk = du/dx0{power: 1.0}'},
                {'text_form': WAVE + "\n{'sparsity': 1e-6}"}]
    assert metrics.snapshot_hits(snapshot, [WAVE])
    assert not metrics.snapshot_hits(snapshot[:1], [WAVE])
