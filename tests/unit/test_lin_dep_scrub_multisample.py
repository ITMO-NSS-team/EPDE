#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multisample semantics of ``EqRightPartSelector._regenerate_dependent_terms``
-- the exact-linear-dependence scrub.

The coefficient fit runs PER TRAJECTORY and the resulting vectors are averaged
(``VWSRSparsity.apply``), so a rank-deficient solve in ONE sample already
corrupts that sample's coefficients and therefore the average. The scrub
consequently:

* flags a term whose column is exactly dependent in ANY sample (union), and
* accepts a regenerated replacement only when it is independent in EVERY
  sample.

Trajectories legitimately differ in length and grid, so nothing may be
concatenated across samples and no cross-sample agreement may be asserted.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.operators.common.right_part_selection import (
    EqRightPartSelector, _flag_dependent_columns, _LIN_DEP_RTOL)


# Two trajectories of DIFFERENT length: any implementation that stacks their
# columns (or demands they flag the same indices) breaks here.
KEYS = (0, 1)
N0, N1 = 12, 9


def _cols(a, b):
    """A term's per-sample column dict."""
    return {0: np.asarray(a, dtype=float), 1: np.asarray(b, dtype=float)}


class FakeTerm:
    """Minimal Term stand-in: an evaluable column pair plus the identity /
    regeneration surface ``_regen_or_drop_term`` touches."""

    def __init__(self, name, cols, replacements=()):
        self.name = name
        self._cols = cols
        self._replacements = list(replacements)
        self.randomize_calls = 0
        self.structure = [object()]

    # -- evaluation ---------------------------------------------------- #
    def evaluate(self, *args, **kwargs):
        return {key: value.copy() for key, value in self._cols.items()}

    # -- identity ------------------------------------------------------ #
    @property
    def factors_labels(self):
        return frozenset({(self.name, self.randomize_calls)})

    def contains_meaningful(self):
        return True

    # -- regeneration -------------------------------------------------- #
    def randomize(self):
        self.randomize_calls += 1
        if self._replacements:
            self._cols = self._replacements.pop(0)

    def resetSavedState(self):
        pass


class FakeEquation:
    def __init__(self, structure, target):
        self.structure = list(structure)
        self.target = target
        self.resets = 0
        self.invalidations = 0

    @property
    def target_idx(self):
        return self.structure.index(self.target)

    def _invalidate_label_cache(self):
        self.invalidations += 1

    def reset_state(self, reset_right_part=True):
        self.resets += 1


def _independent_pair(seed):
    rng = np.random.default_rng(seed)
    return _cols(rng.normal(size=N0), rng.normal(size=N1))


def _run(feature_terms, target=None):
    target = target or FakeTerm('target', _independent_pair(99))
    equation = FakeEquation(list(feature_terms) + [target], target)
    changed = EqRightPartSelector()._regenerate_dependent_terms(
        equation, list(feature_terms))
    return changed, equation


class TestLinDepScrubMultisample:

    def test_no_dependence_leaves_structure_alone(self):
        terms = [FakeTerm('a', _independent_pair(1)),
                 FakeTerm('b', _independent_pair(2))]
        changed, equation = _run(terms)
        assert changed is False
        assert equation.resets == 0
        assert all(term.randomize_calls == 0 for term in terms)

    def test_constant_column_flagged_in_every_sample(self):
        # A constant column is exactly dependent on the intercept, which
        # ``_flag_dependent_columns`` seeds the basis with.
        const = FakeTerm('const', _cols(np.full(N0, 2.0), np.full(N1, 2.0)),
                         replacements=[_independent_pair(3)])
        free = FakeTerm('free', _independent_pair(4))
        changed, equation = _run([const, free])
        assert changed is True
        assert const.randomize_calls == 1
        assert free.randomize_calls == 0

    def test_dependence_in_a_single_sample_is_enough(self):
        # Varies in sample 0, constant in sample 1 -> dependent in sample 1
        # ONLY. The union rule must still flag it; a cross-sample agreement
        # assert (or an all()-guard) would let it ride.
        rng = np.random.default_rng(5)
        partial = FakeTerm('partial',
                           _cols(rng.normal(size=N0), np.full(N1, -1.5)),
                           replacements=[_independent_pair(6)])
        changed, equation = _run([partial])
        assert changed is True
        assert partial.randomize_calls == 1

    def test_replacement_must_be_independent_in_every_sample(self):
        # First draw is independent in sample 0 but CONSTANT in sample 1 --
        # rejected. Second draw is independent in both -- accepted.
        rng = np.random.default_rng(7)
        half_good = _cols(rng.normal(size=N0), np.full(N1, 4.0))
        good = _independent_pair(8)
        const = FakeTerm('const', _cols(np.zeros(N0), np.zeros(N1)),
                         replacements=[half_good, good])
        changed, equation = _run([const])
        assert changed is True
        assert const.randomize_calls == 2
        for key in KEYS:
            np.testing.assert_allclose(const.evaluate()[key], good[key])

    def test_samples_of_different_length_never_concatenated(self):
        # Purely structural: a column pair whose lengths differ must be
        # processed without raising (concatenating them would build a ragged
        # design and a residual incomparable with _LIN_DEP_RTOL).
        terms = [FakeTerm('a', _cols(np.arange(N0, dtype=float),
                                     np.arange(N1, dtype=float)))]
        changed, _ = _run(terms)
        assert changed is False

    def test_non_finite_column_skips_rather_than_regenerates(self):
        bad = FakeTerm('bad', _cols(np.full(N0, np.nan), np.ones(N1)))
        changed, equation = _run([bad])
        assert changed is False
        assert bad.randomize_calls == 0

    def test_flag_dependent_columns_contract_unchanged(self):
        # The per-sample primitive itself is untouched by the multisample
        # rewrite: intercept-seeded basis, zero column flagged, exact
        # threshold on unit-normalised residuals.
        cols = [np.full(6, 3.0), np.zeros(6), np.arange(6, dtype=float)]
        flagged, basis = _flag_dependent_columns(cols)
        assert flagged == [0, 1]
        assert len(basis) == 2                     # intercept + the ramp
        assert _LIN_DEP_RTOL == pytest.approx(1e-10)
