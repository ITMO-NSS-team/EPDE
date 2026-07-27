#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the ``Instability`` objective filler in
``epde/operators/common/objectives.py``.

Pins the fail-loud contract (user directive): ``Instability.compute`` must
NOT swallow exceptions and return a silent fallback value -- a quiet 1.0
neither kills nor flags the candidate and corrupts the Pareto front
invisibly. Any exception raised inside the computation propagates.

Uses lightweight mocks (the ``test_eq_mo_objectives.py`` convention) so the
filler logic is exercised without the full pool/token machinery.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.operators.common.objectives import FitContext, Instability


def _ctx(data_shape=(50,)):
    return FitContext(g_fun_vals=None, data_shape=data_shape,
                      penalty_coeff=0.2)


class _RaisingEquation:
    """Malformed equation: evaluation blows up (e.g. a stale cache label or
    a shape mismatch deep in the tensor cache)."""

    def evaluate(self, normalize, return_val):
        raise ValueError('malformed equation: evaluation failed')


class TestInstabilityFailsLoudly:

    def test_exception_from_evaluate_propagates(self):
        # Pre-fix behavior: any exception was swallowed and compute()
        # returned the silent mid-range 1.0.
        with pytest.raises(ValueError, match='malformed equation'):
            Instability().compute(_RaisingEquation(), _ctx())

    def test_missing_weights_propagates_attribute_error(self):
        # An equation that evaluates but lacks fitted weights is a
        # contract violation upstream -- it must surface, not score 1.0.
        eq = SimpleNamespace(
            evaluate=lambda normalize, return_val: (
                None, np.ones(50), np.ones((50, 2))),
        )
        with pytest.raises(AttributeError):
            Instability().compute(eq, _ctx())


class TestInstabilityLegitimatePaths:

    def test_cached_vc_score_fast_path(self):
        # The sparsity-pass cache short-circuits everything else.
        eq = SimpleNamespace(_cached_vc_score=np.array([0.1, 0.2, 0.3]))
        assert Instability().compute(eq, _ctx()) == pytest.approx(0.6)

    def test_features_none_is_a_legitimate_one(self):
        # Only the target term survived: not an error, scores 1.0 by design.
        eq = SimpleNamespace(
            evaluate=lambda normalize, return_val: (None, np.ones(50), None),
        )
        assert Instability().compute(eq, _ctx()) == pytest.approx(1.0)
