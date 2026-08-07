#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The chi2-scored keep-rule default in ``PhysicsInformedLasso.fit``:
``gram_mode='chi2'`` (the default) builds NO window or varying-coefficient
machinery -- vcoef is the only mode with its own Gram, and runs ONLY on
explicit selection."""

import numpy as np
import pytest

import epde.globals as global_var
import epde.operators.common.sparsity as sparsity_module
from epde.operators.common.sparsity import PhysicsInformedLasso


def _problem(n=240):
    x = np.linspace(0.0, 4.0 * np.pi, n)
    f1 = np.sin(x)
    f2 = np.cos(3.0 * x)
    # Non-exact fit so chi2_scores stays off its exact-fit floor.
    y = 2.0 * f1 + 0.05 * np.cos(7.0 * x)
    return np.column_stack([f1, f2]), y, (n,)


class TestChi2KeepRuleDefault:

    def teardown_method(self):
        global_var.set_gram_config('chi2')

    def test_default_gram_mode_is_chi2(self):
        assert global_var.gram_mode == 'chi2'

    def test_menu_accepts_all_three_and_rejects_unknown(self):
        for value in ('axis', 'vcoef', 'chi2'):
            global_var.set_gram_config(value)
            assert global_var.gram_mode == value
        with pytest.raises(ValueError):
            global_var.set_gram_config('bogus')

    def test_default_fit_scores_via_chi2_and_builds_no_setup(
            self, monkeypatch):
        calls = {'n': 0}
        real = sparsity_module.chi2_scores

        def spy(*args, **kwargs):
            calls['n'] += 1
            return real(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError(
                'no Gram setup may be built under the chi2 default')

        monkeypatch.setattr(sparsity_module, 'chi2_scores', spy)
        monkeypatch.setattr(sparsity_module, 'VaryingCoefSetup', forbidden)
        monkeypatch.setattr(sparsity_module, 'GramSetup', forbidden)
        X, y, gshape = _problem()
        est = PhysicsInformedLasso(grid_shape=gshape)
        est.fit(X, y)
        assert calls['n'] >= 1
        assert est.cached_vc_score_ is None
        assert est.cached_weights_ is None
        assert est.coef_.shape == (2,)
        # The true column's relaxed-refit coefficient survives near 2.0.
        assert est.coef_[0] == pytest.approx(2.0, rel=0.2)

    def test_explicit_vcoef_still_builds_the_vc_machinery(self):
        global_var.set_gram_config('vcoef')
        X, y, gshape = _problem()
        est = PhysicsInformedLasso(grid_shape=gshape)
        est.fit(X, y)
        assert est.cached_vc_score_ is not None
