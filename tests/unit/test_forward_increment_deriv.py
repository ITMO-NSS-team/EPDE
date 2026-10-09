#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward-increment (Ito) time derivatives: the 'forward' preprocessor.

For stochastic data a time-symmetric derivative cannot see a drift that is the
gradient of a function of the state: at stationarity
``E[u' g(u)] = d/dt E[G(u)] = 0`` for ``g = G'``, so confinement or repulsion
fits with coefficient ~0 under the central/polynomial preprocessors. The
forward increment ``(u[k+lag] - u[k]) / (lag dt)`` is not symmetric in time and
estimates the drift. These tests pin the difference rule, keep the central rule
bit-identical, show the drift-visibility property on an Ornstein-Uhlenbeck
process, and check the boundary guard that keeps the undefined tail out of fits.
"""

import numpy as np
import pytest

import epde
from epde.preprocessing.deriv_calculators import AdaptiveFiniteDeriv, ForwardIncrementDeriv


def _forward(u, t, order=1, lag=1, time_axis=0):
    return ForwardIncrementDeriv()(u, [t], order, lag=lag, time_axis=time_axis)


class TestDifferenceRule:

    def test_exact_on_a_line_including_the_padded_tail(self):
        t = np.linspace(0.0, 2.0, 41)
        for lag in (1, 3):
            d = _forward(3.0 * t + 1.0, t, lag=lag)
            assert np.allclose(d[:, 0], 3.0)

    def test_quadratic_matches_the_forward_quotient(self):
        t = np.linspace(0.0, 1.0, 101)
        dt, lag = t[1] - t[0], 4
        d = _forward(t ** 2, t, lag=lag)[:, 0]
        assert np.allclose(d[:-lag], 2 * t[:-lag] + lag * dt)

    def test_second_order_is_the_repeated_forward_difference(self):
        t = np.linspace(0.0, 1.0, 101)
        lag = 2
        d = _forward(t ** 2, t, order=2, lag=lag)
        assert np.allclose(d[:-2 * lag, 1], 2.0)

    def test_other_axes_stay_central(self):
        t = np.linspace(0.0, 1.0, 30)
        x = np.linspace(0.0, 2 * np.pi, 40)
        u = np.outer(t, np.sin(x))
        d = ForwardIncrementDeriv()(u, [t, x], (1, 1), lag=1, time_axis=0)
        assert np.allclose(d[:, 1], np.gradient(u, x, axis=1).ravel())
        dt_fwd = np.empty_like(u)
        dt_fwd[:-1] = (u[1:] - u[:-1]) / (t[1] - t[0]); dt_fwd[-1] = dt_fwd[-2]
        assert np.allclose(d[:, 0], dt_fwd.ravel())

    @pytest.mark.parametrize('lag', [0, -1, 1.5])
    def test_lag_must_be_a_positive_integer(self, lag):
        t = np.linspace(0.0, 1.0, 20)
        with pytest.raises(ValueError, match='positive integer'):
            _forward(t, t, lag=lag)


def test_central_rule_is_unchanged():
    """``AdaptiveFiniteDeriv`` now routes through ``_grad``; its output must stay np.gradient's."""
    t = np.linspace(0.0, 1.0, 50)
    u = np.sin(5 * t) + t ** 3
    d = AdaptiveFiniteDeriv()(u, [t], 2)
    g1 = np.gradient(u, t)
    assert np.array_equal(d[:, 0], g1)
    assert np.array_equal(d[:, 1], np.gradient(g1, t))


def test_gradient_drift_is_visible_forward_and_invisible_central():
    """Ornstein-Uhlenbeck  dx = -k x dt + s dW: regressing the derivative on x recovers -k only
    with forward increments; the central difference gives ~0 for any stationary signal."""
    rng = np.random.default_rng(0)
    k, s, dt, n = 2.0, 1.0, 0.01, 20000
    x = np.empty(n); x[0] = 0.0
    a = np.exp(-k * dt); sd = s * np.sqrt((1 - a ** 2) / (2 * k))
    for i in range(1, n):
        x[i] = a * x[i - 1] + sd * rng.standard_normal()
    t = np.arange(n) * dt
    inner = slice(5, n - 5)

    def slope(d):
        xi, di = x[inner] - x[inner].mean(), d[inner] - d[inner].mean()
        return np.dot(xi, di) / np.dot(xi, xi)

    forward = slope(_forward(x, t)[:, 0])
    central = slope(AdaptiveFiniteDeriv()(x, [t], 1)[:, 0])
    assert abs(forward - (a - 1) / dt) < 0.6     # discrete-time drift, ~ -1.98; SE ~0.14
    assert abs(central) < 0.6                    # ~0 by time symmetry


def _search(boundary, lag):
    search = epde.EpdeSearch(use_solver=False, multiobjective_mode=True, device='cpu')
    search.set_preprocessor(default_preprocessor_type='forward', preprocessor_kwargs={'lag': lag})
    t = np.linspace(0.0, 1.0, 80)
    _, domain = search.createDomain((t,), boundary_width=boundary, ID=0)
    _, trajectory = search.createTrajectory({'u': np.sin(3 * t)}, domain, cache_id=0)
    return search, trajectory


class TestInterfaceAndBoundary:

    def test_forward_key_builds_the_unsmoothed_forward_pipeline(self):
        search, _ = _search(boundary=3, lag=3)
        pipe = search.preprocessor_pipeline
        assert isinstance(pipe.deriv_calculator, ForwardIncrementDeriv)
        assert pipe.deriv_calculator_kwargs['lag'] == 3
        assert pipe.deriv_calculator_kwargs['time_axis'] == 0
        assert type(pipe.smoother).__name__ == 'PlaceholderSmoother'

    def test_boundary_narrower_than_the_undefined_tail_raises(self):
        _, trajectory = _search(boundary=2, lag=3)
        with pytest.raises(ValueError, match='boundary_width >= 3'):
            trajectory.build(max_deriv_order=(1,), data_fun_pow=1, deriv_fun_pow=1)

    def test_second_order_needs_twice_the_tail(self):
        _, trajectory = _search(boundary=3, lag=3)
        with pytest.raises(ValueError, match='boundary_width >= 6'):
            trajectory.build(max_deriv_order=(2,), data_fun_pow=1, deriv_fun_pow=1)

    def test_wide_enough_boundary_builds_forward_derivatives(self):
        _, trajectory = _search(boundary=3, lag=3)
        trajectory.build(max_deriv_order=(1,), data_fun_pow=1, deriv_fun_pow=1)
        t = np.linspace(0.0, 1.0, 80)
        u = np.sin(3 * t)
        stored = np.asarray(trajectory._entries[0].derivatives)[:, 0]
        assert np.allclose(stored[:-3], (u[3:] - u[:-3]) / (t[3] - t[0]))
