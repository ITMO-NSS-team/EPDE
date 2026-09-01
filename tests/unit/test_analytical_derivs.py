#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-computed derivatives are per VARIABLE, not per trajectory.

A derivative tensor describes one variable -- ``(n_points, n_derivs)``, the
columns ordered as ``define_derivatives`` orders them -- so a system of
several variables needs one tensor each. ``Trajectory.build`` used to hand
the whole ``derivs`` argument to every entry, which made the whole
pre-computed path single-variable only:

* a list raised ``AttributeError: 'list' object has no attribute 'T'``
  further down, in ``addEntryToCache``;
* a bare array was accepted and gave EVERY variable the FIRST variable's
  derivatives -- silently, and only detectable by the nonsense that came out
  of the search afterwards.

This matters because supplying derivatives is how a comparison of
statistical or sparsity methods avoids measuring the differentiation scheme:
with the analytical right-hand side as the target, the true equation's
residual is zero by construction, so nothing can undercut it by soaking
finite-difference error.
"""

import numpy as np
import pytest

import epde


A = B = C = E = 20.0


@pytest.fixture
def lv_sample():
    """A short Lotka-Volterra sample and its ANALYTICAL derivatives.

    The trajectory need not solve the ODE for this to be exact: the target is
    the model's right-hand side evaluated at the very samples the feature
    columns are built from, so ``du/dt = A*u - B*u*v`` holds pointwise to
    machine precision whatever the data is.
    """
    t = np.linspace(0.0, 0.5, 60)
    u = 2.0 + 0.5 * np.sin(4.0 * t)
    v = 1.0 + 0.3 * np.cos(3.0 * t)
    du = (A * u - B * u * v).reshape(-1, 1)
    dv = (-C * v + E * u * v).reshape(-1, 1)
    return t, u, v, du, dv


def _trajectory(lv_sample, derivs):
    t, u, v, _, _ = lv_sample
    search = epde.EpdeSearch(use_solver=False, multiobjective_mode=True,
                             device='cpu')
    _, domain = search.createDomain((t,), boundary_width=5, ID=0)
    search.set_preprocessor(default_preprocessor_type='poly',
                            preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': u, 'v': v}, domain,
                                            cache_id=0, derivs=derivs)
    trajectory.build(max_deriv_order=(1,), data_fun_pow=1, deriv_fun_pow=1)
    return trajectory


class TestEachVariableGetsItsOwn:

    def test_dict_form_keyed_by_variable(self, lv_sample):
        _, _, _, du, dv = lv_sample
        trajectory = _trajectory(lv_sample, {'u': du, 'v': dv})
        stored = {entry.var_name: np.asarray(entry.derivatives)
                  for entry in trajectory._entries}
        assert np.allclose(stored['u'].ravel(), du.ravel())
        assert np.allclose(stored['v'].ravel(), dv.ravel())

    def test_list_form_in_entry_order(self, lv_sample):
        _, _, _, du, dv = lv_sample
        trajectory = _trajectory(lv_sample, [du, dv])
        stored = [np.asarray(e.derivatives) for e in trajectory._entries]
        assert np.allclose(stored[0].ravel(), du.ravel())
        assert np.allclose(stored[1].ravel(), dv.ravel())

    def test_the_two_variables_do_not_share_a_tensor(self, lv_sample):
        _, _, _, du, dv = lv_sample
        trajectory = _trajectory(lv_sample, {'u': du, 'v': dv})
        stored = [np.asarray(e.derivatives).ravel()
                  for e in trajectory._entries]
        # The regression: identical columns meant one variable's derivative
        # had been copied onto the other.
        assert not np.allclose(stored[0], stored[1])


class TestWrongShapesFailLoudly:

    def test_one_tensor_for_two_variables_raises(self, lv_sample):
        """The silent case. Accepting this gave ``v`` the derivative of
        ``u``, and the search then optimised against a target that was not
        the variable it claimed to explain."""
        _, _, _, du, _ = lv_sample
        with pytest.raises(ValueError, match='same derivatives'):
            _trajectory(lv_sample, du)

    def test_wrong_length_list_raises(self, lv_sample):
        _, _, _, du, _ = lv_sample
        with pytest.raises(ValueError, match='one per variable'):
            _trajectory(lv_sample, [du])

    def test_dict_missing_a_variable_raises(self, lv_sample):
        _, _, _, du, _ = lv_sample
        with pytest.raises(KeyError, match='v'):
            _trajectory(lv_sample, {'u': du})


def test_a_single_variable_trajectory_still_takes_a_bare_array(lv_sample):
    """One variable, one tensor: no ambiguity, so no error."""
    t, u, _, du, _ = lv_sample
    search = epde.EpdeSearch(use_solver=False, multiobjective_mode=True,
                             device='cpu')
    _, domain = search.createDomain((t,), boundary_width=5, ID=0)
    search.set_preprocessor(default_preprocessor_type='poly',
                            preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': u}, domain, cache_id=0,
                                            derivs=du)
    trajectory.build(max_deriv_order=(1,), data_fun_pow=1, deriv_fun_pow=1)
    stored = np.asarray(trajectory._entries[0].derivatives)
    assert np.allclose(stored.ravel(), du.ravel())
