#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-trajectory tensors for a declared token family.

``createTrajectory(cached_token_tensors={family: {label: tensor}})`` is how a
family whose values differ between samples (a per-particle interaction field,
say) feeds each trajectory its own tensor. ``PreparedTokens.__hash__`` returned
the family's type STRING, and ``__hash__`` must return an int, so building
that dict raised ``TypeError`` -- the path had never been usable.
"""

import numpy as np

import epde
from epde import CacheStoredTokens


def test_a_token_family_is_hashable():
    epde.EpdeSearch(use_solver=False, multiobjective_mode=True, device='cpu')
    fam = CacheStoredTokens(token_type='interaction', token_labels=['f'], token_tensors=None,
                            params_ranges={'power': (1, 1)}, params_equality_ranges=None)
    assert isinstance(hash(fam), int)
    assert {fam: 1}[fam] == 1


def test_each_trajectory_reads_its_own_tensor():
    search = epde.EpdeSearch(use_solver=False, multiobjective_mode=True, device='cpu')
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0.0, 1.0, 50)
    _, domain = search.createDomain((t,), boundary_width=3, ID=0)
    fam = CacheStoredTokens(token_type='interaction', token_labels=['f'], token_tensors=None,
                            params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                            meaningful=True)
    values = [np.sin(4 * t), np.cos(7 * t)]
    trajectories = []
    for i, f in enumerate(values):
        _, traj = search.createTrajectory({'u': t ** 2 + i}, domain, cache_id=i,
                                          cached_token_tensors={fam: {'f': f}})
        traj.build(max_deriv_order=(1,), data_fun_pow=1, deriv_fun_pow=1)
        trajectories.append(traj)
    inner = slice(3, -3)
    for traj, f in zip(trajectories, values):
        stored = np.asarray(traj.get(('f', (1.0,)))).ravel()
        assert np.allclose(stored, f[inner])
