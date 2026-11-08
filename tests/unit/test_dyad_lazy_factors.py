#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The dyad_t slab designs, factored per slab on first use (``survival._SlabFactors``).

Pinned here:

* a slab's factor is :func:`dyad_t_design_factors`' bit for bit, whatever is asked for and in what order;
* only the slabs a solve reads are factored, each once;
* dyad_t, dyad and the subset scores are the eager build's bit for bit.
"""
import numpy as np
import pytest

from epde.operators.common import survival
from epde.operators.common.survival import (_SlabFactors, _dyadic_slab_edges, dyad_scores,
                                            dyad_t_design_factors, dyad_t_scores, dyad_t_subset_scores)


class _Eager(object):
    """The eager build behind the lazy interface: every slab factored up front."""

    def __init__(self, B, sample_weights, parts):
        self._R = dyad_t_design_factors(B() if callable(B) else B, sample_weights, parts)

    def __call__(self, slabs):
        return self._R[np.asarray(slabs, dtype=int)]


def _near_collinear(shape, p=4, eps=1e-7, seed=0):
    """Features with one near-collinear pair and one drifting column, on ``shape``:
    slabs the Cholesky solve cannot certify, so the solve reads their designs."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    X = rng.standard_normal((n, p))
    X[:, -1] = X[:, 0] + eps * rng.standard_normal(n)
    X[:, 1] *= 1.0 + 0.5 * np.linspace(0.0, 1.0, n)
    y = X @ rng.standard_normal(p) + 1e-3 * rng.standard_normal(n)
    return X, y, rng.uniform(0.5, 1.5, n)


def _counting(monkeypatch):
    """Swap in a _SlabFactors that records how many slabs it factored."""
    count = {'factored': 0}

    class Counting(_SlabFactors):
        def __call__(self, slabs):
            before = int(self._done.sum())
            out = super().__call__(slabs)
            count['factored'] += int(self._done.sum()) - before
            return out
    monkeypatch.setattr(survival, '_SlabFactors', Counting)
    return count


class TestTheFactors:

    @pytest.mark.parametrize('shape', [(300,), (40, 30)])
    def test_each_slab_is_the_eager_factor(self, shape):
        X, y, w = _near_collinear(shape)
        B = np.hstack([X, np.ones((X.shape[0], 1)), y[:, None]])
        parts = _dyadic_slab_edges(X.shape[0], shape, 24)
        eager = dyad_t_design_factors(B, w, parts)
        lazy = _SlabFactors(B, w, parts)
        rng = np.random.default_rng(3)
        for _ in range(4):                                   # repeats, any order, some already factored
            slabs = rng.integers(0, eager.shape[0], size=7)
            np.testing.assert_array_equal(lazy(slabs), eager[slabs])
        np.testing.assert_array_equal(lazy(np.arange(eager.shape[0])), eager)

    @pytest.mark.parametrize('n, shape', [(320, (320,)), (3000, (3000,)), (99996, (99996,)),
                                          (30000, (20, 30, 50)), (30000, (50, 30, 20)), (1664, (26, 64))])
    @pytest.mark.parametrize('q', [2, 6, 7, 8])
    def test_the_one_gather_stack_is_the_eager_bytes(self, n, shape, q):
        """The slab rows taken in one gather and zero-padded (Oct 27) against the
        eager row-by-row fill: fine-slab and random reads, repeated, -0.0 kept."""
        rng = np.random.default_rng(q)
        B = rng.standard_normal((n, q))
        B[rng.integers(0, n, 50), rng.integers(0, q, 50)] = -0.0
        w = rng.uniform(0.5, 1.5, n)
        parts = _dyadic_slab_edges(n, shape, 4 * (q + 1))
        eager = dyad_t_design_factors(B, w, parts)
        lazy = _SlabFactors(B, w, parts)
        n_slabs = eager.shape[0]
        fine = np.arange(n_slabs - (parts[-1].size - 1), n_slabs)
        for slabs in (rng.choice(fine, size=min(40, fine.size)), rng.integers(0, n_slabs, size=60),
                      np.arange(n_slabs)):
            assert lazy(slabs).tobytes() == eager[slabs].tobytes()

    def test_a_slab_is_factored_once(self):
        X, y, w = _near_collinear((300,))
        parts = _dyadic_slab_edges(300, (300,), 24)
        lazy = _SlabFactors(np.hstack([X, y[:, None]]), w, parts)
        lazy([2, 2, 5])
        assert int(lazy._done.sum()) == 2
        lazy([5, 1])
        assert int(lazy._done.sum()) == 3


class TestTheScores:

    def test_a_well_conditioned_fit_factors_no_slab(self, monkeypatch):
        count = _counting(monkeypatch)
        rng = np.random.default_rng(1)
        X = rng.standard_normal((400, 3))
        dyad_t_scores(X, X @ np.ones(3), np.ones(400), (400,))
        assert count['factored'] == 0

    def test_a_near_collinear_fit_reads_designs(self, monkeypatch):
        count = _counting(monkeypatch)
        X, y, w = _near_collinear((400,))
        dyad_t_scores(X, y, w, (400,))
        assert count['factored'] > 0

    @pytest.mark.parametrize('shape', [(400,), (40, 30)])
    @pytest.mark.parametrize('fit_intercept', [True, False])
    @pytest.mark.parametrize('score', [dyad_t_scores, dyad_scores])
    def test_the_scores_are_the_eager_builds(self, monkeypatch, shape, fit_intercept, score):
        X, y, w = _near_collinear(shape)
        lazy = score(X, y, w, shape, fit_intercept=fit_intercept)
        monkeypatch.setattr(survival, '_SlabFactors', _Eager)
        np.testing.assert_array_equal(lazy, score(X, y, w, shape, fit_intercept=fit_intercept))

    def test_the_subset_scores_are_the_eager_builds(self, monkeypatch):
        X, y, w = _near_collinear((40, 30))
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        subsets = [[0], [1, 2], [0, 3], [0, 4], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
        lazy = dyad_t_subset_scores(X, y, w, (40, 30), subsets)
        monkeypatch.setattr(survival, '_SlabFactors', _Eager)
        np.testing.assert_array_equal(lazy, dyad_t_subset_scores(X, y, w, (40, 30), subsets))
