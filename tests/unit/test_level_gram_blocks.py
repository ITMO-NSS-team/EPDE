#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``survival._level_gram_blocks`` without the folded constant, and the paired Gram.

Pinned here:

* ``append_constant=False`` is the augmented Gram's leading block byte for byte, on every path (one
  row per level, the slab loop, whole multi-row levels), for C- and F-ordered designs, and so are
  the coarser partitions summed from it;
* a design served the Gram kept for its version without the ones column (plain mode's S and
  S+{1}) gets the bytes it builds itself, and a different design, target, weights, slabs or zero
  sign builds its own;
* dyad_t's and dyad's scores are the augmented-then-sliced build's bytes, paired or not.
"""
import numpy as np
import pytest

from epde.operators.common import survival
from epde.operators.common.survival import (_DESIGN_SOLVES, _dyadic_slab_edges, _level_gram_blocks,
                                            _median_axis0, _reference_width, _solve_min_norm,
                                            dyad_scores, dyad_t_design_factors, dyad_t_scores)


@pytest.fixture(autouse=True)
def _no_kept_grams(monkeypatch):
    monkeypatch.setattr(survival, '_PAIRED_GRAMS', [])


def _bytes_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def _design(n, p, seed=0, order='C'):
    """Columns on four decades of scale, a few -0.0 entries, a noisy linear target."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * np.logspace(-2, 2, p)
    X[rng.integers(0, n, 5), 0] = -0.0
    y = X @ rng.standard_normal(p) + 0.1 * rng.standard_normal(n)
    return np.asarray(X, order=order), y, rng.uniform(0.5, 1.5, n)


def _with_ones(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def _augmented_dyad_t_scores(features, target, sample_weights, grid_shape, fit_intercept=True):
    """dyad_t_scores before the constant-free Gram (Oct 27): the augmented Grams of every
    partition, sliced to the fit's width, and the slab designs factored up front."""
    X = np.asarray(features, dtype=float)
    n_samples, p = X.shape
    parts = _dyadic_slab_edges(n_samples, grid_shape, 4 * (_reference_width(p) + 1))
    fine = parts[-1]
    G_fine, Gy_fine = _level_gram_blocks(X, target, sample_weights, grid_shape, fine)
    sl = slice(None) if fit_intercept else slice(0, p)
    G_all, Gy_all = [], []
    for edges in parts:
        pos = np.searchsorted(fine, edges[:-1])
        G_all.append(np.add.reduceat(G_fine, pos, axis=0)[:, sl, sl])
        Gy_all.append(np.add.reduceat(Gy_fine, pos, axis=0)[:, sl])
    design = None
    if survival._DYAD_SOLVE in _DESIGN_SOLVES:
        R_all = dyad_t_design_factors(np.hstack([X, np.ones((n_samples, 1)),
                                                 np.asarray(target, dtype=float).reshape(-1, 1)]),
                                      sample_weights, parts)
        cols = np.arange(p + 1 if fit_intercept else p)

        def design(rows):
            R = R_all[np.asarray(rows, dtype=int)]
            return R[:, :, cols], R[:, :, p + 1]
    theta = _solve_min_norm(np.concatenate(G_all), np.concatenate(Gy_all), design)[:, :p]
    total = np.zeros(p)
    start = 0
    for edges in parts:
        coefs = theta[start:start + edges.size - 1]
        start += edges.size - 1
        med = _median_axis0(coefs)
        mad = _median_axis0(np.abs(coefs - med[None, :]))
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = mad / np.abs(med)
        total += np.where(np.isnan(rel), 0.0, rel)
    with np.errstate(over='ignore'):
        return np.nan_to_num(total / len(parts))


PATHS = [((1200,), 'one row per level'), ((1200,), 'slab loop'), ((40, 30), 'whole levels'),
         ((6, 5, 40), 'whole levels')]


class TestTheConstantFreeGram:

    @pytest.mark.parametrize('order', ['C', 'F'])
    @pytest.mark.parametrize('weighted', [True, False])
    @pytest.mark.parametrize('grid, path', PATHS)
    def test_it_is_the_augmented_leading_block(self, monkeypatch, order, weighted, grid, path):
        if path == 'slab loop':
            monkeypatch.setattr(survival, '_MAX_OUTER_PRODUCT_DOUBLES', 0)
        n = int(np.prod(grid))
        X, y, w = _design(n, 4, order=order)
        w = w if weighted else None
        edges = _dyadic_slab_edges(n, grid, 28)[-1]
        G, Gy = _level_gram_blocks(X, y, w, grid, edges)
        G0, Gy0 = _level_gram_blocks(X, y, w, grid, edges, append_constant=False)
        assert G.shape[1:] == (5, 5) and G0.shape[1:] == (4, 4)
        assert _bytes_equal(G0, G[:, :4, :4]) and _bytes_equal(Gy0, Gy[:, :4])

    @pytest.mark.parametrize('grid, path', PATHS)
    def test_the_coarser_partitions_sum_the_same_bytes(self, monkeypatch, grid, path):
        if path == 'slab loop':
            monkeypatch.setattr(survival, '_MAX_OUTER_PRODUCT_DOUBLES', 0)
        n = int(np.prod(grid))
        X, y, w = _design(n, 3, seed=1, order='F')
        parts = _dyadic_slab_edges(n, grid, 28)
        fine = parts[-1]
        G, Gy = _level_gram_blocks(X, y, w, grid, fine)
        G0, Gy0 = _level_gram_blocks(X, y, w, grid, fine, append_constant=False)
        for edges in parts:
            pos = np.searchsorted(fine, edges[:-1])
            assert _bytes_equal(np.add.reduceat(G0, pos, axis=0), np.add.reduceat(G, pos, axis=0)[:, :3, :3])
            assert _bytes_equal(np.add.reduceat(Gy0, pos, axis=0), np.add.reduceat(Gy, pos, axis=0)[:, :3])


class TestThePairedGram:

    def _setup(self, n=None, p=3):
        n = survival._PAIRED_MIN_ROWS if n is None else n
        X, y, w = _design(n, p, seed=2)
        return X, y, w, _dyadic_slab_edges(n, (n,), 28)[-1]

    def test_the_version_with_the_ones_column_is_served_its_own_bytes(self):
        X, y, w, edges = self._setup()
        n = X.shape[0]
        S1 = _with_ones(X)
        own = _level_gram_blocks(S1, y, w, (n,), edges, append_constant=False)
        assert survival._PAIRED_GRAMS == [], 'a design with a ones column is never kept'
        S = _level_gram_blocks(X, y, w, (n,), edges, append_constant=False)
        assert len(survival._PAIRED_GRAMS) == 1
        served = _level_gram_blocks(S1, y, w, (n,), edges, append_constant=False)
        kept = survival._PAIRED_GRAMS[0]
        assert served[0] is kept[4] and served[1] is kept[5]
        assert not served[0].flags.writeable and not served[1].flags.writeable
        assert _bytes_equal(served[0], own[0]) and _bytes_equal(served[1], own[1])
        augmented = _level_gram_blocks(S1, y, w, (n,), edges)
        assert _bytes_equal(served[0], augmented[0][:, :4, :4]) and _bytes_equal(served[1], augmented[1][:, :4])
        assert _bytes_equal(S[0], augmented[0][:, :3, :3]) and _bytes_equal(S[1], augmented[1][:, :3])

    @pytest.mark.parametrize('change', ['target', 'weights', 'slabs', 'design', 'zero sign'])
    def test_a_different_version_builds_its_own(self, change):
        X, y, w, edges = self._setup()
        n = X.shape[0]
        _level_gram_blocks(X, y, w, (n,), edges, append_constant=False)
        S1 = _with_ones(X)
        if change == 'target':
            y = y.copy()
            y[7] = np.nextafter(y[7], np.inf)
        elif change == 'weights':
            w = w.copy()
            w[3] = np.nextafter(w[3], np.inf)
        elif change == 'slabs':
            edges = _dyadic_slab_edges(n, (n,), 28)[-2]
        elif change == 'design':
            S1[5, 1] = np.nextafter(S1[5, 1], np.inf)
        else:
            zeros = np.flatnonzero(np.signbit(S1[:, 0]) & (S1[:, 0] == 0.0))
            assert zeros.size, 'fixture needs a -0.0'
            S1[zeros[0], 0] = 0.0
        G, Gy = _level_gram_blocks(S1, y, w, (n,), edges, append_constant=False)
        assert all(G is not kept[4] for kept in survival._PAIRED_GRAMS)
        survival._PAIRED_GRAMS.clear()
        G_own, Gy_own = _level_gram_blocks(S1, y, w, (n,), edges, append_constant=False)
        assert _bytes_equal(G, G_own) and _bytes_equal(Gy, Gy_own)

    def test_below_the_row_floor_nothing_is_kept(self):
        X, y, w, edges = self._setup(n=survival._PAIRED_MIN_ROWS - 1)
        _level_gram_blocks(X, y, w, (X.shape[0],), edges, append_constant=False)
        assert survival._PAIRED_GRAMS == []

    def test_the_slab_loop_keeps_nothing(self, monkeypatch):
        monkeypatch.setattr(survival, '_MAX_OUTER_PRODUCT_DOUBLES', 0)
        X, y, w, edges = self._setup()
        _level_gram_blocks(X, y, w, (X.shape[0],), edges, append_constant=False)
        assert survival._PAIRED_GRAMS == []

    def test_only_the_latest_are_kept(self):
        X, y, w, edges = self._setup()
        n = X.shape[0]
        for k in range(survival._PAIRED_GRAMS_KEPT + 1):
            _level_gram_blocks(X * (k + 1), y, w, (n,), edges, append_constant=False)
        assert len(survival._PAIRED_GRAMS) == survival._PAIRED_GRAMS_KEPT
        assert np.array_equal(survival._PAIRED_GRAMS[0][0][:, :-1], X * (survival._PAIRED_GRAMS_KEPT + 1))


class TestTheScores:

    @pytest.mark.parametrize('shape', [(1200,), (40, 30), (6, 5, 40)])
    @pytest.mark.parametrize('fit_intercept', [True, False])
    def test_dyad_t_is_the_augmented_build(self, shape, fit_intercept):
        X, y, w = _design(int(np.prod(shape)), 4, seed=3)
        assert _bytes_equal(dyad_t_scores(X, y, w, shape, fit_intercept=fit_intercept),
                            _augmented_dyad_t_scores(X, y, w, shape, fit_intercept=fit_intercept))

    @pytest.mark.parametrize('shape', [(1200,), (40, 30), (6, 5, 40)])
    @pytest.mark.parametrize('fit_intercept', [True, False])
    def test_dyad_is_the_augmented_build(self, monkeypatch, shape, fit_intercept):
        X, y, w = _design(int(np.prod(shape)), 4, seed=4)
        new = dyad_scores(X, y, w, shape, fit_intercept=fit_intercept)
        monkeypatch.setattr(survival, 'dyad_t_scores', _augmented_dyad_t_scores)
        assert _bytes_equal(new, dyad_scores(X, y, w, shape, fit_intercept=fit_intercept))

    def test_the_paired_scores_are_the_unpaired_ones(self, monkeypatch):
        n = survival._PAIRED_MIN_ROWS + 2000
        X, y, w = _design(n, 3, seed=5)
        S1 = _with_ones(X)
        served = []
        real = survival._level_gram_blocks

        def spy(*args, **kwargs):
            out = real(*args, **kwargs)
            served.append(any(out[0] is kept[4] for kept in survival._PAIRED_GRAMS))
            return out
        monkeypatch.setattr(survival, '_level_gram_blocks', spy)
        paired = [dyad_scores(X, y, w, (n,), fit_intercept=False),
                  dyad_scores(S1, y, w, (n,), fit_intercept=False)]
        assert served == [False, True]
        monkeypatch.setattr(survival, '_PAIRED_MIN_ROWS', np.inf)
        survival._PAIRED_GRAMS.clear()
        for M, score in zip((X, S1), paired):
            assert _bytes_equal(score, dyad_scores(M, y, w, (n,), fit_intercept=False))
            assert _bytes_equal(score, _augmented_dyad_t_scores(M, y, w, (n,), fit_intercept=False))
