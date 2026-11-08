#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``instability_metric='dyad'``: the dyadic slab statistic along every grid axis, averaged over the axes.

``dyad_t_scores`` is its one-axis component (the dyadic partitions of storage axis 0), not a metric of its
own: the time-only ``'dyad_t'`` was removed (USER Oct 2026, "remove dyad_t, leave full only").

Pinned here:

* one axis, or rows that are not the grid's flattening, is the axis-0 component exactly;
* on a grid it is the mean of the axis-0 component over the axes, each made storage axis 0;
* a coefficient that drifts along space only is invisible to time slabs and seen by dyad;
* a capped axis caps the term; the row floor covers every axis it cuts;
* the subset scores of the assignment table are the summed per-subset scores;
* the config accepts it, refuses 'dyad_t', and the objective and the keep-rule share the estimator.
"""
import numpy as np
import pytest

from epde.interface.search_config import load_search_config
from epde.operators.common import sparsity, survival
from epde.operators.common.survival import (_BASIS_FREE_METRICS, dyad_scores, dyad_subset_scores,
                                            dyad_t_scores, dyad_t_subset_scores, row_floor)


def _drifting(n_t=64, n_x=64, slope=0.8, seed=0):
    """y = c(x) X1 + X2 + noise on a (t, x) grid, c growing along x only."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_x)
    X1, X2 = rng.standard_normal((n_t, n_x)), rng.standard_normal((n_t, n_x))
    c = 1.0 + slope * x[None, :]
    y = c * X1 + X2 + 1e-3 * rng.standard_normal((n_t, n_x))
    return np.stack([X1.ravel(), X2.ravel()], axis=1), y.ravel(), np.ones(n_t * n_x), (n_t, n_x)


class TestTheEstimator:

    def test_one_axis_is_dyad_t(self):
        rng = np.random.default_rng(1)
        X, y = rng.standard_normal((200, 2)), rng.standard_normal(200)
        w = np.ones(200)
        np.testing.assert_array_equal(dyad_scores(X, y, w, (200,)), dyad_t_scores(X, y, w, (200,)))

    def test_rows_that_are_not_the_grid_are_dyad_t(self):
        X, y, w, shape = _drifting()
        rows = slice(0, X.shape[0] // 2)                       # half the rows: not prod(shape)
        np.testing.assert_array_equal(dyad_scores(X[rows], y[rows], w[rows], shape),
                                      dyad_t_scores(X[rows], y[rows], w[rows], shape))

    def test_a_grid_is_the_mean_over_its_axes(self):
        X, y, w, shape = _drifting()
        along_t = dyad_t_scores(X, y, w, shape)
        order = np.arange(X.shape[0]).reshape(shape).T.ravel()
        along_x = dyad_t_scores(X[order], y[order], w[order], shape[::-1])
        np.testing.assert_allclose(dyad_scores(X, y, w, shape), (along_t + along_x) / 2, rtol=1e-12)

    def test_a_drift_along_space_is_seen(self):
        X, y, w, shape = _drifting(slope=0.8)
        time_only, every_axis = dyad_t_scores(X, y, w, shape), dyad_scores(X, y, w, shape)
        assert time_only[0] < 0.05                             # each time slab spans all of x
        assert every_axis[0] > 5 * time_only[0]
        X, y, w, shape = _drifting(slope=0.0)                  # a constant coefficient stays quiet
        assert dyad_scores(X, y, w, shape)[0] < 0.05

    def test_a_capped_axis_caps_the_term(self, monkeypatch):
        X, y, w, shape = _drifting()
        real = survival.dyad_t_scores
        calls = []

        def capped_on_space(X_, y_, w_, shape_, fit_intercept=True):
            calls.append(shape_)
            out = np.asarray(real(X_, y_, w_, shape_, fit_intercept=fit_intercept), dtype=float)
            return np.full_like(out, np.finfo(float).max) if len(calls) == 2 else out
        monkeypatch.setattr(survival, 'dyad_t_scores', capped_on_space)
        assert np.all(dyad_scores(X, y, w, shape) == np.finfo(float).max)


class TestTheWiring:

    def test_the_row_floor_covers_every_axis(self):
        # 3 levels of x make x slabs of 40-80 rows, below 4 * (12 + 1) = 52; time slabs are fine.
        time_slabs = survival._dyadic_slab_edges(120, (40, 3), 52)
        assert min(int(np.diff(edges).min()) for edges in time_slabs) >= 52
        assert 'below 52' in row_floor('dyad', (40, 3), 12)

    def test_the_config_and_the_shared_table(self):
        assert load_search_config(overrides={'instability_metric': 'dyad'}).objectives.instability_metric == 'dyad'
        assert load_search_config().objectives.instability_metric == 'dyad'
        assert _BASIS_FREE_METRICS['dyad'] is dyad_scores
        assert sparsity._KEEP_RULE_ESTIMATORS is _BASIS_FREE_METRICS

    def test_the_time_only_metric_is_gone(self):
        assert 'dyad_t' not in _BASIS_FREE_METRICS
        with pytest.raises(ValueError):
            load_search_config(overrides={'instability_metric': 'dyad_t'})


class TestTheSubsetScores:
    """``dyad_subset_scores``: the assignment table's per-subset dyad, built once per axis."""

    SUBSETS = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2], []]

    def _design(self, shape):
        X, y, w, grid = _drifting()
        if shape != grid:
            n = int(np.prod(shape))
            X, y, w = X[:n], y[:n], w[:n]
        return np.hstack([X, np.ones((X.shape[0], 1))]), y, w

    def test_a_grid_is_the_summed_per_subset_dyad(self):
        X, y, w = self._design((64, 64))
        got = dyad_subset_scores(X, y, w, (64, 64), self.SUBSETS)
        want = [np.inf if not S else float(np.sum(np.nan_to_num(dyad_scores(X[:, S], y, w, (64, 64),
                                                                            fit_intercept=False))))
                for S in self.SUBSETS]
        np.testing.assert_allclose(got, want, rtol=1e-9)

    def test_one_axis_is_the_axis0_subset_scores(self):
        X, y, w = self._design((4096,))
        np.testing.assert_array_equal(dyad_subset_scores(X, y, w, (4096,), self.SUBSETS),
                                      dyad_t_subset_scores(X, y, w, (4096,), self.SUBSETS))

    def test_a_drift_along_space_is_seen_by_the_table(self):
        X, y, w = self._design((64, 64))
        # The drifting term scores 8x its time-only score; the steady one dilutes the sum to 4x.
        every_axis = dyad_subset_scores(X, y, w, (64, 64), [[0, 1]])
        time_only = dyad_t_subset_scores(X, y, w, (64, 64), [[0, 1]])
        assert every_axis[0] > 3 * time_only[0]
