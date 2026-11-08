#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``'assignment'`` mode's ``screen_size`` (``search_space.system_rps_params``):
under the selected-terms cap, the assignment mode scores only the best subsets of each size by weighted least
squares (``epde.operators.common.subset_search``, ported from the
exact-enumeration prototype's ``cxfast``). Pinned here:

* the search's top-B lists equal a brute-force solve of the same ridged
  problem, unit and mixed column costs alike, and it visits every subset;
* the screened masks: every subset when the screen is wide enough, the
  constant on the top bit, zero-norm columns left out, the same subsets
  whatever the column order once the tie order is fixed;
* the settings and their refusals;
* a screen as wide as every size IS the unscreened capped search, bit for bit,
  and a narrow one still keeps every rule of the assignment mode.

The exact engine (``subset_search.ExactScreen``, ``_ENGINE = 'exact'``):

* the kernel: the sweep's leaves ARE ``canonical_values``, bit for bit, at
  every size and whatever the row chunks and leaf blocks, and a value is the
  same in any batch and in any superset's problem;
* the order: each size's list is the first D supports of the brute-force
  ``(v, support tuple)`` order, every support counted -- on Gaussian and
  NS-like columns, exact twins, bitwise duplicates, a zero column dropped at
  assembly, and the all-tied problems (G = I, b ~ 1e-200), which take the
  lexicographic first D with memory bounded; the values are the ridged RSS;
* against the legacy screen: the same supports wherever the cut is clear;
* derivation: a removed set's lists derived from a superset's deep lists are
  the direct search's, or refused when they cannot be certified;
* the call site: the union columns are ``Equation.evaluate()``'s, every key's
  masks are its own direct search, a target that is not its own column goes
  to the legacy screen, and without the screen the engine is never built.

The dyad memo (``right_part_selection._DYAD_MEMO``, on for coupled systems): every
key's scores are its per-key ``dyad_subset_scores`` to round-off, a subset
has one score whichever key asks, repeated subsets are reused, and the uv
search keeps its choices.
"""

import itertools
from math import comb

import numpy as np
import pytest

import epde.operators.common.right_part_selection as rps
import epde.operators.common.subset_search as subset_search
from epde.interface.search_config import (assignment_subset_count, check_rps_params,
                                          load_search_config, resolve_rps_params)
from epde.operators.common.subset_search import (DEP_TOL, RIDGE, BestSubsets, ExactScreen,
                                                 assemble_from_union, canonical_values, derive,
                                                 lexicographic_levels, screened_masks,
                                                 sweep_level, unit_gram)
from epde.operators.multiobjective.moeadd_specific import (_debug_assert_rps_unique,
                                                           has_degenerate_equation)

from conftest import using_config
from test_assignment_rps import _fresh, _soeq_selector
from test_duplicate_equation_guard import uv_pool  # noqa: F401

CAP = 'selected_terms_max_number'
SCREEN = 'screen_size'


def _params(cap, screen=None):
    """The selected-terms cap and the screen as config overrides (``'assignment'``
    mode's settings)."""
    return {'system_rps_params': {CAP: cap, SCREEN: screen}}


def _problem(n=120, p=7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, 3] = X[:, 1] + 0.05 * rng.standard_normal(n)       # a near-dependent pair
    y = X @ (rng.standard_normal(p) * (rng.random(p) < 0.5)) + 0.1 * rng.standard_normal(n)
    sw = rng.uniform(0.5, 1.0, n)
    return X, y, sw


def _brute_force(G, b, yy, costs, c):
    """Every support of level ``c`` with its ridged RSS, sorted by (rss, support)."""
    out = []
    p = len(costs)
    for k in range(1, c + 1):
        for S in itertools.combinations(range(p), k):
            if sum(costs[j] for j in S) != c:
                continue
            GS = G[np.ix_(S, S)] + RIDGE * np.eye(k)
            out.append((float(yy - b[list(S)] @ np.linalg.solve(GS, b[list(S)])), S))
    return sorted(out)


# ---------------------------------------------------------------------------
# The search.
# ---------------------------------------------------------------------------

class TestTheSearch:

    @pytest.mark.parametrize('costs_kind', ['unit', 'mixed'])
    def test_top_b_is_the_brute_force_top_b(self, costs_kind):
        X, y, sw = _problem()
        G, b, yy, kept = unit_gram(X, y, sw)
        costs = (np.ones(len(kept), dtype=int) if costs_kind == 'unit'
                 else np.array([1, 2, 1, 2, 1, 1, 2, 1])[:len(kept)])
        search = BestSubsets(G, b, yy, costs, B=6)
        for c in range(1, 5):
            top, count = search.level(c)
            brute = _brute_force(G, b, yy, costs, c)
            assert count == len(brute)
            assert [s for _r, s in top] == [s for _r, s in brute[:6]]
            np.testing.assert_allclose([r for r, _s in top], [r for r, _s in brute[:6]],
                                       rtol=1e-9, atol=1e-12 * yy)

    def test_every_subset_is_visited(self):
        X, y, sw = _problem(p=9)
        G, b, yy, kept = unit_gram(X, y, sw)
        search = BestSubsets(G, b, yy, np.ones(len(kept), dtype=int), B=3)
        for c in range(1, 6):
            assert search.level(c)[1] == comb(len(kept), c)

    def test_costs_other_than_one_or_two_are_refused(self):
        X, y, sw = _problem()
        G, b, yy, kept = unit_gram(X, y, sw)
        with pytest.raises(ValueError, match='costs'):
            BestSubsets(G, b, yy, np.full(len(kept), 3), B=2)


# ---------------------------------------------------------------------------
# The screened masks.
# ---------------------------------------------------------------------------

class TestTheMasks:

    def test_a_wide_screen_is_every_subset_within_the_cap(self):
        X, y, sw = _problem(p=6)
        masks = screened_masks(X, y, sw, selected_max=3, screen_size=comb(7, 3))
        assert sorted(masks) == rps._capped_masks(7, 3)

    def test_a_narrow_screen_keeps_at_most_b_of_each_size(self):
        X, y, sw = _problem(p=6)
        masks = screened_masks(X, y, sw, selected_max=3, screen_size=2)
        sizes = [bin(S).count('1') for S in masks]
        assert all(sizes.count(k) <= 2 for k in (1, 2, 3)) and max(sizes) <= 3

    def test_the_constant_is_the_top_bit_and_zero_columns_are_left_out(self):
        X, y, sw = _problem(p=5)
        X[:, 2] = 0.0
        masks = screened_masks(X, y, sw, selected_max=2, screen_size=100)
        assert not any(S >> 2 & 1 for S in masks)
        assert (1 << 5) in masks                      # the constant alone
        assert len(masks) == 5 + comb(5, 2)           # 4 features + constant, sizes 1 and 2

    def test_a_target_without_signal_takes_the_first_supports(self):
        """Every support then has the same RSS: the (rss, support) order is the
        lexicographic one, returned without running the search."""
        X, _y, sw = _problem(p=6)
        masks = screened_masks(X, np.zeros(X.shape[0]), sw, selected_max=3, screen_size=4)
        assert sorted(bin(S).count('1') for S in masks) == [1] * 4 + [2] * 4 + [3] * 4
        assert 0b11 in masks and 0b111 in masks

    def test_bad_weights_are_refused(self):
        X, y, sw = _problem(p=4)
        sw[3] = np.nan
        with pytest.raises(ValueError, match='weights'):
            screened_masks(X, y, sw, selected_max=2, screen_size=3)

    def test_a_numpy_order_gives_python_bitmasks(self):
        X, y, sw = _problem(p=5)
        masks = screened_masks(X, y, sw, selected_max=2, screen_size=3, order=np.arange(5))
        assert all(type(S) is int for S in masks)

    def test_the_same_subsets_whatever_the_column_order(self):
        X, y, sw = _problem(p=7, seed=3)
        base = screened_masks(X, y, sw, selected_max=3, screen_size=4)
        perm = [4, 0, 6, 2, 1, 5, 3]                  # column i of Xp is column perm[i] of X
        Xp = X[:, perm]
        # Same tie order: Xp's columns ranked as X's natural order.
        order = sorted(range(7), key=lambda i: perm[i])
        permuted = screened_masks(Xp, y, sw, selected_max=3, screen_size=4, order=order)

        def back(S):
            cols = [perm[i] if i < 7 else 7 for i in range(8) if S >> i & 1]
            return sum(1 << c for c in cols)
        assert {back(S) for S in permuted} == base


# ---------------------------------------------------------------------------
# Settings.
# ---------------------------------------------------------------------------

class TestTheSettings:

    @staticmethod
    def _space(terms=None, **params):
        overrides = {'system_rps_mode': 'assignment', 'system_rps_params': params}
        if terms is not None:
            overrides['equation_terms_max_number'] = terms
        return load_search_config(overrides=overrides).search_space

    def test_the_default_is_no_screen(self):
        assert self._space().system_rps_params[SCREEN] is None

    @pytest.mark.parametrize('value', [0, -1, 2.5, True])
    def test_anything_but_a_positive_integer_is_refused(self, value):
        with pytest.raises(ValueError, match=SCREEN):
            self._space(**{CAP: 3, SCREEN: value})

    def test_it_needs_the_cap(self):
        with pytest.raises(ValueError, match=CAP):
            self._space(**{SCREEN: 10})

    @pytest.mark.parametrize('mode', ['repair', 'plain'])
    def test_only_the_assignment_mode_has_it(self, mode):
        with pytest.raises(ValueError, match=f"{SCREEN} is a setting of system_rps_mode 'assignment'"):
            load_search_config(overrides={'system_rps_mode': mode,
                                          'system_rps_params': {SCREEN: 10}})

    def test_the_screen_lifts_the_ceiling(self):
        with pytest.raises(ValueError, match=f"set '{SCREEN}'"):
            self._space(40, **{CAP: 6})
        space = self._space(40, **{CAP: 6, SCREEN: 100})
        assert (space.equation_terms_max_number, space.system_rps_params[SCREEN]) == (40, 100)
        assert assignment_subset_count(40, 6, 100) == 40 + 5 * 100

    def test_the_ceiling_still_counts_what_is_scored(self):
        with pytest.raises(ValueError, match=f"lower '{SCREEN}'"):
            check_rps_params('assignment', 40, resolve_rps_params('assignment', {CAP: 6, SCREEN: 1000}))

    def test_fit_shares_the_refusals(self):
        """``fit``'s own ``equation_terms_max_number`` meets the mode's screen."""
        import epde
        search = epde.EpdeSearch(use_solver=False, multiobjective_mode=True,
                                 system_rps_mode='assignment',
                                 system_rps_params={CAP: 6, SCREEN: 100},
                                 verbose_params={'show_iter_idx': False})
        with pytest.raises(ValueError, match='visits'):
            search.fit(data=None, equation_terms_max_number=200)

    def test_the_visits_are_bounded(self):
        # 200 columns, at most 6 kept: 8.5e10 subsets to visit.
        with pytest.raises(ValueError, match='visits'):
            check_rps_params('assignment', 200, resolve_rps_params('assignment', {CAP: 6, SCREEN: 100}))
        check_rps_params('assignment', 60, resolve_rps_params('assignment', {CAP: 6, SCREEN: 100}))


# ---------------------------------------------------------------------------
# The screened search.
# ---------------------------------------------------------------------------

def _run(system, seed=0, **overrides):
    np.random.seed(seed)
    with using_config(system_rps_mode='assignment', **overrides):
        _soeq_selector().apply(system, {})
    return system


def _labels_and_weights(system):
    return (system.equations_labels,
            [np.asarray(eq.weights_final).tolist() for eq in system.vals])


class TestTheScreenedSearch:

    @pytest.mark.parametrize('metric', ['chi2', 'dyad'])
    def test_a_screen_as_wide_as_every_size_is_the_capped_search(self, uv_pool, metric):
        capped = _run(_fresh(uv_pool), instability_metric=metric, **_params(2))
        screened = _run(_fresh(uv_pool), instability_metric=metric, **_params(2, 1000))
        assert _labels_and_weights(screened) == _labels_and_weights(capped)

    def test_a_narrow_screen_keeps_the_rules(self, uv_pool):
        system = _run(_fresh(uv_pool), **_params(2, 1))
        assert not any(_debug_assert_rps_unique(system))
        assert not has_degenerate_equation(system)
        for var in ('u', 'v'):
            eq = system.vals[var]
            wi = np.asarray(eq.weights_internal, dtype=float)
            kept = sum(wi[eq.weight_index(i, eq.target_idx)] != 0
                       for i in range(len(eq.structure)) if i != eq.target_idx)
            assert 1 <= kept + int(wi[-1] != 0) <= 2

    def test_the_screen_is_used(self, uv_pool, monkeypatch):
        """The exact engine: every key is screened by the call's
        ``_ScreenCallState`` with the settings' cap and screen size."""
        calls, real = [], rps._ScreenCallState.masks

        def recording(self, *args, **kwargs):
            calls.append((self.M, self.B))
            return real(self, *args, **kwargs)
        monkeypatch.setattr(rps._ScreenCallState, 'masks', recording)
        _run(_fresh(uv_pool), **_params(2, 3))
        assert calls and all(c == (2, 3) for c in calls)

    def test_the_legacy_screen_is_used_under_the_legacy_engine(self, uv_pool, monkeypatch):
        """``_equation_variants`` imports the screen at call time, so the
        module attribute is the one it runs."""
        import epde.operators.common.subset_search as search
        calls, real = [], search.screened_masks

        def recording(*args, **kwargs):
            calls.append(args[3:5])                    # (selected_max, screen_size)
            return real(*args, **kwargs)
        monkeypatch.setattr(search, 'screened_masks', recording)
        monkeypatch.setattr(search, '_ENGINE', 'legacy')
        _run(_fresh(uv_pool), **_params(2, 3))
        assert calls and all(c == (2, 3) for c in calls)

    def test_the_ceiling_is_enforced_at_run_time(self, uv_pool, monkeypatch):
        monkeypatch.setattr(rps, 'MAX_ASSIGNMENT_SUBSETS', 1)
        with pytest.raises(ValueError, match='screens'):
            _run(_fresh(uv_pool), **_params(2, 3))

    def test_without_the_screen_the_exact_engine_is_never_built(self, uv_pool, monkeypatch):
        """The default path (no cap, or a cap without a screen) never screens:
        no key reaches the call state's screen, the exact engine or the legacy
        screen. The call state itself may be built -- the dyad memo
        (``_DYAD_MEMO``) keeps its union slab Grams there -- but only when the
        memo is on, and then without a screen size: the memo is its one
        reason to exist without a screen."""
        import epde.operators.common.subset_search as search

        def refuse(*args, **kwargs):
            raise AssertionError('the screen was reached without equation_screen_size')
        states, init = [], rps._ScreenCallState.__init__

        def recording(self, *args, **kwargs):
            init(self, *args, **kwargs)
            states.append(self)
        monkeypatch.setattr(rps._ScreenCallState, '__init__', recording)
        monkeypatch.setattr(rps._ScreenCallState, 'masks', refuse)
        monkeypatch.setattr(search.ExactScreen, '__init__', refuse)
        monkeypatch.setattr(search, 'screened_masks', refuse)
        for overrides in ({}, _params(2)):
            system, before = _fresh(uv_pool), len(states)
            with using_config(system_rps_mode='assignment', **overrides):
                memo = rps._use_dyad_memo(len(system.vals.equation_keys))
            _run(system, **overrides)
            assert not any(_debug_assert_rps_unique(system))
            built = states[before:]
            assert bool(built) is memo
            assert all(state.dyad_memo and state.B is None for state in built)


# ---------------------------------------------------------------------------
# The exact screen: problems.
# ---------------------------------------------------------------------------

def _union(X, y, sw):
    """The union Gram of ``[X | 1 | y]`` weighted by ``sqrt(sw)``, as the call
    site builds it: the constant at position ``X.shape[1]``, the target last."""
    Z = np.column_stack([X, np.ones(X.shape[0]), y]) * np.sqrt(sw)[:, None]
    return Z.T @ Z


def _assembled(X, y, sw, cols=None):
    """The screen problem of ``[X | 1]`` against ``y`` from the union Gram:
    ``(A, b, yy, kept)``, every column by default."""
    raw = _union(X, y, sw)
    t = X.shape[1] + 1
    return assemble_from_union(raw, raw[:, t], raw[t, t], list(range(t)) if cols is None else cols)


# The data families; ``p`` counts the screen's columns, the constant included.

def _gaussian(p, seed=0, n=150):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p - 1))
    beta = rng.standard_normal(p - 1) * (rng.random(p - 1) < 0.4)
    y = X @ beta + 0.3 + 0.1 * rng.standard_normal(n)
    return X, y, rng.uniform(0.5, 1.0, n)


def _ns_like(p, seed=0, n=400):
    """NS-like columns (p >= 8): 3 heavy and 2 weak, the second weak one at
    cosine ~0.985 to the constant, a twin anti-collinear with a heavy one at
    squared sine ~1e-5, a 0.98 near-duplicate of another, Gaussian filler; the
    target on the heavy and weak ones at 1e-3 noise."""
    rng = np.random.default_rng(seed)
    heavy = rng.standard_normal((n, 3))
    weak = rng.standard_normal(n)
    flat = 1.0 + 0.175 * rng.standard_normal(n)
    twin = -heavy[:, 1] + np.sqrt(1e-5) * rng.standard_normal(n)
    near = 0.98 * heavy[:, 2] + np.sqrt(1.0 - 0.98 ** 2) * rng.standard_normal(n)
    X = np.column_stack([heavy, weak, flat, twin, near, rng.standard_normal((n, p - 8))])
    y = heavy @ [1.0, 0.8, -0.6] + 0.03 * weak + 0.02 * flat + 1e-3 * rng.standard_normal(n)
    return X, y, rng.uniform(0.5, 1.0, n)


def _exact_twin(p, seed=0):
    """A scaled copy (one rounding off the same direction) and an exact identity."""
    X, y, sw = _gaussian(p, seed)
    X[:, 2] = -2.5 * X[:, 1]
    X[:, 3] = X[:, 0] + X[:, 4]
    return X, y, sw


def _duplicate(p, seed=0):
    X, y, sw = _gaussian(p, seed)
    X[:, 4] = X[:, 1]
    return X, y, sw


def _zero_column(p, seed=0):
    X, y, sw = _gaussian(p, seed)
    X[:, 2] = 0.0
    return X, y, sw


def _hadamard(p, signal=None):
    """Columns 1 .. p-1 of a 16-point Hadamard matrix; its column 0 (all ones)
    is the constant. Every product is an exact integer, so G = I bit for bit.
    The target is the sum of the screen positions ``signal`` (default: all p,
    so every b_j is equal)."""
    H = np.ones((1, 1))
    for _ in range(4):
        H = np.block([[H, H], [H, -H]])
    screen = np.column_stack([H[:, 1:p], H[:, :1]])           # the screen's column order
    y = screen.sum(axis=1) if signal is None else screen[:, list(signal)].sum(axis=1)
    return H[:, 1:p], y, np.ones(16)


_DATA = {'gaussian': _gaussian, 'ns_like': _ns_like, 'exact_twin': _exact_twin,
         'duplicate': _duplicate, 'zero_column': _zero_column, 'hadamard': _hadamard}
FAMILIES = list(_DATA) + ['tiny_b']


def _screen_problem(family, p):
    """``(A, b, yy)`` of a family; ``'tiny_b'`` is the Gaussian problem with
    its cross moments scaled to ~1e-200 (every reduction underflows)."""
    if family == 'tiny_b':
        A, b, yy, _kept = _assembled(*_gaussian(p))
        return A, b * 1e-200, yy
    A, b, yy, _kept = _assembled(*_DATA[family](p))
    return A, b, yy


def _all_supports(p, k):
    return np.array(list(itertools.combinations(range(p), k)), dtype=np.intp).reshape(-1, k)


def _brute(A, b, yy, k):
    """Every size-``k`` support's canonical value, sorted by (v, support tuple)
    in Python -- the reference order, independent of the engine's lexsort."""
    S = _all_supports(len(b), k)
    v = canonical_values(A, b, yy, S)
    order = sorted(range(len(S)), key=lambda n: (float(v[n]), tuple(S[n].tolist())))
    return v[order], S[order]


def _bits(v):
    return np.ascontiguousarray(v, dtype=float).view(np.int64)


# ---------------------------------------------------------------------------
# T1: the kernel.
# ---------------------------------------------------------------------------

def _kernel_problem(kind):
    """A 10-column problem. ``'dependent'``: column 6 is an exact copy of
    column 2 IN ``A`` (no ridge between them: the second's pivot is
    round-off, at or below DEP_TOL) and column 8 has a zero row (a zero
    pivot) -- the no-credit paths of every step."""
    A, b, yy, _kept = _assembled(*_gaussian(10, seed=7))
    if kind == 'dependent':
        A = A.copy()
        A[6, :], A[:, 6] = A[2, :], A[:, 2]
        A[6, 6] = A[2, 2]
        A[8, :], A[:, 8] = 0.0, 0.0
        b = b.copy()
        b[6] = b[2]
    return A, b, yy


class TestTheKernel:

    @pytest.mark.parametrize('kind', ['gaussian', 'dependent'])
    @pytest.mark.parametrize('k', range(1, 7))
    def test_the_sweep_leaves_are_the_canonical_values(self, monkeypatch, kind, k):
        """With D at least comb(p, k) every leaf is returned: the full
        enumeration, values and supports, bit for bit."""
        A, b, yy = _kernel_problem(kind)
        p = len(b)
        if kind == 'dependent':
            assert A[6, 6] - A[2, 6] * (A[2, 6] / A[2, 2]) <= DEP_TOL and A[8, 8] <= DEP_TOL
        rv, rS = _brute(A, b, yy, k)
        for leaf_block in (subset_search._LEAF_BLOCK, 1, 7):
            monkeypatch.setattr(subset_search, '_LEAF_BLOCK', leaf_block)
            for chunk in (1, 300, None):
                v, S, count = sweep_level(A, b, yy, k, comb(p, k), chunk=chunk)
                assert count == comb(p, k)
                assert np.array_equal(S, rS), (leaf_block, chunk)
                assert np.array_equal(_bits(v), _bits(rv)), (leaf_block, chunk)

    def test_a_value_is_the_same_in_any_batch_or_superset(self):
        """v(S) is a pure function of (A[S, S], b[S], yy): the same bits for a
        support alone, in a random batch, and in a superset's problem."""
        X, y, sw = _gaussian(14, seed=4)
        A, b, yy, kept = _assembled(X, y, sw)
        F = np.array([0, 2, 3, 5, 8, 9, 12, 13])              # 13 is the constant
        AF, bF, yyF, _kept = _assembled(X, y, sw, cols=kept[F])
        rng = np.random.default_rng(0)
        for k in range(1, 7):
            S = _all_supports(len(F), k)
            batch = canonical_values(AF, bF, yyF, S)
            assert np.array_equal(_bits(canonical_values(A, b, yy, F[S])), _bits(batch))
            pick = rng.choice(len(S), size=min(len(S), 17), replace=False)
            assert np.array_equal(_bits(canonical_values(AF, bF, yyF, S[pick])), _bits(batch[pick]))
            for n in pick[:4]:
                assert _bits(canonical_values(AF, bF, yyF, S[n:n + 1]))[0] == _bits(batch)[n]


# ---------------------------------------------------------------------------
# T2: the exact screen is the brute-force order.
# ---------------------------------------------------------------------------

class TestTheExactScreen:

    @pytest.mark.parametrize('family', FAMILIES)
    @pytest.mark.parametrize('p', [8, 11, 14])
    def test_the_lists_are_the_brute_force_order(self, family, p):
        A, b, yy = _screen_problem(family, p)
        n = len(b)
        brute = {k: _brute(A, b, yy, k) for k in range(1, min(6, n) + 1)}
        for B in (1, 3, 10):
            for D in (B, 2 * B):
                screen = ExactScreen(A, b, yy, 6, D)
                levels = screen.levels()
                assert sorted(levels) == sorted(brute)
                for k, (v, S, whole) in levels.items():
                    rv, rS = brute[k]
                    assert np.array_equal(v, rv[:D]) and np.array_equal(S, rS[:D]), (B, D, k)
                    assert whole == (comb(n, k) <= D)
                    assert screen.counts[k] == comb(n, k)

    def test_small_chunks_and_leaf_blocks_keep_the_order(self, monkeypatch):
        """Ties and near-ties carried across many chunks and flushes."""
        monkeypatch.setattr(subset_search, '_SWEEP_CHUNK', 400)
        monkeypatch.setattr(subset_search, '_LEAF_BLOCK', 3)
        for family in ('ns_like', 'duplicate', 'tiny_b'):
            A, b, yy = _screen_problem(family, 11)
            for D in (1, 6):
                for k, (v, S, _whole) in ExactScreen(A, b, yy, 6, D).levels().items():
                    rv, rS = _brute(A, b, yy, k)
                    assert np.array_equal(v, rv[:D]) and np.array_equal(S, rS[:D]), (family, D, k)

    def test_a_zero_column_is_dropped_at_assembly(self):
        X, y, sw = _zero_column(11)
        raw = _union(X, y, sw)
        A, b, yy, kept = assemble_from_union(raw, raw[:, 11], raw[11, 11], list(range(11)))
        assert kept.tolist() == [c for c in range(11) if c != 2]
        # The problem of the columns without it, bit for bit.
        A2, b2, yy2, kept2 = assemble_from_union(raw, raw[:, 11], raw[11, 11],
                                                 [c for c in range(11) if c != 2])
        assert np.array_equal(A, A2) and np.array_equal(b, b2) and yy == yy2
        assert np.array_equal(kept, kept2)

    @pytest.mark.parametrize('family', ['hadamard', 'tiny_b'])
    def test_tied_values_take_the_lexicographic_first(self, family):
        A, b, yy = _screen_problem(family, 14)
        if family == 'hadamard':
            assert np.array_equal(A, np.eye(14) + RIDGE * np.eye(14)) and np.all(b == b[0])
        for D in (1, 3, 10):
            for k, (v, S, _whole) in ExactScreen(A, b, yy, 6, D).levels().items():
                assert np.all(v == v[0])
                assert np.array_equal(S, _all_supports(14, k)[:D])

    def test_ties_keep_the_level_lists_at_a_few_d(self, monkeypatch):
        """Under total ties every leaf is admitted, yet a list holds at most D
        kept rows plus under 4D buffered plus one pushed leaf block (one row
        of leaves at ``_LEAF_BLOCK = 1``)."""
        monkeypatch.setattr(subset_search, '_LEAF_BLOCK', 1)
        peak, kept = {}, {}
        real_push, real_flush = subset_search._LevelTop.push, subset_search._LevelTop.flush

        def push(self, v, s):
            peak[self.k] = max(peak.get(self.k, 0), len(self.v) + self.nbuf + len(v))
            return real_push(self, v, s)

        def flush(self):
            real_flush(self)
            kept[self.k] = max(kept.get(self.k, 0), len(self.v))
        monkeypatch.setattr(subset_search._LevelTop, 'push', push)
        monkeypatch.setattr(subset_search._LevelTop, 'flush', flush)
        A, b, yy = _screen_problem('tiny_b', 12)
        for D in (1, 4, 12):
            peak.clear()
            kept.clear()
            levels = ExactScreen(A, b, yy, 6, D).levels()
            for k, (v, S, _whole) in levels.items():
                assert np.all(v == yy) and np.array_equal(S, _all_supports(12, k)[:D])
                assert kept[k] <= D
                if k >= 3:
                    assert peak[k] <= 5 * D + 12, (D, k, peak[k])

    @pytest.mark.parametrize('family', ['gaussian', 'ns_like'])
    def test_the_values_are_the_ridged_rss(self, family):
        """v(S) = yy - b_S' solve(A_SS, b_S) to round-off on well-conditioned supports."""
        A, b, yy = _screen_problem(family, 11)
        checked = 0
        for k in range(1, 7):
            S = _all_supports(len(b), k)
            ASS = A[S[:, :, None], S[:, None, :]]
            well = np.linalg.cond(ASS) < 1e4
            bS = b[S][well]
            ref = yy - np.einsum('ni,ni->n', bS, np.linalg.solve(ASS[well], bS[..., None])[..., 0])
            np.testing.assert_allclose(canonical_values(A, b, yy, S[well]), ref,
                                       rtol=1e-9, atol=1e-12 * yy)
            checked += int(well.sum())
        assert checked > 500


# ---------------------------------------------------------------------------
# T4: against the legacy screen.
# ---------------------------------------------------------------------------

class TestAgainstTheLegacyScreen:

    @pytest.mark.parametrize('family', ['gaussian', 'ns_like'])
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_the_same_supports_wherever_the_cut_is_clear(self, family, seed):
        """BestSubsets on ``unit_gram`` and ExactScreen on the union Gram
        assembly of the same data keep the same B supports of every size
        whenever the legacy B-th and (B+1)-th RSS are more than 1e-12 yy apart."""
        X, y, sw = _DATA[family](12, seed)
        G, b, yy, kept = unit_gram(X, y, sw)
        A, bx, yyx, keptx = _assembled(X, y, sw)
        assert np.array_equal(kept, keptx)
        compared = 0
        for B in (1, 3, 10):
            legacy = BestSubsets(G, b, yy, np.ones(len(kept), dtype=int), B + 1)
            for k, (_v, S, _whole) in ExactScreen(A, bx, yyx, 6, B).levels().items():
                top, _count = legacy.level(k)
                if len(top) > B and top[B][0] - top[B - 1][0] <= 1e-12 * yy:
                    continue
                compared += 1
                assert {s for _r, s in top[:B]} == {tuple(row) for row in S.tolist()}, (B, k)
        assert compared >= 15


# ---------------------------------------------------------------------------
# T6: derivation.
# ---------------------------------------------------------------------------

def _positions(n, removed):
    """``derive``'s map: position in F' -> position in F = F' minus ``removed``, -1 if removed."""
    positions = np.full(n, -1, dtype=np.intp)
    F = [c for c in range(n) if c not in set(removed)]
    positions[F] = np.arange(len(F))
    return positions, F


class TestDerivation:

    @pytest.mark.parametrize('family', ['gaussian', 'ns_like'])
    def test_a_removed_set_derives_the_direct_search(self, family):
        """For every R of at most two columns: F \\ R's problem is bitwise a
        sub-block of F's, and its lists derived from F's deep lists (D = 2B)
        are the direct search's, bit for bit -- or refused only where some
        list cannot be certified."""
        X, y, sw = _DATA[family](10)
        raw = _union(X, y, sw)
        t = X.shape[1] + 1
        A, b, yy, kept = assemble_from_union(raw, raw[:, t], raw[t, t], list(range(t)))
        M, B = 6, 3
        deep = ExactScreen(A, b, yy, M, 2 * B).levels()
        derived = refused = 0
        for r in (0, 1, 2):
            for R in itertools.combinations(range(len(kept)), r):
                positions, F = _positions(len(kept), R)
                AF, bF, yyF, _keptF = assemble_from_union(raw, raw[:, t], raw[t, t], kept[F])
                assert np.array_equal(AF, A[np.ix_(F, F)]) and np.array_equal(bF, b[F])
                lists = derive(deep, positions, M, B)
                if lists is None:
                    refused += 1
                    assert any(np.count_nonzero(np.all(positions[deep[k][1]] >= 0, axis=1)) < B
                               and not deep[k][2] for k in range(1, min(M, len(F)) + 1))
                    continue
                derived += 1
                direct = ExactScreen(AF, bF, yyF, M, B).levels()
                assert sorted(lists) == sorted(direct)
                for k, S in lists.items():
                    assert np.array_equal(S, direct[k][1]), (R, k)
                    assert np.array_equal(_bits(canonical_values(AF, bF, yyF, S)),
                                          _bits(direct[k][0]))
        assert derived >= 10 and derived + refused == 1 + 10 + 45

    def test_an_r_that_empties_a_list_falls_back(self):
        """B = 1, D = 2: removing both singles of the deep size-1 list leaves
        no certified single."""
        A, b, yy = _screen_problem('gaussian', 10)
        deep = ExactScreen(A, b, yy, 4, 2).levels()
        R = deep[1][1][:, 0].tolist()
        assert len(R) == 2 and not deep[1][2]
        assert derive(deep, _positions(10, R)[0], 4, 1) is None
        # One of them removed: the singles are still certified.
        assert derive(deep, _positions(10, R[:1])[0], 1, 1) is not None

    @pytest.mark.parametrize('D', [6, 60, 70])
    def test_an_r_that_removes_all_signal_takes_the_lexicographic_lists(self, D):
        """The target only along the removed columns (Hadamard columns, so
        every other cross moment is exactly zero): F \\ R's supports all have
        v = yy, the direct search takes the lexicographic lists, and so does
        the derivation whenever it certifies them: from D = 58 on (the 55
        size-4 supports touching R rank first), not at D = 2B = 6."""
        X, y, sw = _hadamard(8, signal=(0, 1))
        raw = _union(X, y, sw)
        A, b, yy, kept = assemble_from_union(raw, raw[:, 8], raw[8, 8], list(range(8)))
        assert np.all(b[2:] == 0) and np.all(b[:2] != 0)
        M, B = 4, 3
        positions, F = _positions(8, (0, 1))
        AF, bF, yyF, _keptF = assemble_from_union(raw, raw[:, 8], raw[8, 8], kept[F])
        direct = ExactScreen(AF, bF, yyF, M, B).levels()
        lexicographic = lexicographic_levels(6, M, B)
        for k in direct:
            assert np.array_equal(direct[k][1], lexicographic[k])
        lists = derive(ExactScreen(A, b, yy, M, D).levels(), positions, M, B)
        assert (lists is None) == (D < 58)
        if lists is not None:
            assert sorted(lists) == sorted(direct)
            for k in direct:
                assert np.array_equal(lists[k], lexicographic[k])


# ---------------------------------------------------------------------------
# T7: the union Gram at the call site.
# ---------------------------------------------------------------------------

def _recording_masks(monkeypatch):
    """Record every ``_ScreenCallState.masks`` call: (state, i, t, features,
    the key's ``(key, X, y, sw)`` rows, returned masks)."""
    records, real = [], rps._ScreenCallState.masks

    def recording(self, i, t, features, finite):
        out = real(self, i, t, features, finite)
        records.append((self, i, t, list(features), finite, out))
        return out
    monkeypatch.setattr(rps._ScreenCallState, 'masks', recording)
    return records


class TestTheCallSite:

    def test_the_union_columns_are_the_evaluated_ones(self, uv_pool, monkeypatch):
        """Per trajectory, the union's ``[every term | 1]`` columns are
        ``Equation.evaluate()``'s target and features bit for bit, and its
        Gram is theirs, weighted (the union's matrix is caught at its
        ``np.column_stack``)."""
        import epde.globals as global_var
        system = _fresh(uv_pool)
        stacks, real = [], np.column_stack

        def recording(arrays, *args, **kwargs):
            out = real(arrays, *args, **kwargs)
            stacks.append(out)
            return out
        state = rps._ScreenCallState(list(system), 2, 3)
        gfuncs = global_var.samples_manager.gFunc('dm')
        for i, eq in enumerate(system):
            stacks.clear()
            with monkeypatch.context() as patch:
                patch.setattr(np, 'column_stack', recording)
                union = state._union(i)
            n_terms = len(eq.structure)
            matrices = [X for X in stacks if X.ndim == 2 and X.shape[1] == n_terms + 1]
            assert len(matrices) == len(union['keys']) >= 1
            target, features = eq.evaluate()
            others = [j for j in range(n_terms) if j != eq.target_idx]
            for key, X in zip(union['keys'], matrices):
                assert np.array_equal(X[:, eq.target_idx], np.asarray(target[key]).reshape(-1))
                for c, j in enumerate(others):
                    assert np.array_equal(X[:, j], features[key][:, c])
                assert np.all(X[:, -1] == 1.0)
                gram, finite = union['per_key'][key]
                assert finite.all()
                Xw = X * np.sqrt(np.asarray(gfuncs[key], dtype=float).reshape(-1))[:, None]
                assert np.array_equal(gram, Xw.T @ Xw)

    @pytest.mark.parametrize('screen', [1, 2, 3])
    def test_every_key_is_its_own_direct_search(self, uv_pool, monkeypatch, screen):
        """Derivation is transparent: every key's masks are those of an
        ExactScreen run directly on the key's own columns assembled from the
        call's union, whether they were derived or searched."""
        records = _recording_masks(monkeypatch)
        _run(_fresh(uv_pool), **_params(2, screen))
        assert records
        for state, i, t, features, finite, out in records:
            union = state._union(i)
            traj = tuple(key for key in union['keys'] if key in {k for k, *_ in finite})
            raw = state._raw(i, traj)
            cols, bits = state._canonical(union, features)
            A, b, yy, kept = assemble_from_union(raw, raw[:, t], raw[t, t], cols)
            levels = ExactScreen(A, b, yy, state.M, state.B).levels()
            assert out == {sum(1 << bits[int(kept[c])] for c in support)
                           for _v, S, _whole in levels.values() for support in S.tolist()}
        states = list({id(record[0]): record[0] for record in records}.values())
        assert all(s.stats['legacy'] == 0 for s in states)
        assert sum(s.stats['derived'] for s in states) >= 1

    def test_the_exact_masks_are_the_legacy_screens_here(self, uv_pool, monkeypatch):
        """No RSS ties at the cut on the uv system: the exact masks are
        ``screened_masks``'s on each key's own data."""
        records = _recording_masks(monkeypatch)
        _run(_fresh(uv_pool), **_params(2, 3))
        assert records
        for state, i, _t, features, finite, out in records:
            assert out == state._legacy(state._union(i), features, finite)

    def test_a_superset_on_other_trajectories_is_not_used(self, uv_pool, monkeypatch):
        """A stored deep list certifies a key only on the key's own
        trajectories."""
        records = _recording_masks(monkeypatch)
        _run(_fresh(uv_pool), **_params(2, 3))
        state, i, t, features, finite, _out = records[0]
        union = state._union(i)
        traj = tuple(key for key in union['keys'] if key in {k for k, *_ in finite})
        cols, _bits = state._canonical(union, features)
        raw = state._raw(i, traj)
        kept = tuple(c for c in cols if raw[c, c] > 0)
        assert state._derive(i, t, traj, kept) is not None
        assert state._derive(i, t, traj + ('another trajectory',), kept) is None

    def test_a_target_that_is_not_its_column_goes_to_the_legacy_screen(self, uv_pool, monkeypatch):
        """The union's cross moments are the target term's own column; a key
        whose target differs from it (here: ``Equation.evaluate`` scaled by
        1 + 2^-40) is screened by ``screened_masks`` on its own data."""
        from epde.structure.main_structures import Equation
        real_evaluate = Equation.evaluate

        def shifted(self, *, active_only=False):
            target, features = real_evaluate(self, active_only=active_only)
            return ({key: np.asarray(value) * (1.0 + 2.0 ** -40) for key, value in target.items()},
                    features)
        monkeypatch.setattr(Equation, 'evaluate', shifted)
        states, real_init = [], rps._ScreenCallState.__init__

        def init(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            states.append(self)
        monkeypatch.setattr(rps._ScreenCallState, '__init__', init)
        legacy, real_screened = [], subset_search.screened_masks

        def screened(*args, **kwargs):
            legacy.append(args[3:5])
            return real_screened(*args, **kwargs)
        monkeypatch.setattr(subset_search, 'screened_masks', screened)
        _run(_fresh(uv_pool), **_params(2, 3))
        assert states and all(s.stats['legacy'] == s.stats['keys'] > 0 for s in states)
        assert all(s.stats['searches'] == 0 for s in states)
        assert legacy and all(c == (2, 3) for c in legacy)


# ---------------------------------------------------------------------------
# The dyad memo (``rps._DYAD_MEMO``): each subset scored once per call, from
# the call's union slab Grams.
# ---------------------------------------------------------------------------

def _recording_dyad(monkeypatch):
    """Record every ``_ScreenCallState.dyad_scores`` call, with the per-key
    ``dyad_subset_scores`` of the key's own ``Xa`` / ``sw`` (read off
    ``_equation_variants``' frame) computed there, under the run's config --
    the slab partition follows the search's reference width."""
    import sys
    from epde.operators.common import survival
    records, real = [], rps._ScreenCallState.dyad_scores

    def recording(self, i, t, features, key, subsets, grid_shape, y):
        out = real(self, i, t, features, key, subsets, grid_shape, y)
        local = sys._getframe(1).f_locals
        records.append(dict(state=self, i=i, t=t, features=list(features), key=key,
                            subsets=[list(S) for S in subsets], y=y, shape=grid_shape,
                            out=None if out is None else out.copy(),
                            direct=survival.dyad_subset_scores(local['Xa'], y, local['sw'],
                                                                 grid_shape, subsets)))
        return out
    monkeypatch.setattr(rps._ScreenCallState, 'dyad_scores', recording)
    return records


class TestTheDyadMemo:

    @pytest.mark.parametrize('solve, ratios', [
        ('eigh', {}), ('hybrid', {}), ('csne', {}),
        # This pool's slabs are all well conditioned: forced bands put every
        # slab through the slab designs the memo builds (the target one of
        # their columns) -- the hybrid's and csne's design solve, and csne's
        # refinement step.
        ('hybrid', {'_HYBRID_RATIO': 1.0}),
        ('csne', {'_HYBRID_RATIO': 1.0}),
        ('csne', {'_HYBRID_RATIO': 1.0, '_CSNE_RATIO': 1.0})])
    @pytest.mark.parametrize('overrides', [_params(2), _params(2, 3), {}])
    def test_the_memo_scores_are_the_per_key_scores(self, uv_pool, monkeypatch, overrides, solve, ratios):
        """Every key's memo scores are ``dyad_subset_scores`` on its own
        columns, to round-off (the union's slab Grams -- and under the design
        solves its slab designs -- hold the target as one of their columns).

        Except where the score itself is round-off: a subset holding a term
        whose median slab coefficient is zero to round-off scores
        MAD/|median| ~ 1e15 or the cap, and the union's R factors (a QR of
        other columns) round differently from the key's own. Such scores
        must land in that class on both sides; their values are noise in any
        evaluation."""
        from epde.operators.common import survival
        monkeypatch.setattr(survival, '_DYAD_SOLVE', solve)
        for name, value in ratios.items():
            monkeypatch.setattr(survival, name, value)
        monkeypatch.setattr(rps, '_DYAD_MEMO', True)
        gathered, solve_min_norm = [], survival._solve_min_norm

        def counting(G, Gy, design=None):
            if design is None:
                return solve_min_norm(G, Gy)

            def counted(rows):
                gathered.append(len(rows))
                return design(rows)
            return solve_min_norm(G, Gy, counted)
        monkeypatch.setattr(survival, '_solve_min_norm', counting)
        records = _recording_dyad(monkeypatch)
        _run(_fresh(uv_pool), instability_metric='dyad', **overrides)
        assert records
        for r in records:
            assert r['out'] is not None
            noise = r['direct'] > 1e12
            assert np.array_equal(r['out'] > 1e12, noise)
            np.testing.assert_allclose(r['out'][~noise], r['direct'][~noise], rtol=1e-9, atol=1e-12)
        states = list({id(r['state']): r['state'] for r in records}.values())
        assert all(s.stats['dyad_direct'] == 0 for s in states)
        if ratios:
            assert sum(gathered) > 0

    def test_a_repeated_subset_is_reused(self, uv_pool, monkeypatch):
        """A subset already scored in the call (here: asked again by a key
        with one feature fewer) is returned from the memo, not rescored."""
        monkeypatch.setattr(rps, '_DYAD_MEMO', True)
        records = _recording_dyad(monkeypatch)
        _run(_fresh(uv_pool), instability_metric='dyad', **_params(2))
        r = next(r for r in records if len(r['features']) >= 2)
        state = r['state']
        monkeypatch.undo()
        first = state.dyad_scores(r['i'], r['t'], r['features'], r['key'], [[0], [1]],
                                  r['shape'], r['y'])
        scored, reused = state.stats['dyad_scored'], state.stats['dyad_reused']
        again = state.dyad_scores(r['i'], r['t'], r['features'][1:], r['key'], [[0]],
                                  r['shape'], r['y'])
        assert (state.stats['dyad_scored'], state.stats['dyad_reused']) == (scored, reused + 1)
        assert again[0] == first[1]

    def test_a_subset_has_one_score_whichever_key_asks(self, uv_pool, monkeypatch):
        """The same terms, target and trajectory score the same bits from
        two keys with different feature sets: they slice one union slab
        Gram."""
        monkeypatch.setattr(rps, '_DYAD_MEMO', True)
        records = _recording_dyad(monkeypatch)
        _run(_fresh(uv_pool), instability_metric='dyad', **_params(2))
        r = next(r for r in records if len(r['features']) >= 2)
        state = r['state']
        monkeypatch.undo()                              # the plain dyad_scores again
        wide, narrow = r['features'], r['features'][:1]
        scores = []
        for features in (wide, narrow):
            state._dyad_scores.clear()
            # the first feature alone, then with the constant (its index moves)
            scores.append(state.dyad_scores(r['i'], r['t'], features, r['key'],
                                            [[0], [0, len(features)]], r['shape'], r['y']))
        assert np.array_equal(scores[0], scores[1])

    def test_the_search_keeps_its_choices(self, uv_pool, monkeypatch):
        """On the uv system the memo changes no selected structure, and the
        fitted weights only at round-off."""
        monkeypatch.setattr(rps, '_DYAD_MEMO', False)
        off = _labels_and_weights(_run(_fresh(uv_pool), instability_metric='dyad', **_params(2)))
        monkeypatch.setattr(rps, '_DYAD_MEMO', True)
        on = _labels_and_weights(_run(_fresh(uv_pool), instability_metric='dyad', **_params(2)))
        assert on[0] == off[0]
        for w_on, w_off in zip(on[1], off[1]):
            np.testing.assert_allclose(w_on, w_off, rtol=1e-9, atol=1e-12)

    def test_on_for_coupled_systems_and_only_for_dyad(self, uv_pool, monkeypatch):
        """``_DYAD_MEMO`` defaults to ``'coupled'``: the call state is built
        for the coupled uv system under ``instability_metric='dyad'`` only;
        ``False`` keeps it off."""
        assert rps._DYAD_MEMO == 'coupled'

        def refuse(self, *args, **kwargs):
            raise AssertionError('the call state was built')
        monkeypatch.setattr(rps._ScreenCallState, '__init__', refuse)
        _run(_fresh(uv_pool), instability_metric='chi2', **_params(2))
        with pytest.raises(AssertionError, match='call state'):
            _run(_fresh(uv_pool), instability_metric='dyad', **_params(2))
        monkeypatch.setattr(rps, '_DYAD_MEMO', False)
        _run(_fresh(uv_pool), instability_metric='dyad', **_params(2))

    @pytest.mark.parametrize('memo, metric, n_equations, used', [
        ('coupled', 'dyad', 1, False), ('coupled', 'dyad', 2, True), ('coupled', 'dyad', 3, True),
        ('coupled', 'chi2', 2, False), (True, 'dyad', 1, True), (True, 'chi2', 3, False),
        (False, 'dyad', 3, False)])
    def test_the_memo_rule(self, monkeypatch, memo, metric, n_equations, used):
        """A single equation has no repeated subsets to reuse: under
        ``'coupled'`` it is scored per key."""
        monkeypatch.setattr(rps, '_DYAD_MEMO', memo)
        with using_config(instability_metric=metric):
            assert rps._use_dyad_memo(n_equations) is used
