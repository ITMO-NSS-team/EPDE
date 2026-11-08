"""Exact best subsets of each size by weighted least squares, in Gram space.

The screen of ``system_rps_mode='assignment'`` under its ``screen_size``
setting (``search_space.system_rps_params``): of an equation's subsets of at
most ``selected_terms_max_number`` columns, only the ``B`` best of each size
by weighted residual sum of squares are scored with the objectives' own
formulas. Every subset is still visited -- at a rank-one update each -- so the
screen is exact on its own statistic; what it gives up is that the Pareto axes
(WAPE, an instability) are no longer evaluated on every subset.

Ported, single-process, from the exact-enumeration prototype's ``cxfast``
(session c5a71b1a, ``sindy_next/cxfast.py``), whose top-B lists equal a
brute-force solve on 30 libraries. The arithmetic is unchanged:

Enumeration (each support exactly once)
  A level ``c`` holds the supports of ``n1`` one-cost and ``n2`` two-cost
  columns with ``n1 + 2 n2 = c``. The assignment mode passes unit costs (a term
  counts 1 whatever its factors, the constant counts 1), so a level is a subset
  size. Columns are permuted so the one-cost block comes first; a support is one
  strictly increasing index sequence, generated depth-first over prefixes.

Incremental RSS (Gram-space Cholesky)
  For a prefix P with factor rows U, the residual Gram is ``R_P = G - U^T U``,
  ``diag_P`` its diagonal, ``rb_P`` the residual cross-moment with y, and
  ``rss(P + j) = rss(P) - rb_P[j]^2 / diag_P[j]``. Extending P by q costs one
  row ``u = R_P[q, :] / sqrt(diag_P[q])``; an inner node is shared by its whole
  subtree, and the last two positions are evaluated as one array block.

Dependent columns
  G is the unit-norm Gram plus a 1e-12 ridge, so ``diag_P[j]`` is the squared
  sine of column j's angle to the prefix span plus the ridge. Only a pivot that
  round-off pushed below ``DEP_TOL = RIDGE / 2`` counts as dependent (rss
  unchanged). The ridge biases a support whose columns are near-dependent: for
  a pair at squared sine ``s`` whose difference direction carries target
  signal, the RSS excess is ~2e-2 of the target energy at ``s = 1e-10`` and
  ~0.17 at ``s = 1e-11`` (inviscid Burgers has such pairs: ``u*u_x`` against
  ``u^2/x``); supports without such pairs are exact to ~1e-11 of it. The screen
  only ranks: the survivors are refitted and scored by the tables, so the bias
  can change which near-dependent supports survive, never their scores.

Ties at the cut are broken by ``(rss, sorted column-index tuple)``: the caller
fixes the column order (the assignment mode sorts the terms by label), so the
same structure screens the same subsets whatever the order of its terms.

The exact screen (:class:`ExactScreen`, the assignment mode's engine)
  One canonical statistic ``v(S)`` (:func:`canonical_values`): the ridged
  weighted RSS of support S by sequential elimination of its columns in
  ascending canonical position, every operation elementwise -- so ``v(S)`` is
  a pure function of ``(A[S, S], b[S], yy)`` and has the same bits whichever
  superset, chunk or batch it is computed in. The level search
  (:func:`sweep_level`) is an exhaustive vectorised sweep whose leaves ARE
  those values, keeping the first ``D`` supports by ``(v, support tuple)``.
  With the per-call union Gram (:func:`assemble_from_union`: a subset's
  problem is bitwise a sub-block of a superset's) the list of a removed set
  F \\ R is derived from F's deeper list (:func:`derive`) instead of searched
  again. :class:`BestSubsets` stays as the oracle and the ``'legacy'`` engine.
"""
import itertools
from math import comb

import numpy as np

RIDGE = 1e-12
DEP_TOL = RIDGE / 2
QCHUNK = 128
#: The screen engine of ``system_rps_mode='assignment'``: ``'exact'``
#: (:class:`ExactScreen` with per-call derivation) or ``'legacy'``
#: (:func:`screened_masks` per key). Private: the A/B switch of the benchmark
#: harness, not a user setting.
_ENGINE = 'exact'
#: State doubles (all operands together) in one prefix chunk of
#: :func:`sweep_level`: small enough to stay in cache.
_SWEEP_CHUNK = 300_000
#: Leaves closed per array block of :func:`sweep_level` (cache-sized operands).
_LEAF_BLOCK = 32_768


class _Top(object):
    """The best ``B`` rows seen so far, plus every row tied with the B-th."""

    def __init__(self, B, k):
        self.B, self.k = B, k
        self.rss = np.empty(0)
        self.sup = np.empty((0, k), dtype=np.int32)
        self.buf_r, self.buf_s, self.nbuf = [], [], 0
        self.tau = np.inf

    def add(self, r, pre, q, j):
        s = np.empty((len(r), self.k), dtype=np.int32)
        if pre:
            s[:, :len(pre)] = pre
        s[:, -2] = q
        s[:, -1] = j
        self.buf_r.append(r)
        self.buf_s.append(s)
        self.nbuf += len(r)
        if self.nbuf >= 4 * self.B:
            self.flush()

    def flush(self):
        if not self.nbuf:
            return
        r = np.concatenate([self.rss] + self.buf_r)
        s = np.concatenate([self.sup] + self.buf_s)
        self.buf_r, self.buf_s, self.nbuf = [], [], 0
        if len(r) >= self.B:
            kth = np.partition(r, self.B - 1)[self.B - 1]
            keep = r <= kth
            r, s = r[keep], s[keep]
            self.tau = float(kth)
        self.rss, self.sup = r, s


def level_parts(n_one, n_two, c):
    """The ``(n1, n2)`` splits of level ``c``."""
    return [(c - 2 * n2, n2) for n2 in range(c // 2 + 1)
            if 0 <= c - 2 * n2 <= n_one and n2 <= n_two]


class BestSubsets(object):
    """Exact top-``B`` supports of each level of the columns of a unit-norm Gram.

    ``G`` is the Gram of the unit-norm (weighted) columns, ``b`` their cross
    moments with the (weighted) target, ``yy`` the target's energy, ``costs``
    each column's cost (1 or 2). :meth:`level` returns ``([(rss, support)],
    count)`` -- the support a sorted tuple of column indices into ``G``, the
    list sorted by ``(rss, support)``, and ``count`` the supports visited.
    """

    def __init__(self, G, b, yy, costs, B):
        costs = np.asarray(costs)
        if not set(np.unique(costs).tolist()) <= {1, 2}:
            raise ValueError('column costs must be 1 or 2, instead got {0}.'.format(
                sorted(set(np.unique(costs).tolist()))))
        one, two = np.flatnonzero(costs == 1), np.flatnonzero(costs == 2)
        self.order = np.concatenate([one, two])
        self.n_one, self.n_two = len(one), len(two)
        self.p = len(self.order)
        G = np.asarray(G, dtype=float)[np.ix_(self.order, self.order)] + RIDGE * np.eye(self.p)
        if not np.all(np.isfinite(G)):
            raise ValueError('non-finite Gram: a zero or non-finite column reached the screen.')
        self.G = np.ascontiguousarray(G)
        self.b = np.ascontiguousarray(np.asarray(b, dtype=float)[self.order])
        self.yy = float(yy)
        self.diag0 = np.ascontiguousarray(np.diag(self.G))
        self.B = int(B)

    # -- the depth-first search --------------------------------------------------
    def _bounds(self, i, n1, k):
        """Index range [lo, hi) of position i, leaving room for the later positions of its block."""
        if i < n1:
            return 0, self.n_one - (n1 - 1 - i)
        return self.n_one, self.p - (k - 1 - i)

    def _extend(self, state, e):
        Ut, diag, rb, rss = state
        piv = diag[e]
        if piv <= DEP_TOL:
            return state
        r = self.G[e] - Ut[:, e] @ Ut if Ut.shape[0] else self.G[e]
        s = np.sqrt(piv)
        u = r / s
        return (np.vstack((Ut, u[None, :])), diag - u * u, rb - u * (rb[e] / s),
                rss - rb[e] * rb[e] / piv)

    def _block(self, state, pre, n1, k, top):
        """Every support ``pre + (q, j)``: the last two positions as one array block."""
        Ut, diag, rb, rss = state
        iq, ij = k - 2, k - 1
        qlo, qhi = self._bounds(iq, n1, k)
        if pre:
            qlo = max(qlo, pre[-1] + 1)
        jlo0, jhi = self._bounds(ij, n1, k)
        same = (iq < n1) == (ij < n1)
        cnt = 0
        for q0 in range(qlo, qhi, QCHUNK):
            q1 = min(q0 + QCHUNK, qhi)
            jlo = max(jlo0, q0 + 1) if same else jlo0
            R = self.G[q0:q1, jlo:jhi]
            if Ut.shape[0]:
                R = R - Ut[:, q0:q1].T @ Ut[:, jlo:jhi]
            piv = diag[q0:q1]
            dep = piv <= DEP_TOL
            inv = np.where(dep, 0.0, 1.0 / np.where(dep, 1.0, piv))
            rbq = rb[q0:q1]
            m = q1 - q0
            cnt += m * (jhi - 1) - (q0 + q1 - 1) * m // 2 if same else m * (jhi - jlo)
            rss_q = rss - rbq * rbq * inv
            dn = diag[jlo:jhi] - R * R * inv[:, None]
            rn = rb[jlo:jhi] - R * (rbq * inv)[:, None]
            ok = dn > DEP_TOL
            val = rss_q[:, None] - np.where(ok, rn * rn / np.where(ok, dn, 1.0), 0.0)
            if same:
                val[np.arange(jlo, jhi)[None, :] <= np.arange(q0, q1)[:, None]] = np.inf
            rows, cols = np.nonzero((val <= top.tau) & (val < np.inf))
            if rows.size:
                top.add(val[rows, cols], pre, q0 + rows, jlo + cols)
        return cnt

    def _dfs(self, state, pre, n1, k, top):
        d = len(pre)
        if d == k - 2:
            return self._block(state, pre, n1, k, top)
        lo, hi = self._bounds(d, n1, k)
        if pre:
            lo = max(lo, pre[-1] + 1)
        cnt = 0
        for e in range(lo, hi):
            cnt += self._dfs(self._extend(state, e), pre + [e], n1, k, top)
        return cnt

    def _single(self, n1, top):
        lo, hi = (0, self.n_one) if n1 == 1 else (self.n_one, self.p)
        d = self.diag0[lo:hi]
        bb = self.b[lo:hi]
        ok = d > DEP_TOL
        val = self.yy - np.where(ok, bb * bb / np.where(ok, d, 1.0), 0.0)
        rows = np.flatnonzero(val <= top.tau)
        s = np.empty((len(rows), 1), dtype=np.int32)
        s[:, 0] = lo + rows
        top.buf_r.append(val[rows])
        top.buf_s.append(s)
        top.nbuf += len(rows)
        return hi - lo

    def _part(self, n1, n2, root):
        k = n1 + n2
        top = _Top(self.B, k)
        if k == 1:
            cnt = self._single(n1, top)
        else:
            state = (np.empty((0, self.p)), self.diag0, self.b, self.yy)
            for e in root:
                state = self._extend(state, e)
            cnt = self._dfs(state, list(root), n1, k, top)
        top.flush()
        return top.rss, top.sup, cnt

    def _tasks(self, c):
        """(task, exact support count) for every subtree of level c."""
        n_one, n_two, p = self.n_one, self.n_two, self.p
        out = []
        for n1, n2 in level_parts(n_one, n_two, c):
            k = n1 + n2
            if k <= 2:
                out.append(((n1, n2, ()), comb(n_one, n1) * comb(n_two, n2)))
            elif n1 >= 1:
                for e in range(0, n_one - n1 + 1):
                    out.append(((n1, n2, (e,)), comb(n_one - 1 - e, n1 - 1) * comb(n_two, n2)))
            else:
                for e in range(n_one, p - n2 + 1):
                    out.append(((n1, n2, (e,)), comb(p - 1 - e, n2 - 1)))
        out.sort(key=lambda te: -te[1])
        return out

    def level(self, c):
        """Top-``B`` supports of level ``c``: ``([(rss, support)], supports counted)``."""
        cand, tau, total = [], np.inf, 0
        for (n1, n2, root), expected in self._tasks(c):
            r, s, cnt = self._part(n1, n2, root)
            if cnt != expected:
                raise RuntimeError('support count mismatch at level {0}: {1} != {2}'.format(
                    c, cnt, expected))
            total += cnt
            keep = r <= tau
            for rr, row in zip(r[keep], s[keep]):
                cand.append((float(rr), tuple(sorted(int(v) for v in self.order[row]))))
            if len(cand) > 4 * self.B:
                cand.sort()
                cand = cand[:self.B]
                tau = cand[-1][0]
        cand.sort()
        return cand[:self.B], total


def unit_gram(X, y, sw):
    """The unit-norm weighted Gram of ``[X | 1]``, its cross moments with
    ``y`` and ``y``'s weighted energy -- the problem :class:`BestSubsets`
    ranks, the same weighted least squares the subset tables solve. Columns of
    zero weighted norm are left out; returns ``(G, b, yy, kept column indices)``."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    Xa = np.hstack([X, np.ones((n, 1))])
    sw = np.asarray(sw, dtype=float).reshape(-1)
    if not np.all(np.isfinite(sw)) or np.any(sw < 0):
        raise ValueError('sample weights must be finite and non-negative.')
    sq = np.sqrt(sw)
    Xw = Xa * sq[:, None]
    yw = np.asarray(y, dtype=float).reshape(-1) * sq
    norms = np.linalg.norm(Xw, axis=0)
    kept = np.flatnonzero(norms > 0)
    Xn = Xw[:, kept] / norms[kept]
    return Xn.T @ Xn, Xn.T @ yw, float(yw @ yw), kept


def screened_masks(X, y, sw, selected_max, screen_size, order=None) -> set:
    """Bitmasks over the ``X.shape[1] + 1`` columns of ``[X | 1]`` (the
    constant on the top bit) of the ``screen_size`` best subsets of each size
    1 .. ``selected_max`` by weighted RSS, every column costing 1.

    ``order`` lists the feature columns in the order ties are broken in (the
    constant always last); the masks are in the caller's column positions.
    Columns of zero weighted norm are left out: they cannot change a weighted
    fit, so a support holding one fits as the support without it.
    """
    p = X.shape[1]
    order = list(range(p)) if order is None else [int(j) for j in order]
    if sorted(order) != list(range(p)):
        raise ValueError('order must be a permutation of the feature columns.')
    perm = order + [p]                      # position in the screen -> column of [X | 1]
    G, b, yy, kept = unit_gram(np.asarray(X, dtype=float)[:, order], y, sw)
    columns = [perm[int(i)] for i in kept]
    top = min(int(selected_max), len(columns))
    if not np.any(b):
        # The target has no weighted component along any column (a zero
        # target included): every support's RSS is the target's energy, the
        # (rss, support) order is the lexicographic one, and the search would
        # keep every tied support on its way there.
        return {sum(1 << columns[i] for i in support)
                for c in range(1, top + 1)
                for support in itertools.islice(itertools.combinations(range(len(columns)), c),
                                                int(screen_size))}
    search = BestSubsets(G, b, yy, np.ones(len(columns), dtype=int), screen_size)
    masks = set()
    for c in range(1, top + 1):
        best, _count = search.level(c)
        for _rss, support in best:
            masks.add(sum(1 << columns[i] for i in support))
    return masks


# ---------------------------------------------------------------------------
# The exact screen.
# ---------------------------------------------------------------------------

def _extend(U, Dg, rb, rss, A_rows, e):
    """One elimination step of the canonical kernel, for N states at once.

    Row ``n`` of the state (``U`` a list of ``(N, C)`` factor rows, ``Dg`` /
    ``rb`` ``(N, C)`` residual diagonal and cross moments, ``rss`` ``(N,)``)
    eliminates its column ``e[n]``; ``A_rows[n]`` is ``A``'s row of that column
    over the state's ``C`` columns. A pivot at or below ``DEP_TOL`` gets no
    credit and projects nothing (a zero factor row). Every operation is
    elementwise and the factor rows are subtracted one at a time, in order, so
    a column's values do not depend on which other columns the state holds:
    :func:`canonical_values` (the support's own columns) and
    :func:`sweep_level` (every column) share this step and agree bit for bit.
    """
    n = np.arange(len(e))
    piv = Dg[n, e]
    dep = piv <= DEP_TOL
    r = A_rows
    for Ut in U:
        r = r - Ut[n, e][:, None] * Ut
    s = np.sqrt(np.where(dep, 1.0, piv))
    u = np.where(dep[:, None], 0.0, r / s[:, None])
    rbe = rb[n, e]
    return (U + [u], Dg - u * u, rb - u * (rbe / s)[:, None],
            rss - np.where(dep, 0.0, rbe * rbe / np.where(dep, 1.0, piv)))


def _extend_columns(U, Dg, rb, rss, A_rows, t):
    """:func:`_extend` for states stored column-major, ``(C, N)``, every
    state eliminating its column at position ``t``: the same elementwise
    operations on each entry (so the same bits), with contiguous runs over
    the N states for :func:`sweep_level`."""
    piv = Dg[t]
    dep = piv <= DEP_TOL
    r = A_rows
    for Ut in U:
        r = r - Ut[t][None, :] * Ut
    s = np.sqrt(np.where(dep, 1.0, piv))
    u = np.where(dep[None, :], 0.0, r / s[None, :])
    rbe = rb[t]
    return (U + [u], Dg - u * u, rb - u * (rbe / s)[None, :],
            rss - np.where(dep, 0.0, rbe * rbe / np.where(dep, 1.0, piv)))


def _pair_values(Uq, Uj, Dgq, Dgj, rbq, rbj, rss, Aqj):
    """``v`` of the last two positions ``(q, j)`` of supports whose earlier
    positions are eliminated: the 2x2 block in closed form (``Uq`` / ``Uj``
    the factor rows' entries at q and j, all arguments broadcast together)."""
    R = Aqj
    for uq, uj in zip(Uq, Uj):
        R = R - uq * uj
    dep = Dgq <= DEP_TOL
    inv = np.where(dep, 0.0, 1.0 / np.where(dep, 1.0, Dgq))
    cq = rbq * inv
    rssq = rss - rbq * cq
    dn = Dgj - R * R * inv
    rn = rbj - R * cq
    ok = dn > DEP_TOL
    return rssq - np.where(ok, rn * rn / np.where(ok, dn, 1.0), 0.0)


def _single_values(A, b, yy, j):
    """``v`` of the one-column supports ``j``."""
    d = A[j, j]
    bb = b[j]
    ok = d > DEP_TOL
    return yy - np.where(ok, bb * bb / np.where(ok, d, 1.0), 0.0)


def canonical_values(A, b, yy, supports):
    """The canonical statistic ``v(S)`` of every row of ``supports``.

    ``A`` is the ridged unit-norm Gram, ``b`` the cross moments, ``yy`` the
    target energy; ``supports`` an ``(N, k)`` array of strictly increasing
    positions. ``v(S)`` is the ridged RSS of S by sequential elimination of
    its columns in ascending position (:func:`_extend` for all but the last
    two, :func:`_pair_values` for those), every operation elementwise: a pure
    function of ``(A[S, S], b[S], yy)``, the same bits in any batch, superset
    or sweep."""
    S = np.asarray(supports, dtype=np.intp)
    N, k = S.shape
    if k == 0:
        raise ValueError('a support holds at least one column.')
    if k == 1:
        return _single_values(A, b, yy, S[:, 0])
    Dg = A[S, S]
    rb = b[S]
    rss = np.full(N, float(yy))
    U = []
    for t in range(k - 2):
        U, Dg, rb, rss = _extend(U, Dg, rb, rss, A[S[:, t][:, None], S], np.full(N, t))
    q, j = k - 2, k - 1
    return _pair_values([Ut[:, q] for Ut in U], [Ut[:, j] for Ut in U], Dg[:, q], Dg[:, j],
                        rb[:, q], rb[:, j], rss, A[S[:, q], S[:, j]])


def _combinations(n, m):
    """Every strictly increasing ``m``-tuple of ``range(n)``, lexicographic,
    as a ``(comb(n, m), m)`` array (``itertools.combinations``, vectorised)."""
    rows = np.zeros((1, 0), dtype=np.intp)
    for t in range(m):
        last = rows[:, -1] if t else np.full(len(rows), -1)
        counts = np.maximum(n - (m - t) - last, 0)        # room for the later positions
        total = int(counts.sum())
        nxt = (np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
               + np.repeat(last + 1, counts))
        rows = np.hstack([np.repeat(rows, counts, axis=0), nxt[:, None]])
    return rows


class _LevelTop(object):
    """The first ``D`` supports of one size by ``(v, support tuple)``.

    Leaves are pushed with their values; ``tau`` is the D-th value kept so
    far (inf before D are kept), and only leaves at or below it can still
    enter, ties included. A flush keeps exactly ``D`` rows, the total order
    deciding among ties, so memory stays at a few ``D`` whatever the ties."""

    def __init__(self, D, k):
        self.D, self.k = int(D), k
        self.v = np.empty(0)
        self.s = np.empty((0, k), dtype=np.intp)
        self.buf_v, self.buf_s, self.nbuf = [], [], 0
        self.tau = np.inf

    def push(self, v, s):
        if len(v):
            self.buf_v.append(v)
            self.buf_s.append(s)
            self.nbuf += len(v)
            if self.nbuf >= 4 * self.D:
                self.flush()

    def flush(self):
        if not self.nbuf:
            return
        v = np.concatenate([self.v] + self.buf_v)
        s = np.concatenate([self.s] + self.buf_s)
        self.buf_v, self.buf_s, self.nbuf = [], [], 0
        order = np.lexsort(tuple(s[:, i] for i in range(self.k - 1, -1, -1)) + (v,))[:self.D]
        self.v, self.s = v[order], s[order]
        if len(self.v) >= self.D:
            self.tau = float(self.v[-1])

    def result(self):
        self.flush()
        return self.v, self.s


def _admit(top, v, supports_of):
    """Push the leaves of ``v`` at or below ``top.tau``; ``supports_of(idx)``
    builds their supports. A non-finite leaf cannot come from a finite
    problem and is refused rather than silently dropped.

    When many leaves pass, those tied at ``tau`` whose support tuple is not
    below the D-th kept one are dropped first: the final D-th support is at
    or below the current one in the total order, so they cannot reach the
    list (a target orthogonal to every column only to round-off ties every
    leaf at ``yy``, and would otherwise lexsort every leaf)."""
    if not np.all(np.isfinite(v)):
        raise RuntimeError('the exact screen produced a non-finite subset value.')
    idx = np.nonzero(v <= top.tau)
    if not idx[0].size:
        return
    vals, S = v[idx], supports_of(idx)
    if len(vals) > top.D and np.isfinite(top.tau):
        ref = top.s[-1]
        below = np.zeros(len(vals), dtype=bool)
        equal = np.ones(len(vals), dtype=bool)
        for i in range(top.k):
            below |= equal & (S[:, i] < ref[i])
            equal &= S[:, i] == ref[i]
        keep = (vals < top.tau) | below
        vals, S = vals[keep], S[keep]
    top.push(vals, S)


def _prefix_chunks(p, m, budget):
    """The prefixes of :func:`sweep_level` -- every strictly increasing
    ``m``-tuple of ``range(p - 2)`` -- sorted by last position (lexicographic
    within one), in chunks of at most ``budget`` state doubles: ``(P, l)``
    pairs of prefixes and their last positions. Generated one last position
    at a time, so the whole table is never held."""
    pending, n_pending, first = [], 0, None
    for l in range(m - 1, p - 2):
        group = _combinations(l, m - 1)
        group = np.hstack([group, np.full((len(group), 1), l, dtype=np.intp)])
        start = 0
        while start < len(group):
            if first is None:
                first = l
            cap = max(1, budget // ((m + p - 1 - first) * (m + 2)))
            take = min(len(group) - start, cap - n_pending)
            pending.append(group[start:start + take])
            n_pending += take
            start += take
            if n_pending >= cap:
                P = np.concatenate(pending)
                yield P, P[:, -1]
                pending, n_pending, first = [], 0, None
    if pending:
        P = np.concatenate(pending)
        yield P, P[:, -1]


def sweep_level(A, b, yy, k, D, chunk=None):
    """The first ``D`` supports of size ``k`` by ``(v, support tuple)``, over
    every support of the ``len(b)`` columns: ``(values, supports, count)``.

    Sizes 1 and 2 evaluate :func:`canonical_values` on every support. From
    size 3 on, the prefixes of the first ``k - 2`` positions, sorted by their
    last position (:func:`_prefix_chunks`), are eliminated in chunks of at
    most ``chunk`` state doubles (all operands together; :data:`_SWEEP_CHUNK`
    by default). A chunk's states hold only the columns its supports can
    still reach -- each prefix's own columns, then the tail after the chunk's
    smallest last position -- stored column-major, and for each tail position
    q every pair ``(q, j > q)`` is closed in cache-sized blocks. These are
    :func:`canonical_values`'s elementwise operations, entry by entry, so the
    leaf values ARE the canonical ones. ``count`` is the number of supports
    evaluated; it must equal ``comb(len(b), k)``."""
    p = len(b)
    top = _LevelTop(D, k)
    if k <= 2:
        S = _combinations(p, k)
        _admit(top, canonical_values(A, b, yy, S), lambda idx: S[idx[0]])
        v, s = top.result()
        return v, s, len(S)
    m = k - 2
    diag = np.diagonal(A)
    count = 0
    for P, l in _prefix_chunks(p, m, int(_SWEEP_CHUNK if chunk is None else chunk)):
        tail = np.arange(int(l[0]) + 1, p)
        N = len(P)
        cols = np.vstack([P.T, np.broadcast_to(tail[:, None], (len(tail), N))])     # (C, N)
        U, Dg, rb, rss = [], diag[cols], b[cols], np.full(N, float(yy))
        for t in range(m):
            U, Dg, rb, rss = _extend_columns(U, Dg, rb, rss, A[P[:, t][None, :], cols], t)
        for c in range(len(tail) - 1):
            q = int(tail[c])
            n_act = int(np.searchsorted(l, q, side='left'))
            a, J = m + c, tail[c + 1:]
            Aq = A[q, J][:, None]
            step = max(1, _LEAF_BLOCK // len(J))          # leaf blocks that stay in cache
            for r0 in range(0, n_act, step):
                r1 = min(n_act, r0 + step)
                v = _pair_values([Ut[a, None, r0:r1] for Ut in U], [Ut[a + 1:, r0:r1] for Ut in U],
                                 Dg[a, None, r0:r1], Dg[a + 1:, r0:r1], rb[a, None, r0:r1],
                                 rb[a + 1:, r0:r1], rss[None, r0:r1], Aq)       # (len(J), r1 - r0)
                count += v.size

                def supports_of(idx, P=P[r0:r1], q=q, J=J):
                    return np.hstack([P[idx[1]], np.full((len(idx[1]), 1), q), J[idx[0]][:, None]])
                _admit(top, v, supports_of)
    v, s = top.result()
    return v, s, count


def lexicographic_levels(p, M, D):
    """``{k: supports}``: the first ``D`` supports of each size 1 .. ``M`` of
    ``p`` columns in lexicographic order -- the screen of a target with no
    weighted component along any column, where every support's value is the
    target's energy."""
    return {k: np.array(list(itertools.islice(itertools.combinations(range(p), k), int(D))),
                        dtype=np.intp).reshape(-1, k)
            for k in range(1, min(int(M), p) + 1)}


class ExactScreen(object):
    """The exact screen of one problem: the first ``D`` supports of each size
    1 .. ``M`` by ``(v, support tuple)``.

    ``A`` is the ridged unit-norm Gram (:func:`assemble_from_union`), ``b``
    the cross moments, ``yy`` the target energy. :meth:`levels` returns
    ``{k: (values, supports, whole)}`` -- ``supports`` an ``(n, k)`` array of
    positions into ``A``, ``whole`` True when the list holds every support of
    that size -- and records the supports evaluated per size in ``counts``. A
    target with no weighted component along any column takes the
    lexicographic lists without a search (every value is ``yy``)."""

    def __init__(self, A, b, yy, M, D):
        A = np.asarray(A, dtype=float)
        if not np.all(np.isfinite(A)):
            raise ValueError('non-finite Gram: a zero or non-finite column reached the screen.')
        self.A = np.ascontiguousarray(A)
        self.b = np.ascontiguousarray(np.asarray(b, dtype=float))
        self.yy = float(yy)
        self.M, self.D = int(M), int(D)
        self.p = len(self.b)
        self.counts = {}

    def levels(self):
        p, D = self.p, self.D
        if not np.any(self.b):
            out = {}
            for k, S in lexicographic_levels(p, self.M, D).items():
                out[k] = (np.full(len(S), self.yy), S, comb(p, k) <= D)
                self.counts[k] = 0
            return out
        out = {}
        for k in range(1, min(self.M, p) + 1):
            v, S, count = sweep_level(self.A, self.b, self.yy, k, D)
            if count != comb(p, k):
                raise RuntimeError('support count mismatch at size {0}: {1} != {2}'.format(
                    k, count, comb(p, k)))
            self.counts[k] = count
            out[k] = (v, S, comb(p, k) <= D)
        return out


def assemble_from_union(raw, rawy, yy, cols):
    """The screen problem of the union columns ``cols`` (positions into the
    union Gram ``raw``, in canonical order): ``(A, b, yy, kept)``, ``A`` the
    unit-norm Gram plus the ridge, ``b`` the cross moments with the target,
    ``kept`` the positions of ``cols`` whose weighted norm is not zero.

    Every entry is an elementwise function of the union's entries, so the
    problem of a subset of columns is bitwise the sub-block of a superset's
    -- what :func:`derive` relies on."""
    cols = np.asarray(cols, dtype=np.intp)
    kept = cols[raw[cols, cols] > 0]
    n = np.sqrt(raw[kept, kept])
    G = raw[np.ix_(kept, kept)] / (n[:, None] * n[None, :])
    return G + RIDGE * np.eye(len(kept)), rawy[kept] / n, float(yy), kept


def derive(deep, positions, M, B):
    """The first ``B`` supports of each size of a SUBSET F of a searched
    column set F', from F''s deep lists, or None when some size cannot be
    certified.

    ``deep`` is :meth:`ExactScreen.levels` of F'; ``positions[c]`` is the
    position in F of F''s column c, -1 for a column F lacks (both in
    canonical order, so the map is monotone). Restricted to F, F''s order is
    F's order (equal problems on F's supports, :func:`assemble_from_union`),
    so when at least ``B`` of F''s first ``D`` supports of a size lie in F --
    or the list holds every support of that size -- they are F's first ``B``.
    Returns ``{k: (n, k) supports in F's positions}`` for k = 1 .. min(M, |F|)."""
    positions = np.asarray(positions, dtype=np.intp)
    out = {}
    for k in range(1, min(int(M), int(np.count_nonzero(positions >= 0))) + 1):
        _v, S, whole = deep[k]
        mapped = positions[S]
        mapped = mapped[np.all(mapped >= 0, axis=1)]
        if len(mapped) < B and not whole:
            return None
        out[k] = mapped[:B]
    return out
