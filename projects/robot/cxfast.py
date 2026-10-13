r"""Exact complexity-level best-subset enumeration, fast. Drop-in for cxsearch.exhaustive_level and
cxsearch.search: same level definition, same output format, same top-B tie order (rss, support).

Enumeration (each support exactly once)
  Level c = supports of n1 one-atom and n2 two-atom columns, n1 + 2 n2 = c. Columns are permuted so
  the one-atom block comes first; a support of part (n1, n2) is then one strictly increasing index
  sequence of length k = n1 + n2 whose first n1 positions lie in the one-atom block and last n2 in
  the two-atom block. A depth-first tree over prefixes generates them.

Incremental RSS (Gram-space Cholesky)
  For a prefix P with factor rows U (one row per non-dependent prefix column), the residual Gram is
  R_P = G - U^T U, diag_P its diagonal, rb_P the residual cross-moment with y, and
      rss(P + j) = rss(P) - rb_P[j]^2 / diag_P[j].
  Extending P by q costs one row u = R_P[q, :] / sqrt(diag_P[q]); an inner node is shared by its
  whole subtree. At depth k - 2 the last two positions (q, j) are evaluated as one array block.

Dependent columns
  G is the unit-norm Gram plus cxsearch's 1e-12 ridge, so diag_P[j] is sin^2 of column j's angle to
  the prefix span plus a ridge term, never below 1e-12 in exact arithmetic -- the same ridged
  problem cxsearch solves, exact dependencies (sin(2x) vs sin(x)*cos(x)) included. Only a pivot
  that round-off pushed below DEP_TOL = RIDGE / 2 counts as dependent (rss unchanged, no factor
  row). A larger threshold is wrong: inviscid Burgers has real near-identity pairs (u*u_x vs
  u^2/x, sin^2 ~ 1e-11) and DEP_TOL = 1e-9 dropped their credit (cxfast_burg_diag.py); at
  <= 1e-11 the top 1000 equals cxsearch's on all 30 libraries.
  Accuracy limit (shared with cxsearch, not introduced here): the 1e-12 ridge biases a support
  whose columns are near-dependent. For a pair with sin^2 = s whose difference direction carries
  target signal, the RSS excess vs lstsq on the data is ~ g * 1e-12 / (1e-12 + s/2) of that
  signal's energy g: ~2e-2 yy at s = 1e-10, ~0.17 yy at s = 1e-11 (review, Oct 2026). Round-off
  is ~1e-6 yy. Supports without such pairs are exact to ~1e-11 yy.

Parallelism
  Subtrees rooted at the first position are independent tasks (exact support counts known up
  front, largest first). Each task keeps its own top B plus every row tied with its B-th value;
  the parent takes the top B of the union by (rss, sorted original-index tuple), which is the
  level's top B in cxsearch's order. Workers run single-threaded BLAS (the blocks are
  elementwise-bound). Spawned workers re-import the caller's __main__ module: keep heavy imports
  (newsys -> epde -> torch) inside main() of a driver, or each worker costs ~1 GB.

  FastSearch(gm, ks, B, workers) -> .level(c) -> (top, count)   top = [(rss, support)], sorted
  search(gm, ks, cmax, B, limit, workers) -> as cxsearch.search, plus seconds / count per level
"""
import os
import time
import multiprocessing as mp
from math import comb

import numpy as np

RIDGE = 1e-12          # cxsearch.RIDGE
DEP_TOL = RIDGE / 2
QCHUNK = 128           # numpy / 128 beat numexpr (fused) and 512 on every NS level-7 part
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_W = {}


def _init(G, b, yy, n_one, B):
    _W.update(G=G, b=b, yy=float(yy), diag0=np.ascontiguousarray(np.diag(G)), n_one=int(n_one),
              p=G.shape[0], B=int(B))


def _bounds(i, n1, k):
    """Index range [lo, hi) of position i, leaving room for the later positions of its block."""
    if i < n1:
        return 0, _W["n_one"] - (n1 - 1 - i)
    return _W["n_one"], _W["p"] - (k - 1 - i)


class _Top:
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
            # keep every row tied with the B-th value: the parent breaks ties by support tuple
            kth = np.partition(r, self.B - 1)[self.B - 1]
            keep = r <= kth
            r, s = r[keep], s[keep]
            self.tau = float(kth)
        self.rss, self.sup = r, s


def _extend(state, e):
    Ut, diag, rb, rss = state
    piv = diag[e]
    if piv <= DEP_TOL:
        return state
    G = _W["G"]
    r = G[e] - Ut[:, e] @ Ut if Ut.shape[0] else G[e]
    s = np.sqrt(piv)
    u = r / s
    return (np.vstack((Ut, u[None, :])), diag - u * u, rb - u * (rb[e] / s),
            rss - rb[e] * rb[e] / piv)


def _block(state, pre, n1, k, top):
    """All supports pre + (q, j): positions k-2 and k-1 as one array block (chunked over q)."""
    Ut, diag, rb, rss = state
    G = _W["G"]
    iq, ij = k - 2, k - 1
    qlo, qhi = _bounds(iq, n1, k)
    if pre:
        qlo = max(qlo, pre[-1] + 1)
    jlo0, jhi = _bounds(ij, n1, k)
    same = (iq < n1) == (ij < n1)
    cnt = 0
    for q0 in range(qlo, qhi, QCHUNK):
        q1 = min(q0 + QCHUNK, qhi)
        jlo = max(jlo0, q0 + 1) if same else jlo0
        R = G[q0:q1, jlo:jhi]
        if Ut.shape[0]:
            R = R - Ut[:, q0:q1].T @ Ut[:, jlo:jhi]
        piv = diag[q0:q1]
        dep = piv <= DEP_TOL
        inv = np.where(dep, 0.0, 1.0 / np.where(dep, 1.0, piv))
        rbq = rb[q0:q1]
        m = q1 - q0
        # j > q is automatic across blocks (one-atom indices < two-atom indices)
        cnt += m * (jhi - 1) - (q0 + q1 - 1) * m // 2 if same else m * (jhi - jlo)
        rss_q = rss - rbq * rbq * inv
        dn = diag[jlo:jhi] - R * R * inv[:, None]
        rn = rb[jlo:jhi] - R * (rbq * inv)[:, None]
        ok = dn > DEP_TOL
        val = rss_q[:, None] - np.where(ok, rn * rn / np.where(ok, dn, 1.0), 0.0)
        if same:
            val[np.arange(jlo, jhi)[None, :] <= np.arange(q0, q1)[:, None]] = np.inf
        # <= keeps ties with the cutoff; invalid pairs (j <= q) are inf and never kept
        rows, cols = np.nonzero((val <= top.tau) & (val < np.inf))
        if rows.size:
            top.add(val[rows, cols], pre, q0 + rows, jlo + cols)
    return cnt


def _dfs(state, pre, n1, k, top):
    d = len(pre)
    if d == k - 2:
        return _block(state, pre, n1, k, top)
    lo, hi = _bounds(d, n1, k)
    if pre:
        lo = max(lo, pre[-1] + 1)
    cnt = 0
    for e in range(lo, hi):
        cnt += _dfs(_extend(state, e), pre + [e], n1, k, top)
    return cnt


def _single(n1, top):
    """k = 1: one column, no prefix."""
    lo, hi = (0, _W["n_one"]) if n1 == 1 else (_W["n_one"], _W["p"])
    d = _W["diag0"][lo:hi]
    bb = _W["b"][lo:hi]
    ok = d > DEP_TOL
    val = _W["yy"] - np.where(ok, bb * bb / np.where(ok, d, 1.0), 0.0)
    rows = np.flatnonzero(val <= top.tau)
    s = np.empty((len(rows), 1), dtype=np.int32)
    s[:, 0] = lo + rows
    top.buf_r.append(val[rows])
    top.buf_s.append(s)
    top.nbuf += len(rows)
    return hi - lo


def _task(t):
    n1, n2, root = t
    k = n1 + n2
    top = _Top(_W["B"], k)
    if k == 1:
        cnt = _single(n1, top)
    else:
        state = (np.empty((0, _W["p"])), _W["diag0"], _W["b"], _W["yy"])
        for e in root:
            state = _extend(state, e)
        cnt = _dfs(state, list(root), n1, k, top)
    top.flush()
    return t, top.rss, top.sup, cnt


class _Collector:
    """_Top's interface with a FIXED threshold: keeps every support with rss <= tau, none dropped,
    up to ``cap`` rows (then ``overflow`` is set and nothing more is kept)."""

    def __init__(self, k, tau, cap):
        self.k, self.tau, self.cap = k, tau, int(cap)
        self.buf_r, self.buf_s, self.nbuf = [], [], 0
        self.overflow = False

    def add(self, r, pre, q, j):
        if self.nbuf + len(r) > self.cap:
            self.overflow = True
            return
        s = np.empty((len(r), self.k), dtype=np.int32)
        if pre:
            s[:, :len(pre)] = pre
        s[:, -2] = q
        s[:, -1] = j
        self.buf_r.append(r)
        self.buf_s.append(s)
        self.nbuf += len(r)

    def flush(self):
        pass

    def arrays(self):
        if not self.nbuf:
            return np.empty(0), np.empty((0, self.k), dtype=np.int32)
        return np.concatenate(self.buf_r), np.concatenate(self.buf_s)


def _task_collect(t):
    n1, n2, root, tau, cap = t
    k = n1 + n2
    col = _Collector(k, tau, cap)
    if k == 1:
        cnt = _single(n1, col)
    else:
        state = (np.empty((0, _W["p"])), _W["diag0"], _W["b"], _W["yy"])
        for e in root:
            state = _extend(state, e)
        cnt = _dfs(state, list(root), n1, k, col)
    r, s = col.arrays()
    return t, r, s, cnt, col.overflow or col.nbuf > col.cap


def level_parts(n_one, n_two, c):
    return [(c - 2 * n2, n2) for n2 in range(c // 2 + 1)
            if 0 <= c - 2 * n2 <= n_one and n2 <= n_two]


class FastSearch:
    def __init__(self, gm, ks, B=1000, workers=4, min_pool_size=2e6):
        ks = np.asarray(ks)
        assert set(np.unique(ks).tolist()) <= {1, 2}, "atom counts must be 1 or 2"
        one, two = np.flatnonzero(ks == 1), np.flatnonzero(ks == 2)
        self.order = np.concatenate([one, two])
        self.n_one, self.n_two = len(one), len(two)
        self.p = len(self.order)
        G = gm.G[np.ix_(self.order, self.order)] + RIDGE * np.eye(self.p)
        assert np.all(np.isfinite(G)), "non-finite Gram (a zero column?)"
        self.args = (np.ascontiguousarray(G), np.ascontiguousarray(gm.b[self.order]),
                     float(gm.yy), self.n_one, B)
        self.B, self.workers, self.pool = B, workers, None
        self.min_pool_size = min_pool_size

    def _get_pool(self):
        if self.pool is None:
            saved = {v: os.environ.get(v) for v in _THREAD_VARS}
            for v in _THREAD_VARS:
                os.environ[v] = "1"
            try:
                self.pool = mp.get_context("spawn").Pool(self.workers, initializer=_init,
                                                         initargs=self.args)
            finally:
                for v, val in saved.items():
                    if val is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = val
        return self.pool

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def tasks(self, c):
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

    def level(self, c, workers=None):
        """Top-B supports of level c: ([(rss, sorted original-index tuple)], supports counted)."""
        workers = self.workers if workers is None else workers
        tasks = self.tasks(c)
        expected = dict(tasks)
        size = sum(expected.values())
        if workers <= 1 or size < self.min_pool_size:
            _init(*self.args)
            results = map(_task, [t for t, _n in tasks])
        else:
            results = self._get_pool().imap_unordered(_task, [t for t, _n in tasks], chunksize=1)
        cand, tau, total = [], np.inf, 0
        for t, r, s, cnt in results:
            assert cnt == expected[t], ("support count mismatch", t, cnt, expected[t])
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

    def collect(self, c, tau, cap=10 ** 7, workers=None):
        """EVERY support of level c with rss <= tau (no top-B truncation).

        Returns ({k: (rss array, (m, k) int32 array of sorted ORIGINAL column indices)}, supports
        counted, overflow) -- overflow True means more than ``cap`` rows qualified in some task or
        overall, and the result is then incomplete."""
        workers = self.workers if workers is None else workers
        tasks = self.tasks(c)
        expected = dict(tasks)
        size = sum(expected.values())
        jobs = [t + (float(tau), cap) for t, _n in tasks]
        if workers <= 1 or size < self.min_pool_size:
            _init(*self.args)
            results = map(_task_collect, jobs)
        else:
            results = self._get_pool().imap_unordered(_task_collect, jobs, chunksize=1)
        parts, total, overflow, n_rows = {}, 0, False, 0
        for t, r, s, cnt, over in results:
            assert cnt == expected[t[:3]], ("support count mismatch", t[:3], cnt)
            total += cnt
            overflow |= bool(over)
            if len(r) and n_rows + len(r) > cap:
                overflow = True          # global cap: keep memory bounded, drain the rest
                continue
            if len(r):
                orig = np.sort(self.order[s], axis=1).astype(np.int32)
                parts.setdefault(orig.shape[1], []).append((r, orig))
                n_rows += len(r)
        out = {k: (np.concatenate([a for a, _b in v]), np.concatenate([b for _a, b in v]))
               for k, v in parts.items()}
        return out, total, overflow


def search(gm, ks, cmax, B=1000, limit=5e7, workers=4, force_exhaustive_upto=None, log=None):
    """cxsearch.search with the fast exhaustive levels; beam (cxsearch.beam_level) above."""
    import subset_search as CX          # level_size, beam_level (cxsearch, copied verbatim)
    ks = np.asarray(ks)
    one, two = np.flatnonzero(ks == 1), np.flatnonzero(ks == 2)
    tops, out = {}, {}
    exhaustive = True
    with FastSearch(gm, ks, B=B, workers=workers) as fs:
        for c in range(1, cmax + 1):
            size = CX.level_size(len(one), len(two), c)
            small = size <= limit
            if force_exhaustive_upto is not None:
                small = c <= force_exhaustive_upto
            t0 = time.time()
            count = None
            if exhaustive and small:
                tops[c], count = fs.level(c)
                mode = "exhaustive"
            else:
                exhaustive = False
                tops[c] = CX.beam_level(gm, ks, tops, c, B)
                mode = "beam"
            dt = time.time() - t0
            if log:
                log("  c=%d %-10s %8.1fs  level size %.3g" % (c, mode, dt, size))
            if tops[c]:
                out[c] = dict(best=list(tops[c][0][1]), rss=tops[c][0][0], top=tops[c], mode=mode,
                              seconds=round(dt, 2), count=count, level_size=size)
    return out
