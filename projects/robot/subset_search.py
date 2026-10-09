r"""Best-subset search over a token library, scored by WAPE and a window-refit instability.

The general-library pipeline the robot scripts use, kept here so they run from the repository alone.

  library     products of at most two atoms (fields and their powers, time derivatives, coordinate
              powers and trig), symbolic duplicates merged, the target excluded (library, name_of)
  search      the best support per complexity c (a column built from k atoms costs k): exhaustive
              while a level holds <= limit supports, beam search above (search, batched_rss on the
              unit-norm Gram)
  candidates  OLS coefficients of each level's best support, WAPE on all rows, and the window-refit
              instability of every STATS setting (instab)
  instab      each candidate refit on sliding windows along every grid axis, the per-term CV^2 of the
              refits summed and divided by the number of axes; win14 = (1/4 of each axis, 10 window
              positions, relative ridge, linear windows), harness = (1/2, the harness positions,
              absolute ridge 1e-6, linear)
  front       the exact Pareto front of (WAPE, instability): j dominates i when it is no worse on both
              and strictly better on one; candidates without both objectives finite never enter
"""
import itertools
import time
from math import comb

import numpy as np


# ------------------------------------------------------------------ library
def name_of(expo):
    """Canonical product name from {factor key: exponent}; '' = constant."""
    parts = []
    for f in sorted(expo):
        e = expo[f]
        if e == 0:
            continue
        if f[0] == "pow":                       # field or coordinate power
            parts.append(f[1] if e == 1 else "%s^%d" % (f[1], e))
        else:                                   # derivative / trig: repeated factor
            parts.extend([f[1]] * e)
    return "*".join(parts)


def _grad(f, coord, axis):
    return np.gradient(f, coord, axis=axis, edge_order=2)


def _ode(t, fields, bw):
    der = {}
    for v, f in fields.items():
        d1 = _grad(f, t, 0)
        der[v + "_t"], der[v + "_tt"] = d1, _grad(d1, t, 0)
    sl = slice(bw, len(t) - bw)
    vals = {k: a[sl] for k, a in {**fields, **der}.items()}
    return vals, {"t": t[sl]}, (len(t) - 2 * bw,)


def atoms(vals, coords, variables, target, pde):
    out = []
    for v in variables:
        for e in (1, 2, 3):
            out.append((("pow", v), e, vals[v] ** e))
        ders = [v + "_t", v + "_tt"] + ([v + s for s in ("_x", "_y", "_xx", "_yy", "_xy")]
                                         if pde else [])
        for d in ders:
            if d != target and d in vals:
                out.append((("der", d), 1, vals[d]))
    for c, arr in coords.items():
        inv = np.divide(1.0, arr, out=np.zeros_like(arr), where=np.abs(arr) > 1e-12)
        out += [(("pow", c), 1, arr), (("pow", c), -1, inv)]
        for fn, nm in ((np.sin, "sin"), (np.cos, "cos")):
            out.append((("trig", "%s(%s)" % (nm, c)), 1, fn(arr)))
            out.append((("trig", "%s(2*%s)" % (nm, c)), 1, fn(2 * arr)))
    return out


def library(vals, coords, variables, target, pde):
    """(names, columns over the inner crop, atom counts); products of <= 2 atoms, merged."""
    at = atoms(vals, coords, variables, target, pde)
    cols = {}
    for k in (1, 2):
        for combo in itertools.combinations_with_replacement(range(len(at)), k):
            expo = {}
            for i in combo:
                key, e, _a = at[i]
                expo[key] = expo.get(key, 0) + e
            name = name_of(expo)
            if name == "" or name in cols:
                continue
            arr = np.ones_like(at[0][2])
            for i in combo:
                arr = arr * at[i][2]
            if np.all(np.isfinite(arr)) and np.linalg.norm(arr) > 0:
                cols[name] = (k, arr)
    names = sorted(cols, key=lambda n: (cols[n][0], n))
    return names, np.column_stack([cols[n][1] for n in names]), np.array([cols[n][0] for n in names])


# ------------------------------------------------------------------ best support per complexity
RIDGE = 1e-12


def batched_rss(gm, supports, chunk=200000):
    out = np.empty(len(supports))
    if len(supports) == 0:
        return out
    k = supports.shape[1]
    eye = RIDGE * np.eye(k)
    for s in range(0, len(supports), chunk):
        S = supports[s:s + chunk]
        Gs = gm.G[S[:, :, None], S[:, None, :]] + eye
        bs = gm.b[S]
        c = np.linalg.solve(Gs, bs[..., None])[..., 0]
        out[s:s + chunk] = gm.yy - np.einsum("ij,ij->i", bs, c)
    return out


def _combos(idx, n):
    rows = list(itertools.combinations(idx, n))
    return np.array(rows, dtype=int).reshape(len(rows), n)


def level_size(n_one, n_two, c):
    return sum(comb(n_one, c - 2 * n2) * comb(n_two, n2) for n2 in range(c // 2 + 1)
               if c - 2 * n2 <= n_one)


def _merge_top(top, sup, rss, B):
    if len(rss) > B:
        sel = np.argpartition(rss, B)[:B]
        sup, rss = sup[sel], rss[sel]
    top.extend((float(r), tuple(int(v) for v in sorted(s))) for r, s in zip(rss, sup))
    top.sort()
    seen, out = set(), []
    for r, s in top:
        if s not in seen:
            seen.add(s)
            out.append((r, s))
        if len(out) >= B:
            break
    return out


def exhaustive_level(gm, one, two, c, B, block=2000):
    top = []
    for n2 in range(c // 2 + 1):
        n1 = c - 2 * n2
        if n1 > len(one) or n2 > len(two):
            continue
        c1 = _combos(one, n1)
        c2_iter = itertools.combinations(two, n2)
        while True:
            rows = list(itertools.islice(c2_iter, block))
            c2 = np.array(rows, dtype=int).reshape(len(rows), n2)
            if len(c2) == 0:
                break
            sup = np.hstack([np.repeat(c1, len(c2), axis=0), np.tile(c2, (len(c1), 1))])
            top = _merge_top(top, sup, batched_rss(gm, sup), B)
            if n2 == 0:
                break
    return top


def _polish(gm, S, ks, rss, max_sweeps=20):
    p = len(ks)
    S = list(S)
    for _ in range(max_sweeps):
        best = None
        for pos in range(len(S)):
            same = np.flatnonzero(ks == ks[S[pos]])
            cand = np.array([S[:pos] + [j] + S[pos + 1:] for j in same if j not in S], dtype=int)
            if len(cand) == 0:
                continue
            r = batched_rss(gm, cand)
            i = int(np.argmin(r))
            if r[i] < rss - max(1e-9 * rss, 1e-13 * gm.yy) and (best is None or r[i] < best[0]):
                best = (float(r[i]), list(cand[i]))
        if best is None:
            break
        rss, S = best
    return float(rss), tuple(sorted(int(v) for v in S))


def beam_level(gm, ks, tops, c, B, polish=5):
    one, two = np.flatnonzero(ks == 1), np.flatnonzero(ks == 2)
    sups = set()
    for src, pool in ((c - 1, one), (c - 2, two)):
        for _r, S in tops.get(src, []):
            for j in pool:
                if j not in S:
                    sups.add(tuple(sorted(S + (int(j),))))
    by_len = {}
    for s in sups:
        by_len.setdefault(len(s), []).append(s)
    top = []
    for _k, lst in by_len.items():
        arr = np.array(lst, dtype=int)
        top = _merge_top(top, arr, batched_rss(gm, arr), B)
    for r, S in list(top[:polish]):
        top.append(_polish(gm, S, ks, r))
    top = sorted(set(top))[:B]
    return top


def search(gm, ks, cmax, B=1000, limit=5e7, force_exhaustive_upto=None):
    one, two = np.flatnonzero(ks == 1), np.flatnonzero(ks == 2)
    tops, out = {}, {}
    exhaustive = True
    for c in range(1, cmax + 1):
        small = level_size(len(one), len(two), c) <= limit
        if force_exhaustive_upto is not None:
            small = c <= force_exhaustive_upto
        if exhaustive and small:
            tops[c] = exhaustive_level(gm, one, two, c, B)
            mode = "exhaustive"
        else:
            exhaustive = False
            tops[c] = beam_level(gm, ks, tops, c, B)
            mode = "beam"
        if tops[c]:
            out[c] = dict(best=list(tops[c][0][1]), rss=tops[c][0][0], top=tops[c], mode=mode)
    return out


# ------------------------------------------------------------------ window-refit instability
def _cv_squared(weights: np.ndarray) -> np.ndarray:
    """Per-feature CV^2 (var / mean^2) from a (n_windows, n_features) matrix.

    Two saturation paths both land on 1e12, and downstream they are
    indistinguishable from each other and from a genuinely wild coefficient:
    a mean at or below 1e-12, where the ratio is meaningless, and a single
    window, where `ddof=1` makes the standard deviation NaN. A grid small
    enough to yield one window is rare in this suite but not impossible, so
    read a per-term instability of 1e12 as "not measured" rather than as a
    measurement.
    """
    std = weights.std(axis=0, ddof=1)
    mu = weights.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_sq = np.where(np.abs(mu) > 1e-12, (std ** 2) / (mu ** 2), std ** 2 * 1e12)
    return np.where(np.isfinite(cv_sq), cv_sq, 1e12)


def slice_grams(A, b, grid, d):
    k = A.shape[1]
    Ad = np.moveaxis(A.reshape(grid + (k,)), d, 0).reshape(grid[d], -1, k)
    bd = np.moveaxis(b.reshape(grid), d, 0).reshape(grid[d], -1)
    return np.einsum("imk,iml->ikl", Ad, Ad), np.einsum("imk,im->ik", Ad, bd)


def window_coefs(Gs, gs, w, starts, ridge, circular):
    k = Gs.shape[1]
    if circular:
        Gs, gs = np.concatenate([Gs, Gs]), np.concatenate([gs, gs])
    PG = np.concatenate([np.zeros((1, k, k)), np.cumsum(Gs, axis=0)])
    Pg = np.concatenate([np.zeros((1, k)), np.cumsum(gs, axis=0)])
    s = np.asarray(starts)
    Gw, gw = PG[s + w] - PG[s], Pg[s + w] - Pg[s]
    if ridge == "abs":
        Gw = Gw + 1e-6 * np.eye(k)
    else:
        lam = 1e-10 * np.trace(Gw, axis1=1, axis2=2) / k
        Gw = Gw + lam[:, None, None] * np.eye(k)
    try:
        return np.linalg.solve(Gw, gw[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.array([np.linalg.lstsq(g, v, rcond=None)[0] for g, v in zip(Gw, gw)])


def starts_for(n_d, w, pos, circular, harness_rule=False):
    if harness_rule:
        return list(range(0, w + 1, max(1, (w + 1) // 30)))
    n_st = n_d if circular else n_d - w + 1
    if pos == "all":
        return list(range(n_st))
    return list(range(0, n_st, max(1, n_st // int(pos))))


def instab(A, b, grid, f, pos, ridge, mode, harness_rule=False):
    per_axis = []
    for d, n_d in enumerate(grid):
        w = n_d // 2 if harness_rule else max(2, int(np.floor(f * n_d)))
        if w < 2 or w >= n_d and not harness_rule:
            continue
        Gs, gs = slice_grams(A, b, grid, d)
        st = starts_for(n_d, w, pos, mode == "circular", harness_rule)
        per_axis.append(window_coefs(Gs, gs, w, st, ridge, mode == "circular"))
    if not per_axis:
        return 0.0, 0.0
    pooled = _cv_squared(np.vstack(per_axis))
    total = float(np.sum(pooled)) / len(grid)
    max2 = float(max(np.max(_cv_squared(p)) for p in per_axis))
    return total, max2


# ------------------------------------------------------------------ candidates and front
STATS = {"win14": dict(f=0.25, pos="10", ridge="rel", mode="linear", harness_rule=False),
         "harness": dict(f=0.5, pos="30", ridge="abs", mode="linear", harness_rule=True)}


class Gram:
    def __init__(self, X, y):
        n = np.linalg.norm(X, axis=0)
        Xn = X / n
        self.G, self.b, self.yy = Xn.T @ Xn, Xn.T @ y, float(y @ y)


def candidates(eq, label, res=None):
    """res: a precomputed search result (as search returns); default search(...)."""
    X, y, lat, ks, names = eq["X"], eq["y"], eq["lattice"], eq["ks"], eq["names"]
    T = sorted(names.index(t) for t in eq["truth"])
    c_true = int(ks[T].sum())
    cmax = max(5, c_true + 1)
    gm = Gram(X[lat], y[lat])
    t0 = time.time()
    if res is None:
        res = search(gm, ks, cmax, B=1000, limit=5e7)
    out = []
    for c, r in res.items():
        S = r["best"]
        coef = np.linalg.lstsq(X[lat][:, S], y[lat], rcond=None)[0]
        A = X[:, S]
        pred = A @ coef
        cand = dict(target=eq["target"], label=label, c=int(c), mode=r["mode"],
                    terms={names[j]: float(v) for j, v in zip(S, coef)},
                    J=1.0 if (sorted(S) == T and label == "true") else 0.0,
                    wape=float(np.sum(np.abs(y - pred)) / np.sum(np.abs(y))))
        for key, kw in STATS.items():
            cand[key] = instab(A, y, eq["grid"], **kw)[0]
        out.append(cand)
    info = dict(c_true=c_true, cmax=cmax, seconds=round(time.time() - t0, 1),
                modes={int(c): r["mode"] for c, r in res.items()})
    if label == "true" and c_true in res:
        top = res[c_true]["top"]
        # the truth's rss from the same list as its rivals (one solver), the truth itself and
        # repeated entries of one support never counted, round-off ties (1e-12 yy) not ahead
        own = dict((tuple(sorted(S)), r) for r, S in top).get(tuple(T))
        rss_t = own if own is not None else float(batched_rss(gm, np.array([T]))[0])
        ahead = {tuple(sorted(S)) for r, S in top if r < rss_t - 1e-12 * gm.yy}
        rank = len(ahead - {tuple(T)}) + 1
        info.update(truth_rank_at_c=rank if rank <= len(top) else ">%d" % len(top),
                    truth_is_best=sorted(res[c_true]["best"]) == T,
                    runner_up_ratio=(top[1][0] / top[0][0] if len(top) > 1 and top[0][0] > 0
                                     else None))
    return out, info


def exact_front(debugs):
    """Indices of the non-dominated candidates under an exact comparison of the two
    objectives (Discrepancy = wape, Instability): j dominates i when it is no worse on
    both and strictly better on one. Candidates without both objectives finite, or that
    predict nothing, never enter (as in the harness)."""
    import math
    pts = []
    for i, sd in enumerate(debugs):
        if not sd or sd.get("degenerate"):
            continue
        w, s = sd.get("wape"), sd.get("instability")
        if w is None or s is None:
            continue
        w, s = float(w), float(s)
        if not (math.isfinite(w) and math.isfinite(s)):
            continue
        pts.append((i, w, s))
    return [i for i, wi, si in pts
            if not any(j != i and wj <= wi and sj <= si and (wj < wi or sj < si)
                       for j, wj, sj in pts)]


def front(cands, idx, d, i):
    dbg = [{"wape": cands[j].get(d), "instability": cands[j].get(i)} for j in idx]
    for x in dbg:
        if x["wape"] is None or x["instability"] is None:
            x["wape"] = None
    return [idx[j] for j in exact_front(dbg)]
