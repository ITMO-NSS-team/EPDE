"""The fill-and-prune move of plain mode (``system_rps_params['fill_probability']``).

With that probability a plain mutation replaces the move it drew by a FILL of
the equation from the whole library, followed by a probabilistic PRUNE:

1. Candidates: every library term (:func:`library_terms`, the terms the pool
   can draw) whose column is usable, except the target, the other equations'
   targets, the terms related to the target by division (the plain fit would
   drop those) and, when the gene found its term by the factor functions
   alone, the gene's exact signature. The current terms play no part: the
   fill replaces them.
2. Beam: an exact weighted least-squares beam over the candidates
   (:func:`beam_search`) from the empty set up to
   ``K = min(plain cap, max_terms_number - 1, candidates)`` terms, keeping the
   ``fill_beam_width`` supports of lowest residual at every size. The
   residual is relative (the target's weighted norm is 1) and has no constant.
3. Level: ``fill_levels='all'`` draws the size k uniformly from 1..K (a law
   shorter than K is held by no size-K support); ``'max'`` takes K.
4. Fill: one support of size k, drawn among the ``fill_top`` best with
   weights 1 / residual.
5. Prune: a chain from the fill down to two terms; at each stage every term i
   leaves with probability ``min(FILL_PRUNE_CAP, fill_prune_alpha * w_i * n)``,
   ``w_i ~ 1 / (residual rise when i alone leaves)``, at least one leaves
   (drawn by ``w``), never below two. One stage of the chain, drawn uniformly
   (the fill included), becomes the equation's terms besides its target.
6. Install: the target (the same object) plus fresh copies of the library
   terms; a structure equal to the current one falls back to the drawn move.

RNG order per call: the coin; then, once the beam is reached, the level (before
the beam is looked up, so a cached beam draws the same), the fill, the prune
stages, the stage. A fallback before that consumes the coin alone; the Gram,
candidate and beam computations draw nothing. Every fallback is a loop-stats record
(``PlainMutation.fill_fallback.<reason>``); a reason that holds for the whole equation is also warned of once
per fit.

The Gram: the library's columns on the evaluation rows (weak rows: one store
per equation slot), weighted by the fit's row weights and normalised to a unit
diagonal (a non-finite or zero column is unusable), built once per fit at the
first attempt. Beams are kept in memory per store.

Every constant here is the setting of the measured universal search (13
systems, clean and 5% noise): beam width 1000, top 20, prune alpha 0.3, the
removal cap ``search_config.FILL_PRUNE_CAP`` (0.9) and the 1e-15 floor of the
prune weights.
"""
import hashlib
import itertools
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np

import epde.globals as global_var
from epde import _loop_stats

#: In-memory cache bounds (memory only: an evicted entry is recomputed identically).
CAND_MAX = 4096
BEAM_MAX = 2048
#: Free memory the Gram build leaves, besides its own arrays.
MIN_FREE_BYTES = 2 ** 30
#: The prune chain's weight floor (see the module docstring).
PRUNE_FLOOR = 1e-15
#: A beam extension whose new column lies in the span of the support (``den`` below this) is not valid.
SPAN_TOL = 1e-12
#: The fallback reasons that hold for the whole equation (the move cannot run for it at all), warned of once
#: per fit; the other reasons are states of one call.
_FALLBACK_TEXTS = {
    'target_not_in_library': "the target's term is not in the library (the library holds the terms the pool draws "
                             "with discrete parameters)",
    'target_column_unusable': "the target's column is zero or not finite on the evaluation rows",
    'no_candidates': 'no library term is admissible beside the target',
    'k_below_1': 'the equation has no room for a term beside its target',
    'beam_linalg': 'the least-squares beam met a singular or non-finite system',
}

_LIBRARY = {}          # (id(pool), repr(cfg)) -> (pool, [Term]); one entry
_STORES = {}           # slot (None for strong rows) -> Gram store
_WARNED = set()


def reset() -> None:
    """Empty the library, the Gram stores and the warned set: a new fit builds its own (see ``plain_fit.start_fit``)."""
    _LIBRARY.clear()
    _STORES.clear()
    _WARNED.clear()


class _Fallback(Exception):
    """An expected state: the drawn move runs instead."""


# ------------------------------------------------------------------ the library
def _factors_cfg(max_factors=None):
    if max_factors is not None:
        return max_factors
    from epde.interface.search_config import active_config
    return active_config().search_space.equation_factors_max_number


def _counts_and_fmax(cfg):
    if isinstance(cfg, dict):
        counts = [int(n) for n, p in zip(cfg['factors_num'], cfg['probas']) if p > 0]
        return counts, int(cfg['factors_num'][-1])
    return list(range(1, int(cfg) + 1)), int(cfg)


def atom_params(family):
    """Every parameter dict a random factor of ``family`` can take, in ``token_params`` order, or None when a
    parameter is a continuous range with no grid."""
    grids = {name: grid for name, grid in (getattr(family, 'param_grids', None) or {}).items() if grid is not None}
    per_axis = [grid for grid in grids.values() if isinstance(grid, dict)]
    axes = sorted(set.intersection(*(set(grid) for grid in per_axis))) if per_axis else [None]
    nondef = family.status['non_default_power']
    names = list(family.token_params)
    out = []
    for axis in axes:
        choices = []
        for name, bounds in family.token_params.items():
            grid = grids.get(name)
            if isinstance(grid, dict):
                grid = grid[axis]
            if name == 'dim' and axis is not None:
                choices.append([axis])
            elif grid is not None and (name != 'power' or nondef):
                choices.append(list(grid))
            elif name != 'power' or nondef:
                lo, hi = bounds
                if hi > lo:
                    if not isinstance(lo, int):
                        return None
                    choices.append(list(range(lo, hi + 1)))
                else:
                    choices.append([lo])
            else:
                choices.append([1])
        for combo in itertools.product(*choices):
            out.append(dict(zip(names, combo)))
    return out


def _token_atoms(pool):
    """{id(family): {token: [params, ...]}} for every family whose parameters are discrete.

    Each row's factor is built once here and checked to carry the row's parameters. Building a factor draws
    its hash from the global random state (``Factor.__init__``), so this check is also part of the random
    stream a seeded search reproduces."""
    all_vars = [family.variable for family in pool.families_demand_equation]
    atoms = {}
    for family in pool.families:
        params = atom_params(family)
        if params is None:
            continue
        fixed = getattr(family, 'token_fixed_params', None) or {}
        per_token = {}
        for tok in family.tokens:
            seen, rows = set(), []
            for pr in params:
                pr = dict(pr)
                pr.update({name: value for name, value in fixed.get(tok, {}).items() if name in pr})
                key = tuple(sorted(pr.items()))
                if key in seen:
                    continue
                seen.add(key)
                factor = family.create(label=tok, all_vars=all_vars, **pr)[1]
                built = {d['name']: float(factor.params[i]) for i, d in factor.params_description.items()}
                if any(abs(built[name] - float(value)) > 1e-12 for name, value in pr.items() if name in built):
                    raise RuntimeError('fill_prune: token %r does not take the parameters %r (got %r)'
                                       % (tok, pr, built))
                rows.append(pr)
            per_token[tok] = rows
        atoms[id(family)] = per_token
    return atoms


def _enumerate(pool, cfg) -> dict:
    """{factors_labels: Term} for every term the pool can draw under the factors setting ``cfg``."""
    from epde.structure.main_structures import Term
    counts, fmax = _counts_and_fmax(cfg)
    all_vars = [family.variable for family in pool.families_demand_equation]
    atoms = _token_atoms(pool)
    status0 = {}
    for family in pool.families:
        for tok in family.tokens:
            status0[tok] = min(fmax, family.token_params['power'][1])
    families = [family for family in pool.families if id(family) in atoms]
    terms = {}

    def rec(seq, used, n):
        if len(seq) == n:
            factors = [fam.create(label=tok, all_vars=all_vars, **pr)[1] for fam, tok, pr in seq]
            term = Term(pool, passed_term=factors, max_factors_in_term=cfg)
            term.descr_variable_marker = False      # as every non-target term (passed_term never sets it)
            terms.setdefault(term.factors_labels, term)
            return
        first = not seq
        for family in families:
            if first and not (family.status['meaningful'] or getattr(family, 'variable', None) in all_vars):
                continue
            for tok in family.tokens:
                if used.get(tok, 0) + 1 > status0[tok]:
                    continue
                if family.status['unique_token_type']:
                    occ = {t: family.token_params['power'][1] for t in family.tokens}
                elif family.status['unique_specific_token']:
                    occ = {tok: family.token_params['power'][1]}
                else:
                    occ = {tok: 1}
                nxt = dict(used)
                for t, v in occ.items():
                    nxt[t] = nxt.get(t, 0) + v
                for pr in atoms[id(family)][tok]:
                    rec(seq + [(family, tok, pr)], nxt, n)

    for n in counts:
        rec([], {}, n)
    return terms


def library_terms(pool, max_factors=None) -> list:
    """Every term of ``pool`` with discrete parameters, as Terms in enumeration order (families with a
    continuous ungridded parameter are left out). Kept for the last (pool, factors setting)."""
    cfg = _factors_cfg(max_factors)
    key = (id(pool), repr(cfg))
    hit = _LIBRARY.get(key)
    if hit is not None and hit[0] is pool:
        return list(hit[1])
    out = list(_enumerate(pool, cfg).values())
    _LIBRARY.clear()
    _LIBRARY[key] = (pool, out)
    return list(out)


def factors_setting(equation):
    try:
        return equation.metaparameters['max_factors_in_term']['value']
    except Exception:
        return None


def factor_cap(equation) -> int:
    """The most factors a term of ``equation`` can hold under its factors setting."""
    counts, _ = _counts_and_fmax(_factors_cfg(factors_setting(equation)))
    return max(counts)


# ------------------------------------------------------------------ the beam
class Gram(object):
    """A unit-diagonal weighted Gram G, its target column b and a relative residual (the target's norm is 1)."""

    def __init__(self, Graw, braw, yy):
        d = np.sqrt(np.diag(Graw))
        self.ok = d > 0
        d = np.where(d > 0, d, 1.0)
        self.G = Graw / d[:, None] / d[None, :]
        self.b = braw / d / np.sqrt(yy)
        self.M = self.G.shape[0]

    def fit(self, S):
        S = list(S)
        if not S:
            return np.zeros(0), 1.0
        c = np.linalg.solve(self.G[np.ix_(S, S)], self.b[S])
        return c, float(1.0 - self.b[S] @ c)

    def rss(self, S):
        return self.fit(S)[1]

    def gains_mask(self, S):
        """(gains, bad): the exact residual decrease of S+{z} for every z, and the z that are unusable or whose
        column lies in span(S)."""
        S = list(S)
        if not S:
            g = self.b ** 2
            g[~self.ok] = 0
            return g, ~self.ok
        GSS = self.G[np.ix_(S, S)]
        c = np.linalg.solve(GSS, self.b[S])
        A = np.linalg.solve(GSS, self.G[S, :])
        num = (self.b - self.G[:, S] @ c) ** 2
        den = 1.0 - np.einsum('ij,ij->j', self.G[S, :], A)
        bad = (den <= SPAN_TOL) | ~self.ok
        g = np.where(bad, 0.0, num / np.where(bad, 1.0, den))
        g[S] = 0.0
        return g, bad


def _gains_chunk(gm, Gt, Sa):
    """Gram.gains_mask for every parent row of ``Sa`` (C x k, k >= 1), stacked C x M, one matrix at a time."""
    G, b = gm.G, gm.b
    rows = np.arange(Sa.shape[0])[:, None]
    GSS = G[Sa[:, :, None], Sa[:, None, :]]
    c = np.linalg.solve(GSS, b[Sa][:, :, None])[:, :, 0]
    GSr = G[Sa]
    A = np.linalg.solve(GSS, GSr)
    num = (b[None, :] - np.matmul(Gt[Sa].transpose(0, 2, 1), c[:, :, None])[:, :, 0]) ** 2
    den = 1.0 - np.einsum('pij,pij->pj', GSr, A)
    bad = (den <= SPAN_TOL) | ~gm.ok[None, :]
    g = np.zeros_like(num)
    np.divide(num, den, out=g, where=~bad)
    g[rows, Sa] = 0.0
    return g, bad


def _refit(gm, R):
    """[Gram.rss(S) for S in the rows of R]: one stacked solve and dot."""
    G, b = gm.G, gm.b
    bR = b[R]
    c = np.linalg.solve(G[R[:, :, None], R[:, None, :]], bR[:, :, None])
    return (1.0 - np.matmul(bR[:, None, :], c)[:, 0, 0]).tolist()


def _chunk_rows(k, M) -> int:
    """Parents per chunk: about 96k doubles per k x M block stack, 8..256 (no value depends on it)."""
    return max(8, min(256, 98304 // max(1, k * M)))


def beam_search(gm, width, kmax) -> dict:
    """{k: [(support, rss)]}: level k holds the ``width`` lowest-residual distinct supports among all one-term
    extensions of level k-1 (extensions in span excluded), refitted exactly and sorted; a level with no valid
    extension ends the search."""
    M = gm.M
    level = [((), 1.0)]
    out = {}
    Gt = None
    for k in range(1, kmax + 1):
        P = len(level)
        if k == 1:
            g, bad = gm.gains_mask(())
            rss_all = 1.0 - g
            valid = ~bad
            Spar = np.zeros((1, 0), dtype=np.intp)
        else:
            if Gt is None:
                Gt = np.ascontiguousarray(gm.G.T)
            Spar = np.array([S for S, _ in level], dtype=np.intp)
            r0 = np.array([r for _, r in level], dtype=float)
            rss_all = np.empty((P, M))
            valid = np.empty((P, M), dtype=bool)
            ch = _chunk_rows(k - 1, M)
            for a in range(0, P, ch):
                g, bad = _gains_chunk(gm, Gt, Spar[a:a + ch])
                np.subtract(r0[a:a + ch, None], g, out=rss_all[a:a + ch])
                np.logical_not(bad, out=valid[a:a + ch])
            valid[np.arange(P)[:, None], Spar] = False
            rss_all = rss_all.reshape(-1)
            valid = valid.reshape(-1)
        idx = np.where(valid)[0]
        take = min(idx.size, width * (k + 1) * 2)
        top = idx[np.argpartition(rss_all[idx], take - 1)[:take]] if take < idx.size else idx
        top = top[np.argsort(rss_all[top])]
        if top.size:
            rows = np.concatenate([Spar[top // M], (top % M)[:, None]], axis=1)
            rows.sort(axis=1)
            if M ** k < 2 ** 62:                 # an injective int64 code of the sorted support
                code = rows @ (M ** np.arange(k - 1, -1, -1, dtype=np.int64))
                first = np.unique(code, return_index=True)[1]
            else:
                first = np.unique(rows, axis=0, return_index=True)[1]
            R = rows[np.sort(first)[:width]]
            nxt = sorted(zip([tuple(S) for S in R.tolist()], _refit(gm, R)), key=lambda t: t[1])
        else:
            nxt = []
        out[k] = nxt
        if not nxt:
            break
        level = nxt
    return out


# ------------------------------------------------------------------ draws and the prune chain
def draw_fill(top) -> int:
    """Index into ``top`` ([(support, rss)], best first), drawn with weights 1 / rss."""
    w = 1.0 / np.maximum(np.array([r for _, r in top], dtype=float), PRUNE_FLOOR)
    return int(np.random.choice(len(top), p=w / w.sum()))


def prune_chain(gm, S, alpha) -> list:
    """The stages [S, ..., two terms] of the probabilistic prune (module docstring, 5)."""
    from epde.interface import search_config
    S = list(S)
    stages = [tuple(S)]
    while len(S) > 2:
        base = gm.rss(S)
        loss = np.array([max(gm.rss(S[:i] + S[i + 1:]) - base, PRUNE_FLOOR) for i in range(len(S))])
        w = 1.0 / loss
        w = w / w.sum()
        p = np.minimum(search_config.FILL_PRUNE_CAP, alpha * w * len(S))
        drop = set(np.where(np.random.random(len(S)) < p)[0].tolist())
        if not drop:
            drop = {int(np.random.choice(len(S), p=w))}
        if len(S) - len(drop) < 2:
            drop = set(sorted(drop, key=lambda i: -w[i])[:len(S) - 2])
        S = [t for i, t in enumerate(S) if i not in drop]
        stages.append(tuple(S))
    return stages


# ------------------------------------------------------------------ the Gram store
def _free_ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _warn_once(what, message) -> None:
    if what not in _WARNED:
        _WARNED.add(what)
        warnings.warn(message, global_var.EPDEUsageWarning, stacklevel=3)


def _weak() -> bool:
    return bool(getattr(global_var.samples_manager, 'weak_mode', False))


def _trajectory_key():
    manager = global_var.samples_manager
    keys = list(manager.weak.keys) if _weak() else list(manager.inner_shapes.keys())
    if len(keys) != 1:
        _warn_once('trajectories', 'plain mode fill_probability: the fill-and-prune move handles one trajectory; '
                                   'with %d it is skipped (the drawn move runs).' % len(keys))
        raise _Fallback('trajectories')
    return keys[0]


def _column(term, key, slot):
    """``term``'s column on the evaluation rows (weak rows: those of ``slot``)."""
    if slot is not None:
        from epde.structure.weak_rows import term_columns
        return np.asarray(term_columns(term, slot)[key], dtype=float).reshape(-1)
    from epde.structure.structure_template import ComplexStructure
    manager = global_var.samples_manager
    if term.saved[False] or (term.factors_labels, False) in manager:
        values = term.evaluate()
    else:
        values = ComplexStructure.evaluate(term)
    return np.asarray(values[key], dtype=float).reshape(-1)


def gram_store(equation, key, sw, slot) -> dict:
    """The Gram store of ``slot`` (None: strong rows, one store), built at its first use in a fit and rebuilt
    only when the samples manager, the trajectory, the token families or the factors setting change."""
    families = list(equation.pool.families)
    skey = (id(global_var.samples_manager), key, tuple(id(f) for f in families),
            repr(factors_setting(equation)), slot)
    gs = _STORES.get(slot)
    if gs is not None and gs['skey'] == skey and all(a is b for a, b in zip(gs['families'], families)):
        return gs
    for other in [s for s, g in _STORES.items() if s == slot or g['skey'][0] != skey[0]]:
        del _STORES[other]
    lib = library_terms(equation.pool, factors_setting(equation))
    n, M = sw.size, len(lib)
    free = _free_ram_bytes()
    if free is not None and free < n * M * 8 + M * M * 16 + MIN_FREE_BYTES:
        _warn_once(('memory', slot), 'plain mode fill_probability: the Gram of %d rows x %d library terms does '
                                     'not fit in free memory; the fill-and-prune move is skipped.' % (n, M))
        raise _Fallback('memory')
    ok = np.ones(M, dtype=bool)
    L = np.empty((n, M), order='F')
    for i, term in enumerate(lib):
        col = _column(term, key, slot)
        if col.size != n:
            raise RuntimeError('fill_prune: the column of %s has %d rows, the weights %d' % (term.name, col.size, n))
        if not np.all(np.isfinite(col)):
            L[:, i] = 0.0
            ok[i] = False
        else:
            L[:, i] = col
    Graw = np.empty((M, M))
    for a in range(0, M, 128):
        Graw[a:a + 128, :] = (L[:, a:a + 128] * sw[:, None]).T @ L
    del L
    Graw = 0.5 * (Graw + Graw.T)
    d = np.sqrt(np.clip(np.diag(Graw), 0.0, None))
    ok &= d > 0
    dd = np.where(ok, d, 1.0)
    Gn = Graw / dd[:, None] / dd[None, :]
    del Graw
    Gn[~ok, :] = 0.0
    Gn[:, ~ok] = 0.0
    np.fill_diagonal(Gn, 1.0)
    labels = tuple(t.factors_labels for t in lib)
    gs = dict(skey=skey, families=families, lib=lib, labels=labels, index={lab: i for i, lab in enumerate(labels)},
              Gn=Gn, ok=ok, related={}, cand=OrderedDict(), beams=OrderedDict())
    _STORES[slot] = gs
    return gs


def _related_mask(gs, target) -> np.ndarray:
    from epde.operators.common.right_part_selection import _related_to_target
    lab = target.factors_labels
    mask = gs['related'].get(lab)
    if mask is None:
        mask = gs['related'][lab] = np.array([bool(_related_to_target(term, target)) for term in gs['lib']],
                                             dtype=bool)
    return mask


def candidates(equation, target, partner_labels, gs, t):
    """(candidate library indices ascending, their signature), module docstring 1."""
    genes = getattr(equation, '_plain_genes', None) or {}
    nudged = target.factors_labels != genes.get('target')
    ckey = (equation.main_var_to_explain, target.factors_labels, frozenset(partner_labels),
            genes.get('target') if nudged else None)
    cache = gs['cand']
    hit = cache.get(ckey)
    if hit is not None:
        cache.move_to_end(ckey)
        return hit
    excluded = {target.factors_labels} | set(partner_labels)
    if nudged:
        excluded.add(genes.get('target'))
    mask = gs['ok'] & ~_related_mask(gs, target)
    mask &= np.array([lab not in excluded for lab in gs['labels']], dtype=bool)
    cand = np.flatnonzero(mask).astype(np.int32)
    sig = hashlib.blake2b(cand.tobytes() + int(t).to_bytes(4, 'little'), digest_size=16).hexdigest()
    cache[ckey] = (cand, sig)
    while len(cache) > CAND_MAX:
        cache.popitem(last=False)
    return cand, sig


def _beam_levels(gs, var, target_label, cand, sig, t, K, width, top, levels_mode):
    """('ok', {k: first ``top`` supports of level k (library indices)}) or ('fail', reason), one beam search per
    key. 'all' keeps the levels 1..K the beam reached; 'max' keeps level K."""
    bkey = (var, target_label, sig, K, width, top, levels_mode)
    cache = gs['beams']
    hit = cache.get(bkey)
    if hit is not None:
        cache.move_to_end(bkey)
        return hit
    try:
        Gn = gs['Gn']
        gm = Gram(Gn[np.ix_(cand, cand)], Gn[cand, t], 1.0)
        levels = beam_search(gm, width, K)
        if levels_mode == 'max':
            level = levels.get(K, [])
            if not level:
                hit = ('fail', 'beam_empty')
            elif not all(np.isfinite(r) for _, r in level):
                hit = ('fail', 'beam_linalg')
            else:
                hit = ('ok', {K: [(tuple(int(cand[j]) for j in S), float(r)) for S, r in level[:top]]})
        else:
            out = {}
            for k in range(1, K + 1):
                level = levels.get(k, [])
                if not level:
                    break
                if not all(np.isfinite(r) for _, r in level):
                    out = None
                    break
                out[k] = [(tuple(int(cand[j]) for j in S), float(r)) for S, r in level[:top]]
            hit = ('fail', 'beam_linalg') if out is None else (('ok', out) if out else ('fail', 'beam_empty'))
    except np.linalg.LinAlgError:
        hit = ('fail', 'beam_linalg')
    cache[bkey] = hit
    while len(cache) > BEAM_MAX:
        cache.popitem(last=False)
    return hit


# ------------------------------------------------------------------ the move
def partner_targets(system, var) -> set:
    """factors_labels of the targets of ``system``'s equations other than ``var``'s (the fitted target, else
    the term the target gene names)."""
    from epde.operators.common.plain_fit import gene_term
    out = set()
    for key in system.vars_to_describe:
        if key == var:
            continue
        eq = system.vals[key]
        target = eq.target if eq.target is not None else gene_term(eq)
        if target is not None:
            out.add(target.factors_labels)
    return out


def _install(equation, target, lib_terms) -> None:
    from epde.operators.common.plain_fit import gene_term
    new_terms = []
    for term in lib_terms:
        new = deepcopy(term)
        new.pool = equation.pool
        new.resetSavedState()
        new_terms.append(new)
    old = list(equation.structure)
    equation.structure = [target] + new_terms
    equation._invalidate_label_cache()
    problems = []
    if len({t.factors_labels for t in equation.structure}) != len(equation.structure):
        problems.append('duplicate labels')
    if gene_term(equation) is not target:
        problems.append('the gene no longer names the target')
    if len(equation.structure) > int(equation.metaparameters['max_terms_number']['value']):
        problems.append('over max_terms_number')
    if problems:
        equation.structure = old
        equation._invalidate_label_cache()
        raise RuntimeError('fill_prune install: %s' % '; '.join(problems))


def _attempt(equation, target, features, system, params) -> None:
    from epde.interface.search_config import active_config, plain_kept_terms_cap
    if target is None:
        raise _Fallback('no_gene_target')
    var = equation.main_var_to_explain
    if system is None or system.vals[var] is not equation:
        raise _Fallback('no_system')
    partner_labels = frozenset(partner_targets(system, var))
    if target.factors_labels in partner_labels:
        raise _Fallback('target_conflict')
    key = _trajectory_key()
    if _weak():
        slot = var
        sw = np.asarray(global_var.samples_manager.row_weights(slot, 'dm')[key], dtype=float).reshape(-1)
    else:
        slot = None
        sw = np.asarray(global_var.samples_manager.gFunc('dm')[key], dtype=float).reshape(-1)
    gs = gram_store(equation, key, sw, slot)
    t = gs['index'].get(target.factors_labels)
    if t is None:
        raise _Fallback('target_not_in_library')
    if not gs['ok'][t]:
        raise _Fallback('target_column_unusable')
    cand, sig = candidates(equation, target, partner_labels, gs, t)
    if cand.size == 0:
        raise _Fallback('no_candidates')
    cap = plain_kept_terms_cap(active_config().search_space.equation_terms_max_number)
    K = min(cap, int(equation.metaparameters['max_terms_number']['value']) - 1, int(cand.size))
    if K < 1:
        raise _Fallback('k_below_1')
    levels_mode = params['fill_levels']
    level = int(np.random.randint(1, K + 1)) if levels_mode == 'all' else K
    hit = _beam_levels(gs, var, target.factors_labels, cand, sig, t, K, params['fill_beam_width'],
                       params['fill_top'], levels_mode)
    if hit[0] == 'fail':
        raise _Fallback(hit[1])
    top = hit[1].get(level)
    if not top:
        raise _Fallback('beam_empty')
    fill = top[draw_fill(top)][0]
    idx = list(fill)
    stages = prune_chain(Gram(gs['Gn'][np.ix_(idx, idx)], gs['Gn'][idx, t], 1.0), list(range(len(fill))),
                         params['fill_prune_alpha'])
    stage = stages[np.random.randint(len(stages))]
    lib_idx = [fill[p] for p in stage]
    if frozenset(gs['labels'][i] for i in lib_idx) == frozenset(f.factors_labels for f in features):
        raise _Fallback('same_structure')
    _install(equation, target, [gs['lib'][i] for i in lib_idx])


def fill(equation, target, features, system, params) -> bool:
    """The fill-and-prune move on ``equation`` with probability ``params['fill_probability']``: True when a
    new structure was installed (module docstring), False when the drawn move should run."""
    p = params['fill_probability']
    if p <= 0 or np.random.random() >= p:
        return False
    try:
        _attempt(equation, target, features, system, params)
    except _Fallback as exc:
        reason = str(exc)
        _loop_stats.record('PlainMutation.fill_fallback.' + reason, 1, 1)
        if reason in _FALLBACK_TEXTS:
            _warn_once(('fallback', equation.main_var_to_explain, reason),
                       'plain mode fill_probability: the fill-and-prune move is skipped for %r (the drawn move '
                       'runs instead): %s.' % (equation.main_var_to_explain, _FALLBACK_TEXTS[reason]))
        return False
    return True
