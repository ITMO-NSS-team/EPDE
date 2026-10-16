"""Baseline outside EPDE: sparse regression on a fixed library (SINDy / PDE-FIND).

The library is built from the same building blocks as the EPDE pool of the
run's config, and every column is named exactly as EPDE prints the term, so
the structural metric is shared:

* powers of every variable up to ``data_fun_pow`` (``u{power: 2.0}``);
* every derivative of every variable up to ``max_deriv_order`` per axis,
  in powers up to ``deriv_fun_pow`` (``du/dx1{power: 1.0}``);
* coordinates (``x{power: 1.0, dim: 1.0}``) and sin/cos at the fixed
  frequency, when the config's pool has them;
* the problem's extra fields (forcing, exact terms, couplings);
* products of two building blocks from different families -- EPDE's
  ``factors_num: [1, 2]``.

By default the fields and derivatives use EPDE's configured preprocessing,
including supplied derivative stacks. Independent PySINDy differentiation
must be selected explicitly and is recorded in the result metadata.
Multiplicative temporal left-hand sides and spatial constraints are outside
this baseline's representation; known such problems raise UnsupportedProblem.

Nothing is chosen with knowledge of the truth. For every variable each time
derivative order is tried as the left-hand side (``u_t`` vs ``u_tt``) and
STLSQ thresholds are swept, with and without column normalisation (with it
the weakest true term of a wide library can drop out before spurious ones;
without it the threshold acts on raw coefficients). Every model is scored on a held-out tail (last
20 % of the interior time levels) by its mean squared error relative to the
target's variance; the SPARSEST model within ``ABS_TOL + REL_TOL * best`` of
the best error wins (a one-tolerance rule: an information criterion keeps
adding terms with 1e-6 coefficients that only absorb differentiation error on
clean data). Data are
trimmed by the same boundary as the EPDE run, and long records are row-
subsampled (seeded) to keep the regression in memory.
"""

import time
from itertools import combinations

import numpy as np

from .config import resolve_for_problem
from .library import _blocks, _interior

VAL_FRACTION = 0.2
MAX_TRAIN_ROWS = 60000
MAX_VAL_ROWS = 20000
#: cap on rows x columns of the design matrix (2e7 doubles = 160 MB); wide
#: libraries (systems, 2-D PDEs) get proportionally fewer rows
MAX_CELLS = 2e7
#: model selection tolerance on the held-out error (fraction of the variance)
ABS_TOL = 1e-3
REL_TOL = 0.05


def unsupported_reason(problem):
    """Known equation forms outside this evolution-only baseline's library.

    Accept a Problem or registry name so campaign metadata can classify an
    unsupported cell without loading its data. This is a representation check;
    neither the truth strings nor a regression result determine support.
    """
    name = getattr(problem, 'name', problem)
    return {
        'pde_divide': 'the baseline cannot represent x*u_t on the left-hand side '
                      'and has no inverse-x coefficient token',
        'ns': 'the baseline cannot represent the spatial continuity equation; '
              'one temporal equation per variable would substitute p_t for continuity',
    }.get(name)


class UnsupportedProblem(ValueError):
    """The baseline cannot represent this problem's equation form."""

    def __init__(self, problem, reason):
        self.problem = getattr(problem, 'name', problem)
        self.reason = reason
        super().__init__(f'{self.problem}: PySINDy baseline unsupported: {reason}')


def _columns(blocks, lhs_order):
    """Single blocks and products of two blocks of different families; a
    term must hold at least one meaningful factor (as in EPDE). Time
    derivatives of order >= the left-hand side's (of ANY variable) are left
    out, as in SINDy/PDE-FIND: u_t = f(u, u_x, ...), u_tt = f(u, u_t, ...)."""
    usable = [b for b in blocks if b[4] < lhs_order]
    cols = [(b[0], b[1]) for b in usable if b[3]]
    for a, b in combinations(usable, 2):
        if a[2] == b[2] or not (a[3] or b[3]):
            continue
        cols.append((f'{a[0]} * {b[0]}', a[1] * b[1]))
    return cols


def _fit(X_tr, y_tr, X_va, y_va, threshold, normalize):
    import pysindy as ps
    opt = ps.STLSQ(threshold=threshold, alpha=1e-6, normalize_columns=normalize, max_iter=30)
    opt.fit(X_tr, y_tr)
    coef = np.ravel(opt.coef_)
    k = int(np.count_nonzero(coef))
    if k == 0:
        return None
    resid = y_va - X_va @ coef
    return {'coef': coef, 'k': k, 'mse': float(np.mean(resid ** 2))}


def _n_columns(blocks) -> int:
    usable = [(fam, meaningful) for _, _, fam, meaningful, order in blocks if order < 1]
    n = sum(1 for _, m in usable if m)
    n += sum(1 for (fa, ma), (fb, mb) in combinations(usable, 2) if fa != fb and (ma or mb))
    return n


def _rows(problem, search, seed, n_columns=1):
    """Flat indices (into the trimmed interior) of the training and the
    held-out rows: split in time, then subsampled if too many."""
    inner = _interior(problem, search)
    n_t = problem.shape[0]
    t_idx = np.arange(n_t)[inner[0]]
    split = t_idx[int(len(t_idx) * (1 - VAL_FRACTION))]
    time_of_row = np.broadcast_to(
        np.arange(n_t).reshape((-1,) + (1,) * (len(problem.shape) - 1)),
        problem.shape)[inner].ravel()
    rng = np.random.default_rng(seed)
    train = np.flatnonzero(time_of_row < split)
    val = np.flatnonzero(time_of_row >= split)
    shrink = min(1.0, MAX_CELLS / ((MAX_TRAIN_ROWS + MAX_VAL_ROWS) * max(n_columns, 1)))
    max_train, max_val = int(MAX_TRAIN_ROWS * shrink), int(MAX_VAL_ROWS * shrink)
    if train.size > max_train:
        train = np.sort(rng.choice(train, max_train, replace=False))
    if val.size > max_val:
        val = np.sort(rng.choice(val, max_val, replace=False))
    return inner, train, val


def run_pysindy(problem, data, cfg, seed: int = 0):
    """Returns ``(front, objectives, fit_seconds, info)`` like an EPDE run:
    a one-member front. Raises UnsupportedProblem for equation forms outside
    the library before preparing derivatives or fitting any model."""
    reason = unsupported_reason(problem)
    if reason:
        raise UnsupportedProblem(problem, reason)
    opts = dict(cfg.get('pysindy', {}))
    method = opts.get('differentiation', 'epde')
    if method not in ('epde', 'finite_difference', 'smoothed_finite_difference'):
        raise ValueError(f'Unknown baseline differentiation method: {method!r}')
    if method != 'epde' and problem.derivs is not None:
        raise ValueError('independent baseline differentiation cannot discard supplied derivatives; '
                         "use pysindy.differentiation='epde'")
    search = resolve_for_problem(cfg, problem)
    t0 = time.perf_counter()
    blocks, lhs_options = _blocks(problem, data, search, method=method)
    inner, train, val = _rows(problem, search, seed, _n_columns(blocks))
    rows = np.concatenate([train, val])
    n_train = train.size

    def reduce(array):                       # full field -> the selected rows only
        return np.asarray(array)[inner].ravel()[rows]

    blocks = [(name, reduce(arr), fam, meaningful, order)
              for name, arr, fam, meaningful, order in blocks]
    lhs_options = {var: [(name, reduce(arr), k) for name, arr, k in opts_]
                   for var, opts_ in lhs_options.items()}
    max_terms = int(search.get('search_space', {}).get('equation_terms_max_number', 10))
    thresholds = np.logspace(-4, 2, 49)

    equations = []
    info = {'lhs': {}, 'threshold': {}, 'library_size': {}, 'train_rows': int(n_train),
            'val_rows': int(val.size), 'differentiation': method}
    total_mse, total_terms = 0.0, 0
    for var, options in lhs_options.items():
        candidates = []
        for lhs_name, y, lhs_order in options:
            cols = _columns(blocks, lhs_order)
            names = [c[0] for c in cols]
            X = np.stack([c[1] for c in cols], axis=1)
            scale = float(np.std(y)) or 1.0
            X_tr, y_tr, X_va, y_va = X[:n_train], y[:n_train] / scale, X[n_train:], y[n_train:] / scale
            for normalize in (True, False):
                for thr in thresholds:
                    res = _fit(X_tr, y_tr, X_va, y_va, thr, normalize)
                    if res is None:
                        continue
                    candidates.append(dict(res, lhs=lhs_name, names=names, scale=scale,
                                           threshold=float(thr), normalize=normalize))
            info['library_size'][var] = len(names)
        if not candidates:
            raise RuntimeError(f'STLSQ zeroed every coefficient for {var} at all thresholds')
        within = [c for c in candidates if c['k'] <= max_terms] or candidates
        floor = min(c['mse'] for c in within)
        good = [c for c in within if c['mse'] <= floor + ABS_TOL + REL_TOL * floor]
        best = min(good, key=lambda c: (c['k'], c['mse']))
        coef = best['coef'] * best['scale']
        terms = [f'{float(c)!r} * {n}' for c, n in zip(coef, best['names']) if c != 0]
        equations.append(' + '.join(terms) + f' = {best["lhs"]}')
        info['lhs'][var], info['threshold'][var] = best['lhs'], best['threshold']
        info.setdefault('normalize_columns', {})[var] = best['normalize']
        total_mse += best['mse']
        total_terms += best['k']
    fit_seconds = time.perf_counter() - t0
    return [equations], [[total_mse, float(total_terms)]], fit_seconds, info
