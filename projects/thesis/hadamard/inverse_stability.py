"""Inverse-map continuous dependence: is ``data -> equation`` Hadamard-stable?

PDE identification is a classically ill-posed inverse problem: tiny changes in
the measured field can swing the recovered coefficients. Hadamard's third
condition for the *inverse* map ``T : data -> equation`` is continuous
dependence -- small ``||delta data||`` should give small ``||delta theta||``.
We quantify it by perturb-and-observe on the data, referenced to the
*unperturbed* fit (self-consistency, never the seeded truth):

1. ``condition_number`` -- of the (column-equilibrated, weighted) library
   matrix. For a fixed support this is the local Lipschitz constant of the
   coefficient map: ``||delta theta|| / ||delta data|| <= cond``. Large cond =
   collinear library = the data barely constrains the coefficients.

2. ``wls_metrics`` -- the analytic coefficient covariance
   ``Cov = sigma^2 (Theta^T W Theta)^{-1}`` for *all* active coefficients, hence
   the relative standard error ``SE_i / |theta_i|``. (This generalises the
   ``Var(gamma_0)`` diagonal that ``stability.VaryingCoefSetup`` already
   computes for the constant block.)

3. ``mc_ensemble`` -- Monte-Carlo over independent NOISE realisations: empirical
   coefficient covariance + per-coefficient CV, and the ratio
   ``R = Var_empirical / Var_analytic``. The June-2026 audit found
   ``R ~ 0.01-0.23`` (see ``projects/thesis/stability_audit.md``): the analytic
   covariance over-states the seed-to-seed spread 5-100x because the dominant
   perturbation on real data is *deterministic discretisation* error, not
   stochastic noise. That is precisely why family (4) includes a differentiation
   perturbation.

4. ``perturbation_response`` -- relative coefficient change ``||delta theta|| /
   ||theta||`` versus perturbation magnitude, over three data-level families:
   additive noise, grid decimation, and Gaussian pre-smoothing (a
   differentiation-scheme proxy). The log-log slope is the local Lipschitz
   exponent: ``~1`` => well-posed inverse, ``>1`` / blow-up => ill-posed.

The cheap path reuses ``kdv_sindy_test.build_pool_only`` + the seeded-equation
feature evaluation that ``noise_stability_sweep`` already drives -- the support
is fixed (the truth structure), so coefficient vectors are column-aligned
across perturbed datasets and can be compared element-wise.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_THESIS = os.path.abspath(os.path.join(_THIS, '..'))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..', '..'))
for _p in (_ROOT, _THESIS, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402
import epde.globals as gv  # noqa: E402
from epde.interface.equation_translator import translate_equation  # noqa: E402
from kdv_sindy_test import build_pool_only, _normalize_grid_labels  # noqa: E402
from thesis_runner import load_config, pipeline_settings, _set_seeds  # noqa: E402

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


# --------------------------------------------------------------------------
# Data-level perturbation wrappers (companions to noise_stability._patched_load)
# --------------------------------------------------------------------------
def _noise_load(orig, noise_level, seed):
    """Wrap ``load_data`` to add iid Gaussian noise (sigma = nl*0.01*std)."""
    def _load():
        coords, data, vars_, dim = orig()
        rng = np.random.default_rng(int(seed))

        def _n(a):
            a = np.asarray(a, dtype=np.float64)
            sigma = noise_level * 0.01 * float(np.std(a))
            return a + rng.normal(0.0, sigma, a.shape) if sigma > 0 else a

        if isinstance(data, np.ndarray):
            nd = _n(data)
        elif isinstance(data, (list, tuple)):
            nd = type(data)(_n(a) for a in data)
        else:
            nd = data
        return coords, nd, vars_, dim
    return _load


def _decimate_load(orig, stride):
    """Wrap ``load_data`` to coarsen the grid by ``stride`` on every axis."""
    def _load():
        coords, data, vars_, dim = orig()
        n_axes = 1 if dim == 0 else dim + 1
        slc = tuple(slice(None, None, int(stride)) for _ in range(n_axes))

        def _d(a):
            return np.asarray(a)[slc]

        coords2 = tuple(_d(c) for c in coords)
        if isinstance(data, (list, tuple)):
            data2 = type(data)(_d(a) for a in data)
        else:
            data2 = _d(data)
        return coords2, data2, vars_, dim
    return _load


def _smooth_load(orig, sigma):
    """Wrap ``load_data`` to Gaussian pre-smooth the field (diff-scheme proxy).

    Spatial axes only for PDEs (axis 0 = time is left intact, matching the
    repo's smoothing convention); the single time axis is smoothed for ODEs.
    More smoothing trades derivative variance for bias -- the deterministic
    differentiation perturbation the audit identified as dominant.
    """
    def _load():
        coords, data, vars_, dim = orig()
        if gaussian_filter is None or sigma <= 0:
            return coords, data, vars_, dim

        def _s(a):
            a = np.asarray(a, dtype=np.float64)
            if dim == 0:
                sig = [sigma] * a.ndim
            else:
                sig = [0.0] + [sigma] * (a.ndim - 1)
            return gaussian_filter(a, sigma=sig, mode='nearest')

        if isinstance(data, (list, tuple)):
            data2 = type(data)(_s(a) for a in data)
        else:
            data2 = _s(data)
        return coords, data2, vars_, dim
    return _load


_FAMILY_LOADERS = {
    'noise': _noise_load,        # (orig, level, seed)
    'resolution': _decimate_load,  # (orig, level)
    'diffscheme': _smooth_load,    # (orig, level)
}


# --------------------------------------------------------------------------
# Feature extraction (seeded support, perturbed data) -- the cheap path.
# --------------------------------------------------------------------------
def _seeded_features(system, perturb=None, seed=0):
    """Build the pool on (perturbed) data and return per-variable
    ``(features, target, weights, term_names)`` for the seeded truth support.

    ``perturb`` is an optional callable ``orig_load -> wrapped_load``. The
    support (truth structure) is identical across calls, so the returned
    coefficient vectors are column-aligned for element-wise comparison.
    """
    cfg = load_config(system)
    if perturb is not None:
        cfg.load_data = perturb(cfg.load_data)
    _set_seeds(seed)
    gv.set_gram_config('vcoef')

    search = build_pool_only(cfg, pipeline_settings('new'))
    sw = np.asarray(gv.grid_cache.g_func[gv.grid_cache.g_func_mask]).reshape(-1)
    _, _, variable_names, _ = cfg.load_data()
    all_vars = list(variable_names)

    truth = yaml.safe_load(open(os.path.join(_THESIS, 'configs',
                                             f'{system}.yaml')))
    teqs = truth.get('truth_equations') or []
    seeded = (teqs[0] if len(all_vars) == 1
              else {v: teqs[i] for i, v in enumerate(all_vars)})
    seeded = _normalize_grid_labels(seeded)
    tsoeq = translate_equation(seeded, search.pool, all_vars=all_vars)

    out = {}
    for v in all_vars:
        eq = tsoeq.vals[v]
        eq.main_var_to_explain = v
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        _, t, f = eq.evaluate(normalize=False, return_val=False)
        f = np.asarray(f, dtype=float)
        if f.ndim == 1:
            f = f[:, None]
        t = np.asarray(t, dtype=float).reshape(-1)
        terms = [term.name for i, term in enumerate(eq.structure)
                 if i != eq.target_idx]
        out[v] = {'f': f, 't': t, 'w': sw, 'terms': terms}
    return out


def _build_perturb(family, level, seed=0):
    """Return a ``load_data`` wrapper for one (family, level) draw."""
    if family == 'noise':
        return lambda orig: _noise_load(orig, level, seed)
    if family == 'resolution':
        return lambda orig: _decimate_load(orig, level)
    if family == 'diffscheme':
        return lambda orig: _smooth_load(orig, level)
    raise ValueError(f"unknown perturbation family {family!r}")


# --------------------------------------------------------------------------
# Core linear-algebra: condition number, WLS fit, covariance.
# --------------------------------------------------------------------------
def wls_metrics(f, t, w):
    """Weighted-least-squares fit + conditioning + analytic covariance.

    Returns a dict with the coefficient vector ``theta``, its covariance
    ``cov = sigma^2 (A^T A)^{-1}`` (A = weighted design), relative standard
    errors ``rel_se = SE/|theta|``, the column-equilibrated condition number
    ``cond`` (collinearity, scale-free -- the genuine local Lipschitz constant
    modulo units), the raw condition number ``cond_raw``, and ``sigma2``.
    """
    f = np.asarray(f, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    sw = np.sqrt(np.clip(np.asarray(w, dtype=float).reshape(-1), 0, None))
    A = f * sw[:, None]
    b = t * sw

    col = np.linalg.norm(A, axis=0)
    col_safe = np.where(col > 0, col, 1.0)
    A_eq = A / col_safe
    cond = float(np.linalg.cond(A_eq))
    cond_raw = float(np.linalg.cond(A))

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ theta - b
    n, p = A.shape
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    G = A.T @ A
    Ginv = np.linalg.pinv(G)
    cov = sigma2 * Ginv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    rel_se = se / np.maximum(np.abs(theta), 1e-30)
    return {'theta': theta, 'cov': cov, 'se': se, 'rel_se': rel_se,
            'cond': cond, 'cond_raw': cond_raw, 'sigma2': sigma2}


def condition_number(f, w):
    """Column-equilibrated weighted condition number (collinearity only)."""
    return wls_metrics(f, np.zeros(np.asarray(f).shape[0]), w)['cond']


# --------------------------------------------------------------------------
# (3) Monte-Carlo noise ensemble + R = Var_emp / Var_analytic.
# --------------------------------------------------------------------------
def mc_ensemble(system, noise_level=2.0, M=6, base_seed=0):
    """Empirical coefficient variability over ``M`` independent noise draws.

    For each variable returns: empirical CV per coefficient, and
    ``R = Var_empirical / Var_analytic`` (median over coefficients). ``R << 1``
    reproduces the audit's finding that deterministic discretisation -- not the
    injected noise -- dominates the actual coefficient spread.
    """
    thetas = {}
    analytic = {}
    for m in range(M):
        perturb = _build_perturb('noise', noise_level, seed=base_seed + m)
        feats = _seeded_features(system, perturb=perturb, seed=base_seed + m)
        for v, d in feats.items():
            met = wls_metrics(d['f'], d['t'], d['w'])
            thetas.setdefault(v, []).append(met['theta'])
            analytic.setdefault(v, []).append(np.diag(met['cov']))

    out = {}
    for v in thetas:
        TH = np.array(thetas[v])               # (M, p)
        AN = np.array(analytic[v])             # (M, p)
        emp_mean = TH.mean(axis=0)
        emp_var = TH.var(axis=0, ddof=1)
        emp_cv = np.sqrt(emp_var) / np.maximum(np.abs(emp_mean), 1e-30)
        analytic_var = AN.mean(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.where(analytic_var > 0, emp_var / analytic_var, np.nan)
        out[v] = {
            'M': int(M), 'noise_level': float(noise_level),
            'emp_cv_max': float(np.max(emp_cv)),
            'emp_cv_median': float(np.median(emp_cv)),
            'R_median': float(np.nanmedian(R)),
            'terms': feats[v]['terms'],
        }
    return out


# --------------------------------------------------------------------------
# (4) Perturbation response -> local Lipschitz slope.
# --------------------------------------------------------------------------
_REF_LEVEL = {'noise': 0.0, 'resolution': 1, 'diffscheme': 0.0}


def _coeffs(system, family, level, seed):
    """Coefficient vector(s) for one (family, level, seed) draw.

    At the family's reference level (no noise / stride 1 / zero smoothing) the
    data is left untouched; any other level applies the perturbation.
    """
    perturb = (None if level == _REF_LEVEL[family]
               else _build_perturb(family, level, seed))
    feats = _seeded_features(system, perturb=perturb, seed=seed)
    return {v: wls_metrics(d['f'], d['t'], d['w'])['theta']
            for v, d in feats.items()}


def perturbation_response(system, family, levels, seeds=(0,)):
    """Relative coefficient change vs perturbation magnitude + Lipschitz slope.

    The reference is the *unperturbed* fit (noise level 0 / stride 1 / sigma 0).
    Random families (noise) average ``||delta theta|| / ||theta_ref||`` over
    ``seeds``; deterministic families (resolution, diffscheme) use one draw.
    Returns per-variable ``{levels, rel_err, lipschitz_slope}`` where the slope
    is the log-log fit (``~1`` Lipschitz/well-posed; ``>1`` super-linear).

    Note on resolution: decimation changes the sample count, so the reference
    is re-fit at each stride's own grid only for coefficient *values* (the
    support is identical); ``rel_err`` is taken against the finest grid.
    """
    ref = _coeffs(system, family, _REF_LEVEL[family], seed=seeds[0])

    per_var = {v: {'levels': [], 'rel_err': []} for v in ref}
    for lev in levels:
        # average over seeds for the random family
        acc = {v: [] for v in ref}
        sd = seeds if family == 'noise' else (seeds[0],)
        for s in sd:
            th = _coeffs(system, family, lev, seed=s)
            for v in ref:
                if v not in th:
                    continue
                t0 = ref[v]
                t1 = th[v]
                # guard against decimation changing p (shouldn't: fixed support)
                p = min(len(t0), len(t1))
                denom = np.linalg.norm(t0[:p]) or 1.0
                acc[v].append(float(np.linalg.norm(t1[:p] - t0[:p]) / denom))
        for v in ref:
            if acc[v]:
                per_var[v]['levels'].append(float(lev))
                per_var[v]['rel_err'].append(float(np.mean(acc[v])))

    for v in per_var:
        L = np.array(per_var[v]['levels'], dtype=float)
        E = np.array(per_var[v]['rel_err'], dtype=float)
        slope = np.nan
        good = (L > 0) & (E > 0)
        if np.count_nonzero(good) >= 2:
            slope = float(np.polyfit(np.log(L[good]), np.log(E[good]), 1)[0])
        per_var[v]['lipschitz_slope'] = slope
    return per_var


# --------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------
DEFAULT_LEVELS = {
    'noise': [0.5, 1.0, 2.0, 4.0],
    'resolution': [2, 3, 4],
    'diffscheme': [0.5, 1.0, 2.0],
}


def analyze_inverse(system, *, families=('noise', 'resolution', 'diffscheme'),
                    levels=None, mc_M=6, mc_noise=2.0, response_seeds=(0, 1)):
    """Full inverse continuous-dependence battery for one system.

    Returns a JSON-friendly dict per variable: baseline conditioning + relative
    SE, the noise MC summary (CV, R), and the Lipschitz slope per perturbation
    family.
    """
    levels = levels or DEFAULT_LEVELS

    # Baseline (unperturbed) conditioning + analytic covariance.
    base = _seeded_features(system, perturb=None, seed=0)
    out = {}
    for v, d in base.items():
        met = wls_metrics(d['f'], d['t'], d['w'])
        out[v] = {
            'n_samples': int(d['f'].shape[0]),
            'n_terms': int(d['f'].shape[1]),
            'terms': d['terms'],
            'cond': met['cond'],
            'cond_raw': met['cond_raw'],
            'rel_se_median': float(np.median(met['rel_se'])),
            'rel_se_max': float(np.max(met['rel_se'])),
        }

    mc = mc_ensemble(system, noise_level=mc_noise, M=mc_M)
    for v in out:
        if v in mc:
            out[v]['mc'] = mc[v]

    for fam in families:
        resp = perturbation_response(system, fam, levels[fam],
                                     seeds=response_seeds)
        for v in out:
            if v in resp:
                out[v].setdefault('lipschitz', {})[fam] = {
                    'slope': resp[v]['lipschitz_slope'],
                    'levels': resp[v]['levels'],
                    'rel_err': resp[v]['rel_err'],
                }
    return out
