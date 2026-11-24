"""Select weak windows by the perturbation sensitivity of generic derivative regressions."""
import numpy as np
from selector import Geometry, WF, neighbors, probes


def regressions(a, noise, targets):
    """Freeze identifiable predictor directions in native, unit-normalized coordinates."""
    scale = np.sqrt(np.mean(a*a, axis=0))
    active = np.flatnonzero(scale > 0)
    fits = []
    for target in targets:
        predictors = active[active != target]
        if scale[target] == 0 or not len(predictors):
            continue
        x, e = a[:, predictors], noise[:, predictors]
        directions = WF.signal_directions(x.T @ x, e.T @ e)
        if directions.shape[1]:
            basis = np.linalg.qr(directions, mode='reduced')[0]
            fits.append((target, predictors, basis))
    return np.where(scale > 0, scale, 1.), fits


def sensitivity(a, noise, scale, fits):
    """Largest standardized coefficient half-difference under the same +/- field noise."""
    if not fits:
        return np.inf
    plus, minus = (a+noise)/scale, (a-noise)/scale
    worst = 0.
    for target, predictors, basis in fits:
        high, _, rh, _ = np.linalg.lstsq(plus[:, predictors] @ basis, plus[:, target], rcond=None)
        low, _, rl, _ = np.linalg.lstsq(minus[:, predictors] @ basis, minus[:, target], rcond=None)
        if min(rh, rl) < basis.shape[1]:
            return np.inf
        drift = basis @ ((high-low)/2)
        worst = max(worst, float(np.max(np.abs(drift))))
    return worst


def improves(score, previous):
    if not (np.isfinite(score) and np.isfinite(previous)):
        return False
    return bool(previous-score > np.finfo(float).eps * max(1., previous))


def select(fields, plan, orders):
    if isinstance(plan, WF.WindowPlan):
        plan = {k: getattr(plan, k) for k in ('ms', 'ss', 'shape', 'q', 'cut')}
    if len(orders) != fields.D or any(o < 0 or int(o) != o for o in orders):
        raise ValueError('Supply a nonnegative maximum derivative order for each coordinate axis.')
    terms = probes(fields.values, tuple(map(int, orders)))
    targets = [i for i, term in enumerate(terms) if term.data and any(term.data[0][1])]
    primary = Geometry(plan, fields.shape, plan['ms'], (0,)*fields.D)
    models = {var: WF.noise_model(v)[:2] for var, v in fields.values.items()}

    def perturb(seed):
        rng = np.random.default_rng(seed)
        return WF.Fields({var: WF.ar_noise(rng, fields.shape, *models[var])
                          for var in sorted(models)}, fields.axes)

    def measure_noise(geometry, draw, base=None):
        values = geometry.matrix(draw, terms, base)
        values[:, -fields.D:] = 0.  # coordinate probes are deterministic
        return values

    a = primary.matrix(fields, terms)
    scale, fits = regressions(a, measure_noise(primary, perturb(0)), targets)
    first = perturb(1)
    noise = measure_noise(primary, first)
    baseline = sensitivity(a, noise, scale, fits)
    best, selected, chosen = baseline, None, None
    if np.isfinite(baseline):
        for geometry in neighbors(plan, fields.shape):
            mixed = geometry.matrix(fields, terms, a)
            candidate = sensitivity(mixed, measure_noise(geometry, first, noise), scale, fits)
            if improves(candidate, best):
                best, selected, chosen = candidate, geometry, mixed
    audit = dict(targets=len(fits), original=baseline if np.isfinite(baseline) else None,
                 proposed=best if np.isfinite(best) else None)
    if selected is not None:
        check = perturb(2)
        check_noise = measure_noise(primary, check)
        old = sensitivity(a, check_noise, scale, fits)
        new = sensitivity(chosen, measure_noise(selected, check, check_noise), scale, fits)
        audit.update(check_original=old if np.isfinite(old) else None, check_proposed=new if np.isfinite(new) else None)
        if not improves(new, old):
            selected = None
    return primary, selected, audit
