"""Variable projection of trigonometric frequencies on a candidate's measurement rows."""
from copy import deepcopy
from itertools import chain, islice

import numpy as np
from scipy.fft import next_fast_len
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from epde.preprocessing import weak_form as wf
from epde.structure.weak_rows import weak_term, _param


def fitted_factors(term):
    return sorted((f for f in term.structure if f.status.get('fit_frequencies')),
                  key=lambda f: (f.label, _param(f, 'dim'), _param(f, 'power')))


def frequency_structure(term):
    """The term identity with each fitted frequency replaced by its free coordinate."""
    return tuple(sorted((('tone', f.label, _param(f, 'dim'), _param(f, 'power'))
                         if f.status.get('fit_frequencies') else ('fixed', f.structural_label))
                        for f in term.structure))


def residual_modes(residual, shape, axis, step, span, upper, count=1):
    """Distinct angular-frequency starts from the line-averaged residual periodogram."""
    lines = np.moveaxis(np.asarray(residual).reshape(shape), axis, -1)
    n = lines.shape[-1]
    subcycle = min(np.pi / (2 * span), upper / 2)
    if n < 3 or step is None:
        return [subcycle] * count
    # Interleave the original Fourier bins with their midpoints. A short
    # record can otherwise put the initializer beyond a local residual maximum.
    nfft = next_fast_len(2*n, real=True)
    lines = (lines - lines.mean(axis=-1, keepdims=True)).reshape(-1, n)
    spectrum = np.fft.rfft(lines, n=nfft, axis=-1)
    power = (np.abs(spectrum)**2).sum(axis=0)
    indices = find_peaks(np.r_[-np.inf, power[1:], -np.inf])[0]
    peaks = []
    for i in sorted(indices, key=power.__getitem__, reverse=True):
        if power[i] <= 0:
            break
        offset = 0.
        if 0 < i < len(power)-1:
            a,b,c = power[i-1:i+2]
            if a+c < 2*b:
                offset = np.clip((a-c)/(2*(a-2*b+c)), -.5, .5)
        peaks.append(min(2*np.pi*(i+offset)/(nfft*step), upper))
        if len(peaks) == count:
            break
    return peaks + [subcycle] * (count - len(peaks))


def row_residual_modes(rows, key, slot, residual, axis, upper, count=1):
    """Seed the continuous fit on the actual row coordinates of this trajectory."""
    plan = rows._plan(key, slot)
    coords = rows.fields[key].axes[axis]
    span = float(np.ptp(coords))
    step = rows.uniform_step(key, axis)
    if step is not None:
        return residual_modes(residual, plan.shape, axis, step * plan.ss[axis], span, upper, count)
    # A nonuniform grid has no FFT frequency bins. Profile sine and cosine on
    # the same half-bin spacing as the regular-grid initializer, one pair at a time.
    centres = plan.cut[axis] + plan.ms[axis] + plan.ss[axis] * np.arange(plan.shape[axis])
    x = coords[centres]
    subcycle = min(np.pi / (2 * span), upper / 2) if span else upper / 2
    if x.size < 3:
        return [subcycle] * count
    lines = np.moveaxis(np.asarray(residual).reshape(plan.shape), axis, 0).reshape(len(x), -1)
    lines = lines - lines.mean(axis=0)
    spacing = np.pi / np.ptp(x)
    frequencies = np.arange(1, int(np.floor(upper / spacing)) + 1) * spacing
    powers = []
    for omega in frequencies:
        basis = np.column_stack([np.cos(omega*x), np.sin(omega*x)])
        basis -= basis.mean(axis=0)
        coefficients = np.linalg.lstsq(basis, lines, rcond=None)[0]
        powers.append(float(np.sum(coefficients * (basis.T @ lines))))
    peaks = find_peaks(np.r_[-np.inf, powers, -np.inf])[0] - 1
    starts = [frequencies[i] for i in sorted(peaks, key=powers.__getitem__, reverse=True) if powers[i] > 0]
    return (starts + [subcycle] * count)[:count]


class ToneColumn:
    """Pointwise modulation followed by the same separable quadrature as weak_matrix.

    A term with an opaque factor has no integration-by-parts reductions. Collapse
    axes without a tone once. Pure trigonometric products are separable.
    """
    def __init__(self, rows, key, slot, term):
        fields = rows.fields[key]
        plan = rows._plan(key, slot)
        tones = fitted_factors(term)
        self.tones = [(f.label, int(_param(f, 'dim')), int(_param(f, 'power'))) for f in tones]
        symbolic, opaque = weak_term(term, fields.D, frozenset(rows.tokens.get(key, {})))
        fixed = [opaque[label] for label in symbolic.opaque if not opaque[label].status.get('fit_frequencies')]
        self.base = None
        if symbolic.data or any(symbolic.coords) or fixed:
            base = fields.point(symbolic.data, symbolic.coords)
            for f in fixed:
                base = base * rows._opaque_values(f, key)
            self.base = wf._crop(base, plan.cut)
        self.shape = plan.shape
        self.axes = [a[c:len(a)-c] for a, c in zip(fields.axes, plan.cut)]
        kernels = wf.window_kernels(plan.ms, fields.steps, plan.q, (0,) * fields.D)
        self.kernels = [k[0] for k in kernels]
        self.strides = plan.ss
        self.active = sorted({axis for _, axis, _ in self.tones})
        if self.base is not None:
            for axis in range(fields.D):
                if axis not in self.active:
                    self.base = self.integrate(self.base, axis)

    def integrate(self, values, axis):
        kernel = self.kernels[axis]
        windows = np.lib.stride_tricks.sliding_window_view(values, len(kernel), axis=axis)
        slices = [slice(None)] * windows.ndim
        slices[axis] = slice(None, None, self.strides[axis])
        return windows[tuple(slices)] @ kernel

    def __call__(self, frequencies):
        factors = {}
        for w, (label, axis, power) in zip(frequencies, self.tones):
            x = self.axes[axis]
            value = (np.sin(w*x) if label == 'sin' else np.cos(w*x)) ** power
            factors[axis] = factors[axis] * value if axis in factors else value
        value = self.base
        for axis, factor in factors.items():
            if self.base is None:
                kernel = self.kernels[axis]
                windows = np.lib.stride_tricks.sliding_window_view(factor, len(kernel))
                factor = windows[::self.strides[axis]] @ kernel
            shape = [1] * len(self.shape)
            shape[axis] = -1
            factor = factor.reshape(shape)
            value = factor if value is None else value * factor
        if self.base is None:
            return np.broadcast_to(value, self.shape).ravel()
        for axis in self.active:
            value = self.integrate(value, axis)
        return value.ravel()


def project_frequencies(samples, axes, upper, subcycle, spectral, intercept, count, row_periods=(),
                        observed_peaks=None, shared_coefficients=False):
    """Return frequencies minimizing the profiled training residual across trajectories.

    Each sample contains a target, square-root weights and fixed or callable
    columns. Each callable owns its own consecutive frequency coordinates.
    Observed peaks are ranked by energy separately on each axis.
    """
    normalizer = np.linalg.norm(np.concatenate([y * weight for y, weight, _ in samples]))
    if not np.isfinite(normalizer) or normalizer == 0:
        return None

    def evaluate(w):
        blocks = []
        targets = []
        for target, weights, columns in samples:
            design = [col(w[indices]) if indices is not None else col for col, indices in columns]
            if intercept:
                design.append(np.ones(len(target)))
            x = np.column_stack(design) * weights[:, None]
            blocks.append(x)
            targets.append(target * weights)
        if shared_coefficients:
            blocks = [np.vstack(blocks)]
            targets = [np.concatenate(targets)]
        residuals = []
        full_rank = True
        for x, y in zip(blocks, targets):
            count()
            norms = np.linalg.norm(x, axis=0)
            unit = x / np.where(norms > 0, norms, 1.)
            coef, _, rank, _ = np.linalg.lstsq(unit, y, rcond=None)
            residuals.append(y - unit @ coef)
            full_rank &= rank == unit.shape[1]
        return np.concatenate(residuals) / normalizer, full_rank

    def residual(w):
        return evaluate(w)[0]

    # Scan each axis through Nyquist, including the subcycle interval. Other
    # tones stay present so products can gain from either frequency moving.
    low = np.asarray(subcycle, dtype=float)
    # The fixed-term projection can suppress a forcing peak and expose one
    # of its neighbouring lobes. One Fourier bin of the observed record is
    # four times the quarter-cycle start (or the available band on short axes).
    spacing = 4 * low
    spectral = np.minimum(np.asarray(spectral, dtype=float), upper)
    left = np.maximum(low / 2, spectral - spacing)
    right = np.minimum(upper, spectral + spacing)
    best = low.copy()
    for j in range(len(best)):
        candidates = np.unique(np.minimum(upper[j], np.r_[low[j] * np.exp2(
            np.arange(np.ceil(np.log2(upper[j]/low[j])) + 1)), left[j], spectral[j], right[j]]))
        trials = np.tile(best, (len(candidates), 1))
        trials[:, j] = candidates
        best = min(trials, key=lambda w: np.linalg.norm(residual(w))).copy()
    starts = [low / 2, low, best, left, right, spectral]
    if len(set(axes)) < len(axes):
        # The residual can identify two modes before it identifies which
        # modulation carries which mode. Try the opposite assignment as well.
        alternate = starts[-1].copy()
        for axis in set(axes):
            indices = np.flatnonzero(np.asarray(axes) == axis)
            alternate[indices] = alternate[indices[::-1]]
        starts.append(alternate)
    if observed_peaks:
        # Start jointly at the strongest observed peaks, then try each runner-up
        # separately and all together. This keeps the number of starts linear.
        first = spectral.copy()
        second = spectral.copy()
        for j, axis in enumerate(axes):
            ranked = observed_peaks.get(axis, ())
            peaks = list(islice((p for p in ranked if 0 < p <= upper[j]), 2))
            if peaks:
                first[j], second[j] = peaks[0], peaks[-1]
        starts.extend([first, second])
        for j in np.flatnonzero(first != second):
            mixed = first.copy()
            mixed[j] = second[j]
            starts.append(mixed)
    optimum, cost = None, np.inf
    # Symmetric weak windows on a strided lattice can make two frequencies
    # give proportional columns even below the input Nyquist limit. With
    # normalized residuals, compare costs to the precision of their dot product.
    precision = np.finfo(float).eps * sum(len(y) for y, _, _ in samples)
    for start in np.unique(starts, axis=0):
        fit = least_squares(residual, start, bounds=(np.zeros(len(low)), upper),
                            x_scale=np.maximum(low, start), method='trf')
        lower_alias = (optimum is not None and abs(fit.cost-cost) <= precision and
                       np.linalg.norm(fit.x / upper) < np.linalg.norm(optimum / upper))
        if (np.isfinite(fit.cost) and (fit.cost < cost - precision or lower_alias) and
                evaluate(fit.x)[1]):
            optimum, cost = fit.x, fit.cost
    if optimum is not None:
        for periods in row_periods:
            for j, period in enumerate(periods):
                if optimum[j] <= period/2:
                    continue
                candidate = optimum.copy()
                candidate[j] = abs((candidate[j] + period/2) % period - period/2)
                r, full_rank = evaluate(candidate)
                folded_cost = (r @ r) / 2
                if full_rank and folded_cost <= cost + precision:
                    optimum, cost = candidate, folded_cost
    return optimum


def fit_support(equation, kept, intercept, rows):
    """A fitted copy for this intercept state. Only accepted frequencies enter native caches."""
    kept = sorted(kept, key=lambda j: frequency_structure(equation.structure[j]))
    terms = [equation.structure[j] for j in kept]
    slot = equation.main_var_to_explain
    key = (slot, equation.target.factors_labels, tuple(map(frequency_structure, terms)), intercept)
    if key not in rows.frequency_fits:
        rows.frequency_fits[key] = _fit_values(rows, equation.target, slot, terms, intercept)
    best = rows.frequency_fits[key]
    if best is None:
        return None
    variant = deepcopy(equation)
    values = iter(best)
    for j in kept:
        term = variant.structure[j]
        for factor in fitted_factors(term):
            factor.set_param(float(next(values)), name='freq')
        term.resetSavedState()
    variant.reset_state(reset_right_part=False)
    variant._invalidate_label_cache()
    if len(variant.terms_labels) != len(variant.structure):
        return None
    return variant


def _fit_values(rows, target_term, slot, terms, intercept):
    tones = [fitted_factors(term) for term in terms]
    factors = list(chain.from_iterable(tones))
    axes = [int(_param(f, 'dim')) for f in factors]
    from epde.operators.common.multisample import fit_row_weights, joint_enabled
    joint = joint_enabled(len(rows.keys))
    samples, starts, low, upper, row_periods = [], [], [], [], []
    target = rows.columns(target_term, slot)
    weights = rows.weights(slot)
    fixed = [None if fs else rows.columns(t, slot) for t, fs in zip(terms, tones)]
    for key in rows.keys:
        y = target[key]
        columns = []
        index = 0
        for term, fs, known in zip(terms, tones, fixed):
            if fs:
                cols = ToneColumn(rows, key, slot, term)
                if cols.base is not None and not np.all(np.isfinite(cols.base)):
                    return None
                columns.append((cols, slice(index, index+len(fs))))
                index += len(fs)
            else:
                columns.append((known[key], None))
        weight = np.sqrt(fit_row_weights(weights[key]) if joint else weights[key])
        samples.append((y, weight, columns))
        fixed_columns = [col for col, idx in columns if idx is None]
        if intercept:
            fixed_columns.append(np.ones(len(y)))
        if not np.all(np.isfinite(y)) or any(not np.all(np.isfinite(x)) for x in fixed_columns):
            return None
        r = y
        if fixed_columns:
            x = np.column_stack(fixed_columns)
            r = y - x @ np.linalg.lstsq(x * weight[:, None], y * weight, rcond=None)[0]
            rows.frequency_solves += 1
        fields, plan = rows.fields[key], rows._plan(key, slot)
        steps = {d: rows.uniform_step(key, d) for d in set(axes)}
        row_periods.append([2*np.pi/(steps[d]*plan.ss[d]) if steps[d] else np.inf for d in axes])
        span = [np.ptp(fields.axes[d]) for d in axes]
        bounds = [f.status['frequency_nyquist'][d] for f,d in zip(factors, axes)]
        upper.append(bounds)
        low.append([min(np.pi/(2*l), b/2) if l else b/2 for l,b in zip(span,bounds)])
        start = np.empty(len(axes))
        for d in sorted(set(axes)):
            indices = [j for j, axis in enumerate(axes) if axis == d]
            first = indices[0]
            start[indices] = row_residual_modes(rows, key, slot, r, d, bounds[first], len(indices))
        starts.append(start)
    def count():
        rows.frequency_solves += 1
    observed = {axis: rows.observed_frequencies(axis) for axis in set(axes)}
    return project_frequencies(samples, axes, np.min(upper, axis=0), np.min(low, axis=0),
                               np.median(starts, axis=0), intercept, count, row_periods, observed, shared_coefficients=joint)
