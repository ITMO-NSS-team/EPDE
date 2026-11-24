"""Weighted least squares for shared coefficients in plain and assignment modes.

Column scaling is computed across the pooled rows. Coefficients are returned
in physical units. Instability is evaluated on each trajectory's grid.
"""
import numpy as np

from epde.interface.search_config import active_config


def joint_enabled(n_samples: int) -> bool:
    """Return whether the search pools multiple trajectories."""
    return (n_samples > 1 and
            getattr(active_config().objectives, 'multisample_mode', 'mean') == 'joint')


def fit_row_weights(weights, weighting=None):
    """Validate one trajectory's weights and normalize to unit mass in samples mode.

    Individual weights can be zero. At least one weight must be positive.
    """
    if weighting is None:
        weighting = getattr(active_config().objectives, 'multisample_weighting', 'samples')
    if weighting not in ('samples', 'points'):
        raise ValueError("multisample weighting must be 'samples' or 'points'.")
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if (not weights.size or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.any(weights > 0)):
        raise ValueError('sample weights must be finite, non-negative, and have positive total mass.')
    if weighting == 'points':
        return weights
    # Scale first so the sum cannot overflow for large weights.
    scaled = weights / weights.max()
    return scaled / scaled.sum()


def stable_l2_norm(values, axis=None):
    """Compute an L2 norm without squaring the unscaled values."""
    values = np.asarray(values, dtype=float)
    peak = np.max(np.abs(values), axis=axis, keepdims=True, initial=0.0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        scaled = values / np.where(peak > 0, peak, 1.0)
        norm = np.sqrt(np.sum(scaled * scaled, axis=axis, keepdims=True)) * peak
    return np.squeeze(norm, axis=axis) if axis is not None else float(norm.reshape(-1)[0])


def _samples(samples):
    """Validate the sample blocks and sort them for reproducible reductions."""
    result, keys, width = [], set(), None
    for key, X, y, weights in samples:
        if key in keys:
            raise ValueError(f'duplicate trajectory key {key!r} in a joint fit.')
        keys.add(key)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[0] != y.size or y.size != weights.size:
            raise ValueError('joint fit features, target, and weights must have matching non-empty rows.')
        if width is None:
            width = X.shape[1]
        if X.shape[1] != width:
            raise ValueError('joint fit feature columns must agree across trajectories.')
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError('joint fit features and targets must be finite on every trajectory.')
        fit_row_weights(weights, 'points')
        result.append((key, X, y, weights))
    if not result:
        raise ValueError('a joint fit needs at least one trajectory.')
    return sorted(result, key=lambda sample: (type(sample[0]).__name__, repr(sample[0])))


class _FittedColumns:
    """Expose a fitted subset to the discrepancy and instability scorers."""
    def __init__(self, samples, weights, main_var=None):
        weights = np.asarray(weights, dtype=float)
        active = weights[:-1] != 0
        self.weights_final = np.r_[weights[:-1][active], weights[-1]]
        self.weights_internal = self.weights_final
        self.active_mask = np.ones(int(active.sum()), dtype=bool)
        self.main_var_to_explain = main_var
        self.targets = {key: y for key, _X, y, _w in samples}
        self.features = ({key: X[:, active] for key, X, _y, _w in samples}
                         if active.any() else None)

    def evaluate(self, *, active_only=False):
        return self.targets, self.features


def _context(samples, penalty_coeff, shapes=None):
    from epde.operators.common.objectives import FitContext
    return FitContext(g_fun_vals={key: weights for key, _X, _y, weights in samples},
                      data_shape=({key: np.asarray(y.shape, dtype=int)
                                   for key, _X, y, _w in samples}
                                  if shapes is None else
                                  {key: np.asarray(shape, dtype=int) for key, shape in shapes.items()}),
                      penalty_coeff=penalty_coeff, for_rps=False)


def joint_subset_table(samples, *, discrepancy, subsets=None, weighting=None,
                       penalty_coeff=0.2):
    """Fit and score the specified supports with one vector across samples.

    Each block is ``(trajectory_key, X, y, raw_row_weights)``. Bit ``p`` selects
    the intercept after the ``p`` feature columns. Returns
    ``(discrepancy, amplification, p+1, coefficients, full_rank)`` in subset
    order, or in bitmask order when ``subsets`` is omitted.

    The weighted pooled design can be full rank even when each block is
    singular. The SVD tests rank without squaring the condition number.
    """
    from epde.operators.common.objectives import Discrepancy
    from epde.operators.common.right_part_selection import _LIN_DEP_RTOL
    samples = _samples(samples)
    p = samples[0][1].shape[1]
    width = p + 1
    masks = list(range(1 << width) if subsets is None else subsets)
    if any(not isinstance(mask, (int, np.integer)) or mask < 0 or mask >= 1 << width for mask in masks):
        raise ValueError('joint fit subset masks must address the feature columns and optional constant.')
    design, target = [], []
    fit_weights = [fit_row_weights(weights, weighting) for _key, _X, _y, weights in samples]
    # Rescale the square roots together to avoid overflow and premature underflow.
    root_scale = np.sqrt(max(weights.max() for weights in fit_weights))
    for (_key, X, y, _weights), weights in zip(samples, fit_weights):
        root = np.sqrt(weights) / root_scale
        design.append(np.column_stack([X, np.ones(y.size)]) * root[:, None])
        target.append(y * root)
    design = np.vstack(design)
    target = np.concatenate(target)
    norms = stable_l2_norm(design, axis=0)
    scaled = design / np.where(norms > 0, norms, 1.0)
    coefficients = np.zeros((len(masks), width))
    full_rank = np.zeros(len(masks), dtype=bool)
    discrepancy_values = np.full(len(masks), np.inf)
    amplification = np.full(len(masks), np.inf)
    scorer = Discrepancy(discrepancy)
    context = _context(samples, penalty_coeff)
    for position, mask in enumerate(masks):
        cols = [j for j in range(width) if (mask >> j) & 1]
        if not cols:
            continue
        solution, _residuals, rank, _singular = np.linalg.lstsq(scaled[:, cols], target,
                                                               rcond=_LIN_DEP_RTOL)
        full = coefficients[position]
        full[cols] = solution / np.where(norms[cols] > 0, norms[cols], 1.0)
        full_rank[position] = rank == len(cols) and np.all(norms[cols] > 0)
        view = _FittedColumns(samples, full)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            discrepancy_values[position] = scorer.compute(view, context)
        ratios = []
        active = full[:-1] != 0
        for _key, X, y, _weights in samples:
            denominator = stable_l2_norm(y)
            # Skip zero coefficients to avoid multiplying by an infinite norm.
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                numerator = (float(np.abs(full[:-1][active]) @ stable_l2_norm(X[:, active], axis=0))
                             + abs(full[-1]) * np.sqrt(y.size))
                ratio = numerator / denominator if denominator > 0 else np.inf
            ratios.append(ratio if np.isfinite(ratio) else np.inf)
        amplification[position] = float(np.mean(ratios))
    return discrepancy_values, amplification, width, coefficients, full_rank


def joint_instability_table(samples, coefficients, shapes, *, main_var,
                            penalty_coeff=0.2):
    """Score each shared support on the original grids and row weights."""
    from epde.operators.common.objectives import Instability
    samples = _samples(samples)
    if set(shapes) != {sample[0] for sample in samples}:
        raise ValueError('joint instability needs one grid shape per trajectory.')
    if any(int(np.prod(shapes[key])) != y.size for key, _X, y, _w in samples):
        raise ValueError('joint instability grid shapes must match their trajectory rows.')
    context = _context(samples, penalty_coeff, shapes)
    scorer = Instability()
    return np.asarray([scorer.compute(_FittedColumns(samples, full, main_var), context)
                       for full in coefficients], dtype=float)
