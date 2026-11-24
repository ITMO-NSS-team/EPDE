"""Full-record measurement errors. Each trial uses a local RNG in field order."""
import numpy as np
from scipy.signal import lfilter


FAMILIES = ('spikes', 'correlated', 'signal_dependent')
LEVEL = 5
SPIKE_FRACTION = 0.01
RHO = 0.8


def rms(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def errors(values, family, rng):
    if family == 'spikes':
        count = max(1, int(np.floor(values.size * SPIKE_FRACTION + 0.5)))
        positions = rng.choice(values.size, count, replace=False)
        draw = np.zeros(values.shape)
        draw.flat[positions] = rng.choice([-1., 1.], count) * np.sqrt(values.size / count)
        return draw
    draw = rng.standard_normal(values.shape)
    if family == 'signal_dependent':
        return draw * (np.abs(values) / rms(values))
    # A stationary first sample on each axis and independent subsequent innovations.
    # Sequential 1-D filters give a separable covariance on the full grid.
    innovation = np.sqrt(1 - RHO**2)
    for axis in range(draw.ndim):
        series = np.moveaxis(draw, axis, 0)
        series[1:] *= innovation
        draw = lfilter([1.], [1., -RHO], draw, axis=axis)
    return draw


def observe(problem, level, seed, *, family):
    if family not in FAMILIES or level != LEVEL:
        raise ValueError((family, level))
    if problem.derivs is not None:
        raise ValueError('Artificial noise requires recomputing derivatives from observations.')
    rng = np.random.default_rng(seed)
    fields, actual_rms, target_sigma = {}, {}, {}
    for name, original in problem.data.items():
        values = np.asarray(original, dtype=np.float64)
        sigma = 0. if np.ptp(values) == 0 else float(np.std(values)) * (level / 100)
        target_sigma[name] = sigma
        fields[name] = values + sigma * errors(values, family, rng) if sigma else values.copy()
        actual_rms[name] = rms(fields[name] - values)
    params = {'spikes': {'fraction': SPIKE_FRACTION},
              'correlated': {'lag_one': RHO, 'axes': list(problem.axis_names)},
              'signal_dependent': {'weight': 'abs(field)/rms(field)'}}
    return fields, dict(family=family, percent=level, seed=seed, parameters=params[family],
                        field_rms=actual_rms, target_sigma=target_sigma)
