"""How a trigonometric token family's frequency is defined.

``frequency_grid`` turns the keywords of ``TrigonometricTokens`` /
``PhasedSine1DTokens`` into one of these kinds -- the set of keywords given
(not None) selects the kind, anything else is refused:

    ============================  ==========  ===================================
    keywords                      kind        frequencies
    ============================  ==========  ===================================
    (none)                        continuous  any value in freq=(lo, hi)
    freq_num [, 'linear']         linspace    np.linspace(lo, hi, freq_num)
    freq_num, freq_spacing='log'  geomspace   np.geomspace(lo, hi, freq_num)
    freq_step                     step        lo, lo + step, ... up to hi
    freq_values                   values      the listed values
    freq_base, freq_num           harmonics   base, 2*base, ..., freq_num*base
    freq_from_data                data        the freq_from_data strongest
                                              spectral peaks of the fit data,
                                              of each axis's own spectrum;
                                              freq, if given, bounds them
    ============================  ==========  ===================================

``dominant_frequencies`` finds the data kind's peaks when the pool is built.
"""
import bisect
import warnings

import numpy as np

KINDS = {
    frozenset(): 'continuous',
    frozenset({'freq_num'}): 'linspace',
    frozenset({'freq_num', 'freq_spacing'}): 'spaced',
    frozenset({'freq_step'}): 'step',
    frozenset({'freq_values'}): 'values',
    frozenset({'freq_base', 'freq_num'}): 'harmonics',
    frozenset({'freq_from_data'}): 'data',
}

_TABLE = ('freq=(lo, hi) alone (continuous); freq_num [with freq_spacing=\'linear\' or \'log\']; '
          'freq_step; freq_values; freq_base with freq_num; freq_from_data')


def _number(value) -> bool:
    """A real number, not a bool, a string or a container."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _positive_int(name, value) -> int:
    if not (_number(value) and np.isfinite(value) and int(value) == value and value >= 1):
        raise ValueError(f'{name} must be a positive integer, got {value!r}.')
    return int(value)


def _positive_float(name, value) -> float:
    if not (_number(value) and np.isfinite(value) and value > 0):
        raise ValueError(f'{name} must be a positive finite number, got {value!r}.')
    return float(value)


def _range(freq, positive: bool) -> tuple:
    try:
        ok = len(freq) == 2 and all(_number(v) for v in freq) and freq[1] >= freq[0]
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f'freq must be (low, high) with low <= high, got {freq!r}.')
    lo, hi = float(freq[0]), float(freq[1])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f'freq must be finite, got {freq!r}.')
    if positive and not lo > 0:
        raise ValueError(f'Grid frequencies must be positive, got freq={freq!r}.')
    return lo, hi


def frequency_grid(freq, freq_given: bool, freq_num=None, freq_spacing=None, freq_step=None,
                   freq_values=None, freq_base=None, freq_from_data=None):
    """``(kind, bounds, grid)``. ``grid`` is a tuple of positive floats for the grid
    kinds and None for 'continuous' and 'data'. ``bounds`` is ``freq`` untouched
    for 'continuous', the float (lo, hi) the grid lies in for the grid kinds, and
    the band (or None) for 'data'. ``freq_given`` tells whether the caller passed
    ``freq`` itself (it is refused where the kind defines its own range)."""
    given = {name for name, value in (('freq_num', freq_num), ('freq_spacing', freq_spacing),
                                      ('freq_step', freq_step), ('freq_values', freq_values),
                                      ('freq_base', freq_base), ('freq_from_data', freq_from_data))
             if value is not None}
    kind = KINDS.get(frozenset(given))
    if kind is None:
        raise ValueError(f'These frequency keywords do not define one kind: {sorted(given)}. '
                         f'Use one of: {_TABLE}.')
    if kind == 'continuous':
        return kind, freq, None
    if freq_spacing is not None and freq_spacing not in ('linear', 'log'):
        raise ValueError(f"freq_spacing must be 'linear' or 'log', got {freq_spacing!r}.")

    if kind in ('linspace', 'spaced'):
        num = _positive_int('freq_num', freq_num)
        lo, hi = _range(freq, positive=True)
        if lo == hi and num > 1:
            raise ValueError(f'freq=({lo}, {hi}) holds one frequency; freq_num={num} would repeat it. '
                             'Use freq_num=1 (or freq_values) to fix a frequency.')
        if kind == 'spaced' and freq_spacing == 'log':
            kind, values = 'geomspace', np.geomspace(lo, hi, num)
        else:
            kind, values = 'linspace', np.linspace(lo, hi, num)
    elif kind == 'step':
        step = _positive_float('freq_step', freq_step)
        lo, hi = _range(freq, positive=True)
        # The 1e-9 (in steps) only absorbs float noise in (hi - lo) / step, so an
        # hi meant to be on the grid stays on it; the cap keeps that last value
        # exactly hi rather than hi + noise.
        last = max(int(np.floor((hi - lo) / step + 1e-9)), 0)
        values = [min(lo + k * step, hi) for k in range(last + 1)]
    elif kind == 'values':
        if freq_given:
            raise ValueError('freq_values defines the frequencies itself; do not pass freq as well.')
        if (not isinstance(freq_values, (list, tuple, np.ndarray)) or len(freq_values) == 0
                or not all(_number(v) and np.isfinite(v) and v > 0 for v in np.ravel(freq_values).tolist())):
            raise ValueError(f'freq_values must be a list of positive finite numbers, got {freq_values!r}.')
        values = np.ravel(freq_values).astype(float)
    elif kind == 'harmonics':
        if freq_given:
            raise ValueError('freq_base with freq_num defines the frequencies itself; do not pass freq as well.')
        base = _positive_float('freq_base', freq_base)
        values = [base * k for k in range(1, _positive_int('freq_num', freq_num) + 1)]
    else:                                                     # 'data'
        _positive_int('freq_from_data', freq_from_data)
        band = None
        if freq_given:
            band = _range(freq, positive=False)
            if not (band[0] >= 0 and band[1] > band[0]):
                raise ValueError(f'The band freq for freq_from_data needs 0 <= low < high, got {freq!r}.')
        return kind, band, None

    grid = tuple(sorted({float(v) for v in values}))
    return kind, (grid[0], grid[-1]), grid


def _hann(n: int) -> np.ndarray:
    """The periodic Hann window, for which the three-point interpolation below is unbiased."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _spectrum_peaks(lines: np.ndarray, spacing: float):
    """``[(energy, cyclic frequency, bin width)]`` of the local maxima of the
    line-averaged, unit-total Hann power spectrum of ``lines`` (rows), or None
    when the lines carry no fluctuation."""
    lines = lines - lines.mean(axis=1, keepdims=True)
    n = lines.shape[1]
    power = (np.abs(np.fft.rfft(lines * _hann(n), axis=1)) ** 2).mean(axis=0)
    total = power.sum()
    if not total > 0:
        return None
    power = power / total
    magnitude = np.sqrt(power)
    width = 1.0 / (n * spacing)
    peaks = []
    # k = 2 .. K-2: the Hann main lobe of the removed mean reaches bin 1, and the
    # interpolation needs both neighbours.
    for k in range(2, power.size - 1):
        if power[k] > power[k - 1] and power[k] >= power[k + 1]:
            a, b, c = magnitude[k - 1], magnitude[k], magnitude[k + 1]
            delta = 2.0 * (c - a) / (a + 2.0 * b + c)
            peaks.append((float(power[k - 1] + power[k] + power[k + 1]), (k + delta) * width, width))
    return peaks


def _merge_peaks(peaks):
    """``[(energy, cyclic frequency)]`` clusters of ``peaks`` (``(energy,
    frequency, bin width)`` triples), in the order their leaders are taken.

    Greedy, strongest first (ties by frequency): the strongest unmerged peak
    absorbs every unmerged peak ``j`` with ``|f_j - f| < max(width,
    width_j)``; the cluster's energy is the members' sum in sort order and its
    frequency is the finest-resolution member's (the strongest among equals).
    Each bin width's unmerged peaks are kept sorted by frequency, so a leader
    looks only at the peaks inside a window of its reach (a bisection per
    width) instead of at every later peak; the window is twice the reach and
    every candidate is tested with the exact comparison above, so float
    rounding at the window's edge cannot change what merges.
    """
    peaks = sorted(peaks, key=lambda p: (-p[0], p[1]))
    by_width = {}
    for i, (_energy, freq, width) in enumerate(peaks):
        by_width.setdefault(width, []).append((freq, i))
    groups = []
    for width, entries in by_width.items():
        entries.sort()
        groups.append((width, [f for f, _ in entries], [i for _, i in entries]))
    merged, clusters = [False] * len(peaks), []
    for i, (energy, freq, width) in enumerate(peaks):
        if merged[i]:
            continue
        merged[i] = True
        absorbed = []
        for group_width, freqs, index in groups:
            reach = 2.0 * max(width, group_width)
            for pos in range(bisect.bisect_left(freqs, freq - reach), bisect.bisect_right(freqs, freq + reach)):
                j = index[pos]
                if not merged[j] and abs(peaks[j][1] - freq) < max(width, peaks[j][2]):
                    merged[j] = True
                    absorbed.append(j)
        members = [peaks[i]] + [peaks[j] for j in sorted(absorbed)]
        finest = min(members, key=lambda p: (p[2], -p[0]))
        clusters.append((sum(p[0] for p in members), finest[1]))
    return clusters


def dominant_frequencies(trajectories, axes, n: int, unit: str, band=None, per_axis: bool = False):
    """The ``n`` strongest spectral peaks of the observed fields, in ``unit``
    ('angular': 2*pi*f, for sin(freq * x); 'cyclic': f, for sin(2*pi*freq*x)),
    inside ``band`` when given, ascending.

    With ``per_axis`` each axis keeps its own peaks: ``{axis: frequencies}``,
    the ``n`` strongest of that axis's spectra alone, for every axis with a
    peak. A frequency is in the units of its axis's coordinate, so it is never
    put on another axis: heat_solar's daily cycle, 7.3e-5 rad/s along time in
    seconds, made cos(7.3e-5 x) along a depth of 1.5 m -- 1 to nine digits. An
    axis without a peak gets none (with an ``EPDEUsageWarning``); when no axis
    has one this raises as below.

    Every trajectory's raw field of every variable (full grid, unmasked) is read
    along every axis of ``axes``. Per axis: the coordinates must be uniform; each
    line along the axis is demeaned and Hann-windowed; the power is averaged over
    the lines and normalised to total 1. A local maximum's frequency is refined by
    the three-point Hann interpolation (unbiased for an isolated tone away from DC), and its
    energy is the main lobe's three bins. Peaks from all spectra are pooled: the
    strongest unmerged peak absorbs every peak closer than the coarser of the two
    bin widths, the cluster's energy is the sum, and its frequency is its
    finest-resolution member's. Nothing here is tuned -- the resolution of the
    data decides what merges.

    A clean tone gives one local maximum, so fewer than ``n`` peaks (with an
    ``EPDEUsageWarning``) is normal for clean data with few tones. A tone needs a
    few periods in the record: one below 2 cycles per record cannot be told from
    the removed mean, and a record without any resolvable peak raises.
    """
    from epde.integrate.data_profile import axis_coordinates, _lines   # lazy: interface -> integrate cycle

    if unit not in ('angular', 'cyclic'):
        raise ValueError(f"unit must be 'angular' or 'cyclic', got {unit!r}.")
    peaks, live = [], False
    by_axis = {axis: [] for axis in axes}
    for trajectory in trajectories:
        grids = trajectory.grids('full')
        for name in trajectory.variable_names:
            field = np.asarray(trajectory.raw_field(name), dtype=float)
            if not np.all(np.isfinite(field)):
                raise ValueError(f'freq_from_data: the field {name!r} holds non-finite values.')
            for axis in axes:
                if axis >= field.ndim:
                    raise ValueError(f'freq_from_data: axis {axis} asked for, but the data has '
                                     f'{field.ndim} axis(es) (dimensionality is the highest axis index).')
                coords = axis_coordinates(grids[axis], axis)
                if coords.size < 4:
                    continue
                steps = np.diff(coords)
                spacing = float(np.median(steps))
                if not (spacing > 0 and np.allclose(steps, spacing, rtol=1e-6, atol=0)):
                    raise ValueError(f'freq_from_data needs uniform spacing along axis {axis}; '
                                     'the spectrum is taken with an FFT.')
                found = _spectrum_peaks(_lines(field, axis), spacing)
                if found is not None:
                    live = True
                    peaks.extend(found)
                    by_axis[axis].extend(found)
    if not live:
        raise ValueError('freq_from_data: every field is constant along the asked axes; no spectrum.')

    if not per_axis:
        chosen = _strongest(peaks, n, unit, band)
        if not chosen:
            _no_peak(band)
        return chosen
    found = {axis: _strongest(axis_peaks, n, unit, band, axis=axis) for axis, axis_peaks in by_axis.items()}
    if not any(found.values()):
        _no_peak(band)
    import epde.globals as global_var
    for axis in [axis for axis, values in found.items() if not values]:
        warnings.warn(f'freq_from_data: no spectral peak along axis {axis}'
                      f'{" in the band " + str(band) if band is not None else ""}; that axis gets no '
                      'frequency.', global_var.EPDEUsageWarning, stacklevel=2)
    return {axis: values for axis, values in found.items() if values}


def _strongest(peaks, n: int, unit: str, band, axis=None) -> tuple:
    """The ``n`` strongest clusters of ``peaks`` (``_merge_peaks``) in ``unit``
    inside ``band``, ascending; () when there is none, with a warning when
    there are fewer than ``n``."""
    clusters = _merge_peaks(peaks)
    scale = 2.0 * np.pi if unit == 'angular' else 1.0
    clusters = [(energy, freq * scale) for energy, freq in clusters]
    if band is not None:
        clusters = [c for c in clusters if band[0] <= c[1] <= band[1]]
    if not clusters:
        return ()
    clusters.sort(key=lambda c: (-c[0], c[1]))
    chosen = clusters[:n]
    if len(chosen) < n:
        import epde.globals as global_var
        where = '' if axis is None else f' along axis {axis}'
        warnings.warn(f'freq_from_data asked for {n} frequencies{where}; the data has only {len(chosen)} '
                      f'spectral peak(s){" in the band " + str(band) if band is not None else ""}.',
                      global_var.EPDEUsageWarning, stacklevel=3)
    return tuple(sorted(float(freq) for _, freq in chosen))


def _no_peak(band):
    if band is not None:
        raise ValueError(f'freq_from_data: no spectral peak inside the band {band}.')
    raise ValueError('freq_from_data: no spectral peak in the data -- the record is too short, or its '
                     'tones make fewer than 2 cycles per record.')
