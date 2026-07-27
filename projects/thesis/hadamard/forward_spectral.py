"""Forward Hadamard well-posedness, estimated purely from data.

A discovered equation can fit the measured field yet correspond to an
*ill-posed* evolution operator (e.g. a backward-heat ``u_t = -nu u_xx`` form):
small high-frequency perturbations of the initial condition then blow up, so
the solution does NOT depend continuously on the data. Whether that is the case
is a property of the *dynamics behind the field*, and we estimate it from the
measured snapshots alone -- never parsing or simulating any model.

Two complementary, model-free estimators:

A1. ``dmd_spectrum`` -- global growth spectrum via Dynamic Mode Decomposition.
    Fit the best-fit linear propagator ``x_{t+1} ~ A x_t`` from snapshots,
    map its discrete eigenvalues to continuous-time rates ``mu = log(lam)/dt``.
    The spectral abscissa ``alpha = max Re mu`` is the least-stable growth
    rate. A *rank-robustness sweep* recomputes ``alpha`` over SVD ranks: if it
    keeps climbing as finer modes are admitted, that is the data-side signature
    of high-frequency blow-up (ill-posedness); if it saturates, the resolved
    dynamics are well-posed. Works for PDEs (state = flattened space) and ODEs
    (state = the stacked variables).

A2. ``dispersion_from_data`` -- empirical dispersion relation for PDEs. FFT the
    field over space, then for each wavenumber ``k`` fit the dominant temporal
    mode ``u_hat(k, t) ~ exp(s(k) t)`` by a one-mode (rank-1) least-squares
    propagator. ``Re s(k)`` is the per-wavenumber growth rate; its high-``k``
    trend exponent ``p`` (``Re s ~ |k|^p``) classifies the operator:
    ``p<=0`` & ``Re s -> -inf`` parabolic; ``p<=0`` & ``Re s ~ 0``
    hyperbolic/dispersive; ``p>0`` ILL-POSED.

The Petrowsky/Hadamard criterion read off the data: the Cauchy problem is
well-posed iff ``Re s(k)`` stays bounded above as ``|k|`` grows.

Honest limits (see the writeup): the data only resolves wavenumbers up to the
grid Nyquist ``k_max = pi/dx``, so well-posedness beyond the grid scale is an
*extrapolation* of the high-``k`` trend, never a certificate; a single
trajectory mostly excites the least-stable modes (fine for the ``sup``, weak
for the strongly-damped tail); DMD is a linear/Koopman approximation for
nonlinear systems.

This module is intentionally pure numpy/scipy -- no EPDE imports -- so it runs
on any field array and can be unit-tested on synthetic PDEs without the
discovery pipeline.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import sqrtm as _sqrtm
except Exception:  # pragma: no cover - scipy always present in this repo
    _sqrtm = None


# --------------------------------------------------------------------------
# Geometry helpers: turn (coords, data) into snapshots + spacings.
# --------------------------------------------------------------------------
def axis_spacing(coords, axis: int) -> float:
    """Return the (uniform) grid spacing along ``axis``.

    ``coords`` is the EPDE coordinate tuple: either a single 1-D array for an
    ODE (``(t,)``) or an ``ij``-meshgrid tuple for a PDE (each entry shaped
    like the data). Spacing is the mean successive difference of the sorted
    unique coordinate values along that axis -- robust for the uniform grids
    every thesis system uses.
    """
    c = np.asarray(coords[axis])
    if c.ndim <= 1:
        vals = np.asarray(c).ravel()
    else:
        # ij-meshgrid: coords[axis] varies only along `axis`; take a 1-D slab.
        idx = [0] * c.ndim
        idx[axis] = slice(None)
        vals = np.asarray(c[tuple(idx)]).ravel()
    diffs = np.diff(vals)
    if diffs.size == 0:
        return 1.0
    return float(np.mean(np.abs(diffs)))


def field_to_snapshots(data, coords, dim: int, variable_names):
    """Assemble DMD snapshots and the per-variable spatial field(s).

    Returns a dict with:
        ``X``            -- (state_dim, n_time) snapshot matrix (columns are
                            successive time states; multi-field / spatial data
                            flattened and stacked into the state vector).
        ``dt``           -- time spacing.
        ``dx_per_axis``  -- list of spatial spacings (empty for ODEs).
        ``fields``       -- list of (n_time, *spatial) arrays, one per variable
                            (for the dispersion estimator); ``[]`` for ODEs.
        ``n_time``       -- number of snapshots.
        ``time_axis``    -- always 0 (EPDE convention: ``dx0`` is time).

    EPDE convention: data axis 0 is time; the remaining ``dim`` axes are space.
    """
    arrays = list(data) if isinstance(data, (list, tuple)) else [data]
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]

    time_axis = 0
    dt = axis_spacing(coords, time_axis)

    if dim == 0:
        # ODE: each variable is a 1-D time series; state = stacked variables.
        cols = [a.reshape(a.shape[0], -1) for a in arrays]  # (n_time, 1) each
        n_time = cols[0].shape[0]
        X = np.hstack(cols).T  # (n_vars, n_time)
        return {'X': X, 'dt': dt, 'dx_per_axis': [], 'fields': [],
                'n_time': n_time, 'time_axis': time_axis}

    # PDE: each variable shaped (n_time, *spatial).
    n_time = arrays[0].shape[0]
    dx_per_axis = [axis_spacing(coords, ax) for ax in range(1, dim + 1)]
    # State vector: flatten spatial dims, stack variables.
    flat = [a.reshape(n_time, -1) for a in arrays]   # (n_time, n_space) each
    X = np.hstack(flat).T                            # (n_vars*n_space, n_time)
    return {'X': X, 'dt': dt, 'dx_per_axis': dx_per_axis, 'fields': arrays,
            'n_time': n_time, 'time_axis': time_axis}


# --------------------------------------------------------------------------
# A1. DMD spectral abscissa + rank-robustness.
# --------------------------------------------------------------------------
def _eig_to_rates(eigs, dt):
    """Map discrete-time eigenvalues to continuous-time rates ``log(lam)/dt``."""
    eigs = np.asarray(eigs, dtype=complex)
    # Guard against a spurious zero eigenvalue -> -inf rate.
    safe = np.where(np.abs(eigs) < 1e-300, 1e-300, eigs)
    mu = np.log(safe) / dt
    return mu


def _atil(U, s, Vh, X2, r):
    """Rank-``r`` reduced DMD operator ``Atil = U_r* X2 V_r S_r^-1`` (POD coords)."""
    Ur, sr, Vr = U[:, :r], s[:r], Vh[:r, :].conj().T
    return Ur.conj().T @ X2 @ Vr @ np.diag(1.0 / sr)


def dmd_spectrum(X, dt, ranks=None, debias=True, energy_keep=0.9999):
    """Global growth spectrum of the data via DMD.

    The reported abscissa is taken at the *energy-capturing* rank (the smallest
    rank whose singular values hold ``energy_keep`` of the total) so that
    low-energy noise directions -- whose DMD eigenvalues are unreliable and
    routinely drift off the unit circle, inflating the apparent growth of
    oscillatory/conservative systems -- are excluded. The rank-robustness sweep
    still climbs to full rank: if the abscissa keeps rising there because
    *energetic* high modes genuinely grow, that is the real ill-posedness
    signal; if it only rises on noise modes, the energy-rank abscissa stays put.

    Returns a dict:
        ``eigs``           -- discrete eigenvalues at the energy rank.
        ``growth_rates``   -- ``Re mu`` at the energy rank (sorted descending).
        ``freqs``          -- ``Im mu`` at the energy rank.
        ``abscissa``       -- ``max Re mu`` at the energy rank (forward DMD).
        ``abscissa_fb``    -- forward-backward debiased abscissa (or None).
        ``energy_rank``    -- the rank used for the reported spectrum.
        ``rank_curve``     -- list of ``(rank, abscissa)`` over the sweep.
        ``rank_verdict``   -- 'saturates' | 'rank-sensitive'.
        ``max_rank``       -- the largest rank available.
    """
    X = np.asarray(X, dtype=np.float64)
    state, n_time = X.shape
    X1, X2 = X[:, :-1], X[:, 1:]
    m = X1.shape[1]
    if min(state, m) < 1:
        raise ValueError("need at least 2 snapshots for DMD")

    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    tol = s.max() * 1e-12 if s.size else 0.0
    max_rank = max(int(np.sum(s > tol)), 1)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    energy_rank = int(np.searchsorted(energy, energy_keep) + 1)
    energy_rank = max(1, min(energy_rank, max_rank))

    if ranks is None:
        if max_rank <= 6:
            ranks = list(range(1, max_rank + 1))
        else:
            ranks = sorted(set(
                int(round(x)) for x in
                np.unique(np.linspace(2, max_rank, 6).round())
            ) | {energy_rank})

    rank_curve = []
    for r in ranks:
        r = int(min(r, max_rank))
        try:
            mu = _eig_to_rates(np.linalg.eigvals(_atil(U, s, Vh, X2, r)), dt)
            rank_curve.append((r, float(np.max(mu.real))))
        except np.linalg.LinAlgError:
            continue

    # Reported spectrum at the energy-capturing rank.
    eigs = np.linalg.eigvals(_atil(U, s, Vh, X2, energy_rank))
    mu = _eig_to_rates(eigs, dt)
    order = np.argsort(mu.real)[::-1]
    growth = mu.real[order]
    freqs = mu.imag[order]
    abscissa = float(growth[0])

    abscissa_fb = None
    if debias:
        abscissa_fb = _forward_backward_abscissa(X1, X2, dt, energy_rank)

    # Verdict: does the least-stable rate keep climbing as finer modes are added?
    rank_verdict = 'saturates'
    if len(rank_curve) >= 2:
        absc = np.array([a for _, a in rank_curve])
        spread = absc.max() - absc.min()
        scale = max(abs(absc.max()), abs(absc.min()), 1e-12)
        last_jump = absc[-1] - absc[-2]
        if last_jump > 0.05 * scale and spread > 0.5 * scale:
            rank_verdict = 'rank-sensitive'

    return {
        'eigs': eigs,
        'growth_rates': growth,
        'freqs': freqs,
        'abscissa': abscissa,
        'abscissa_fb': abscissa_fb,
        'energy_rank': energy_rank,
        'rank_curve': rank_curve,
        'rank_verdict': rank_verdict,
        'max_rank': max_rank,
    }


def _forward_backward_abscissa(X1, X2, dt, rank):
    """Forward-backward DMD abscissa (noise-bias-reduced).

    fbDMD estimates ``A_fb`` with ``A_fb^2 = A_f A_b^{-1}`` where ``A_f`` maps
    ``X1->X2`` and ``A_b`` maps ``X2->X1`` in a shared POD subspace. Its
    eigenvalues are the geometric mean of the forward and (inverse) backward
    eigenvalues, cancelling the leading noise bias. Returns the abscissa or
    ``None`` if the matrix square root is unavailable / fails.
    """
    if _sqrtm is None:
        return None
    try:
        # Shared POD basis from the stacked data so A_f, A_b are comparable.
        U, s, _ = np.linalg.svd(np.hstack([X1, X2]), full_matrices=False)
        r = int(min(rank, np.sum(s > s.max() * 1e-12)))
        r = max(r, 1)
        Ur = U[:, :r]
        Y1, Y2 = Ur.conj().T @ X1, Ur.conj().T @ X2
        Af = Y2 @ np.linalg.pinv(Y1)
        Ab = Y1 @ np.linalg.pinv(Y2)
        Afb = _sqrtm(Af @ np.linalg.pinv(Ab))
        mu = _eig_to_rates(np.linalg.eigvals(Afb), dt)
        return float(np.max(mu.real))
    except Exception:
        return None


# --------------------------------------------------------------------------
# A2. Empirical dispersion relation Re s(k) for PDEs.
# --------------------------------------------------------------------------
def _rank1_growth(series, dt):
    """One-mode propagator rate for a complex series: ``s = log(z)/dt`` with
    ``z = <c(t), c(t+1)> / <c(t), c(t)>``. Fallback for very short series."""
    c = np.asarray(series, dtype=complex)
    a, b = c[:-1], c[1:]
    denom = np.vdot(a, a)
    if abs(denom) < 1e-300:
        return np.nan + 0j
    z = np.vdot(a, b) / denom
    if not np.isfinite(z) or abs(z) < 1e-300:
        return np.nan + 0j
    return np.log(z) / dt


def _prony_growth(series, dt, order=2, amp_floor=0.05):
    """Least-stable continuous rate of a complex time series via order-``order``
    linear prediction (Prony / matrix-pencil), pruned by mode amplitude.

    Fits ``c[n] = a_1 c[n-1] + ... + a_q c[n-q]`` by least squares, takes the
    roots ``z_i`` of the characteristic polynomial, fits each root's amplitude
    in the signal, DISCARDS roots that carry negligible amplitude, and returns
    the continuous-time rate ``s = log(z)/dt`` of the least-stable *surviving*
    root -- the per-wavenumber spectral abscissa.

    Order >= 2 is essential for second-order-in-time / conservative dynamics: a
    real cosine ``cos(omega t)`` is a conjugate pair ``exp(+/- i omega t)``,
    which a one-mode fit aliases into spurious growth/decay; order 2 recovers
    ``Re s ~ 0``. The amplitude prune is essential the other way: for a single
    decaying mode the surplus root would otherwise contribute a spurious
    (possibly growing) root, so we keep only roots actually present in the data.
    """
    c = np.asarray(series, dtype=complex)
    N = c.size
    q = int(min(order, N - 2))
    if q < 1:
        return _rank1_growth(c, dt)
    cols = [c[q - 1 - i: N - 1 - i] for i in range(q)]
    H = np.column_stack(cols)            # (N-q, q): predictors c[n-1..n-q]
    y = c[q:N]                           # targets   c[n]
    try:
        a, *_ = np.linalg.lstsq(H, y, rcond=None)
    except np.linalg.LinAlgError:
        return _rank1_growth(c, dt)
    roots = np.roots(np.r_[1.0, -a])     # z^q - a1 z^{q-1} - ... - aq = 0
    roots = roots[np.abs(roots) > 1e-300]
    if roots.size == 0:
        return _rank1_growth(c, dt)

    # Amplitude prune: fit c ~ sum_i amp_i z_i^n. A spurious root with huge |z|
    # overflows z^n -> its Vandermonde column is zeroed -> amplitude ~ 0 -> it
    # is dropped, so overflow is self-correcting rather than fatal.
    nidx = np.arange(N)
    with np.errstate(over='ignore', invalid='ignore'):
        V = np.power(roots[None, :].astype(complex), nidx[:, None])
    V[~np.isfinite(V)] = 0.0
    try:
        amp, *_ = np.linalg.lstsq(V, c, rcond=None)
        mag = np.abs(amp)
        keep = mag > amp_floor * mag.max() if mag.max() > 0 else \
            np.ones(roots.size, dtype=bool)
    except np.linalg.LinAlgError:
        keep = np.ones(roots.size, dtype=bool)
    roots = roots[keep]
    if roots.size == 0:
        return _rank1_growth(c, dt)
    s = np.log(roots) / dt
    s = s[np.isfinite(s.real)]
    if s.size == 0:
        return _rank1_growth(c, dt)
    return s[int(np.argmax(s.real))]


def dispersion_from_data(field, dx_per_axis, dt, horizon=None,
                         kfrac_lo=0.5, energy_floor=1e-10, prony_order=2):
    """Empirical dispersion relation ``s(k)`` from a single field.

    ``field`` is (n_time, *spatial). FFT over the spatial axes; for every
    spatial wavenumber bin fit the dominant temporal mode and record its
    growth rate ``Re s`` and frequency ``Im s``. Wavenumbers are binned by
    ``|k|`` (magnitude) so multi-dimensional spatial grids collapse to one
    dispersion curve.

    Returns a dict:
        ``k``             -- magnitude grid (rad / unit length).
        ``Re_s``          -- growth rate at each ``k`` (energy-weighted over
                             the bin).
        ``Im_s``          -- frequency at each ``k``.
        ``k_max``         -- resolved Nyquist (min over axes).
        ``sup_Re_s``      -- max ``Re_s`` over the resolved band.
        ``highk_exponent``-- slope ``p`` of ``log|Re_s|`` vs ``log k`` on the
                             upper band (``k > kfrac_lo * k_max``).
        ``highk_mean``    -- mean ``Re_s`` over the upper band (sign = growing
                             vs decaying at high k).
        ``amplification`` -- ``max_k exp(Re_s * horizon)`` (horizon defaults to
                             the full observed time span).
    """
    field = np.asarray(field, dtype=np.float64)
    n_time = field.shape[0]
    spatial_axes = tuple(range(1, field.ndim))
    if not spatial_axes:
        raise ValueError("dispersion needs at least one spatial axis")
    if horizon is None:
        horizon = (n_time - 1) * dt

    # Remove the temporal mean field (k-DC drift is not a dispersion mode).
    fluct = field - field.mean(axis=0, keepdims=True)

    # Spatial FFT -> u_hat(k, t): move time to the front, fft the rest.
    uhat = np.fft.fftn(fluct, axes=spatial_axes)  # (n_time, *kspatial)

    # Wavenumber magnitude grid.
    kgrids = []
    for ax, dx in zip(spatial_axes, dx_per_axis):
        n = field.shape[ax]
        kgrids.append(2.0 * np.pi * np.fft.fftfreq(n, d=dx))
    kmesh = np.meshgrid(*kgrids, indexing='ij')
    kmag = np.sqrt(np.sum([km ** 2 for km in kmesh], axis=0))  # (*kspatial,)
    k_max = float(min(np.max(np.abs(kg)) for kg in kgrids))

    # Per spatial mode: dominant temporal growth rate + energy.
    n_modes = kmag.size
    uflat = uhat.reshape(n_time, n_modes)
    kflat = kmag.reshape(n_modes)
    energy = np.sum(np.abs(uflat) ** 2, axis=0)
    s_per_mode = np.full(n_modes, np.nan + 0j, dtype=complex)
    emax = energy.max() if energy.size else 0.0
    for j in range(n_modes):
        if energy[j] <= energy_floor * emax or kflat[j] == 0.0:
            continue
        s = _prony_growth(uflat[:, j], dt, order=prony_order)
        if np.isfinite(s.real):
            s_per_mode[j] = s

    # Bin by |k| (energy-weighted growth/frequency per bin).
    valid = np.isfinite(s_per_mode.real) & (kflat > 0)
    if not np.any(valid):
        return {'k': np.array([]), 'Re_s': np.array([]), 'Im_s': np.array([]),
                'k_max': k_max, 'sup_Re_s': np.nan, 'highk_exponent': np.nan,
                'highk_mean': np.nan, 'amplification': np.nan,
                'obs_cycles': np.nan, 't_obs': (n_time - 1) * dt}

    kv, sv, ev = kflat[valid], s_per_mode[valid], energy[valid]
    # Bin over the populated (energetic) wavenumber range, not the raw Nyquist:
    # band-limited real fields carry no energy near Nyquist, so binning to
    # k_max would scatter a handful of energetic modes into the first bin.
    k_hi = float(kv.max())
    nbins = max(6, int(np.sqrt(kv.size)))
    edges = np.linspace(0, k_hi * (1 + 1e-6), nbins + 1)
    centers, re_s, im_s, bin_energy = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (kv >= lo) & (kv < hi)
        if not np.any(sel):
            continue
        w = ev[sel]
        w = w / w.sum()
        centers.append(0.5 * (lo + hi))
        re_s.append(float(np.sum(w * sv[sel].real)))
        im_s.append(float(np.sum(w * np.abs(sv[sel].imag))))
        bin_energy.append(float(ev[sel].sum()))
    k = np.array(centers)
    re_s = np.array(re_s)
    im_s = np.array(im_s)
    bin_energy = np.array(bin_energy)

    sup_re_s = float(np.max(re_s))
    amplification = float(np.max(np.exp(np.clip(re_s * horizon, -50, 50))))

    # Temporal-sampling confidence: how many oscillation periods of the
    # (energy-weighted) dynamics are actually observed. A growth rate cannot be
    # separated from a transient if the field is sampled over a fraction of a
    # period -- so this gates the classifier's confidence downstream.
    t_obs = (n_time - 1) * dt
    cyc = np.abs(im_s) * t_obs / (2.0 * np.pi)
    obs_cycles = (float(np.sum(bin_energy * cyc) / bin_energy.sum())
                  if bin_energy.sum() > 0 else np.nan)

    # High-k trend: slope of log|Re_s| vs log k on the upper band. The band is
    # taken relative to the largest *populated* (energetic) wavenumber, not the
    # raw Nyquist -- band-limited real data (smooth solitons) carries no energy
    # near Nyquist, so judging the trend there would read pure noise.
    k_top = float(k.max()) if k.size else k_max
    hi_band = k > kfrac_lo * k_top
    highk_exponent = np.nan
    highk_mean = np.nan
    if np.count_nonzero(hi_band) >= 2:
        kk = k[hi_band]
        rr = re_s[hi_band]
        highk_mean = float(np.mean(rr))
        mag = np.abs(rr)
        good = mag > 1e-12
        if np.count_nonzero(good) >= 2:
            p = np.polyfit(np.log(kk[good]), np.log(mag[good]), 1)
            highk_exponent = float(p[0])

    return {'k': k, 'Re_s': re_s, 'Im_s': im_s, 'k_max': k_max,
            'sup_Re_s': sup_re_s, 'highk_exponent': highk_exponent,
            'highk_mean': highk_mean, 'amplification': amplification,
            'obs_cycles': obs_cycles, 't_obs': t_obs}


# --------------------------------------------------------------------------
# A4. Classification + orchestration.
# --------------------------------------------------------------------------
def classify_wellposedness(dispersion, *, growth_tol=None, k_ref=None):
    """Classify the operator type from an empirical dispersion result.

    Petrowsky/Hadamard criterion: well-posed iff ``Re s(k)`` stays bounded
    above as ``|k|`` grows. We read that off the upper-band trend:

        growing at high k (``highk_mean`` > tol, exponent > 0) -> ILL-POSED;
        strongly decaying (``Re s`` very negative, exponent ~ 2)-> parabolic;
        bounded near zero                                       -> hyperbolic /
                                                                   dispersive
            (hyperbolic if ``Im s ~ k`` linear, dispersive if super-linear);
        otherwise                                               -> mixed.

    ``growth_tol`` defaults to a small fraction of the inverse time-scale set
    by the spectrum's own magnitude, so the zero-band test is scale-aware.
    """
    re_s = np.asarray(dispersion.get('Re_s', []))
    im_s = np.asarray(dispersion.get('Im_s', []))
    k = np.asarray(dispersion.get('k', []))
    if re_s.size == 0:
        return {'type': 'unknown', 'wellposed': None,
                'reason': 'no resolved dispersion modes'}

    # Two scales. ``growth_tol`` (growth-rate based) gates the ILL-POSED test:
    # a mode counts as "growing" only if its rate is an appreciable fraction of
    # the spectrum's own growth magnitude. ``zero_tol`` (dynamics based, also
    # counting the oscillation rate |Im s|) gates the "Re s ~ 0" test, so a
    # purely oscillatory hyperbolic/dispersive operator -- large |Im s|, tiny
    # |Re s| -- is correctly read as non-growing rather than "mixed".
    re_scale = float(np.median(np.abs(re_s))) + 1e-12
    finite_im = im_s[np.isfinite(im_s)]
    freq_scale = float(np.max(np.abs(finite_im))) if finite_im.size else 0.0
    dyn_scale = max(re_scale, freq_scale)
    if growth_tol is None:
        growth_tol = 0.1 * re_scale
    zero_tol = 0.05 * dyn_scale

    highk_mean = dispersion.get('highk_mean', np.nan)
    p = dispersion.get('highk_exponent', np.nan)
    sup_re = dispersion.get('sup_Re_s', np.nan)

    # Temporal-sampling confidence. If the dynamics are oscillatory (freq scale
    # dominates the growth scale) yet the field is observed for under half a
    # period, no method can separate a growth rate from the transient -- so any
    # non-well-posed verdict below is downgraded to 'undetermined' rather than a
    # false ill-posed. (Purely decaying/growing data, Im s ~ 0, is unaffected:
    # its rate is estimable from a fraction of an e-fold.)
    obs_cycles = dispersion.get('obs_cycles', np.nan)
    oscillatory = freq_scale > re_scale
    undersampled = (oscillatory and np.isfinite(obs_cycles)
                    and obs_cycles < 0.5)

    # Upper band relative to the largest populated wavenumber (see dispersion).
    k_top = float(k.max()) if k.size else np.inf
    highk_band = k > 0.5 * k_top
    re_hi = re_s[highk_band] if np.any(highk_band) else re_s

    # 1. Ill-posed: growth that increases with wavenumber (fails the Petrowsky
    #    criterion sup_k Re s < inf). Gated on the growth-rate scale.
    growing = (np.isfinite(highk_mean) and highk_mean > growth_tol and
               sup_re > growth_tol)
    if growing and (not np.isfinite(p) or p > 0.3):
        if undersampled:
            return {'type': 'undetermined', 'wellposed': None,
                    'reason': f'apparent high-k growth but only '
                              f'{obs_cycles:.2g} oscillation periods observed; '
                              f'growth rate not separable from transient'}
        return {'type': 'ill-posed', 'wellposed': False,
                'reason': f'Re s grows with k (highk_mean={highk_mean:.3g}, '
                          f'p={p:.2g}); fails sup_k Re s < inf'}

    # 2. Bounded ~ 0 (relative to the oscillation rate) -> well-posed, and
    #    purely conservative. Check this BEFORE parabolic so a tiny noisy Re s
    #    on top of a fast oscillation is not mistaken for diffusion. Hyperbolic
    #    if Im s ~ |k| (linear), dispersive if super-linear.
    near_zero = np.all(np.abs(re_hi) < zero_tol)
    if near_zero:
        kind = 'hyperbolic'
        good = (k > 0) & np.isfinite(im_s) & (np.abs(im_s) > 1e-12)
        q = np.nan
        if np.count_nonzero(good) >= 2:
            q = np.polyfit(np.log(k[good]), np.log(np.abs(im_s[good])), 1)[0]
            if q > 1.4:
                kind = 'dispersive'
        return {'type': kind, 'wellposed': True,
                'reason': f'Re s bounded ~0; Im s ~ |k|^{q:.2g} '
                          f'(well-posed {kind})'}

    # 3. Parabolic: strong high-k damping (Re s -> -inf like |k|^2), where the
    #    damping is dynamically significant (below -zero_tol), not just noise.
    strongly_decaying = (np.isfinite(p) and p > 1.3 and
                         np.mean(re_hi) < -zero_tol)
    if strongly_decaying:
        return {'type': 'parabolic', 'wellposed': True,
                'reason': f'Re s -> -inf like |k|^{p:.2g} (diffusive)'}

    if undersampled:
        return {'type': 'undetermined', 'wellposed': None,
                'reason': f'mixed-sign Re s but only {obs_cycles:.2g} '
                          f'oscillation periods observed; insufficient '
                          f'temporal sampling for a forward verdict'}
    return {'type': 'mixed', 'wellposed': bool(sup_re <= growth_tol),
            'reason': f'mixed signs (sup Re s={sup_re:.3g}, p={p})'}


def analyze_forward(data, coords, dim, variable_names, *, horizon=None,
                    debias=True):
    """Full data-only forward well-posedness analysis for one system.

    Runs DMD (A1) always, plus the empirical dispersion (A2) + classification
    (A4) when the field has a spatial axis. For multi-field PDEs the dispersion
    is computed on the first (primary) variable; DMD always uses the full
    stacked state.

    Returns a JSON-friendly dict (numpy scalars cast to float; arrays dropped
    or downcast to lists) summarising both estimators.
    """
    snaps = field_to_snapshots(data, coords, dim, variable_names)
    dmd = dmd_spectrum(snaps['X'], snaps['dt'], debias=debias)

    out = {
        'dim': int(dim),
        'n_time': int(snaps['n_time']),
        'dt': float(snaps['dt']),
        'dx_per_axis': [float(d) for d in snaps['dx_per_axis']],
        'dmd_abscissa': dmd['abscissa'],
        'dmd_abscissa_fb': dmd['abscissa_fb'],
        'dmd_rank_verdict': dmd['rank_verdict'],
        'dmd_rank_curve': dmd['rank_curve'],
        'dmd_energy_rank': dmd['energy_rank'],
        'dmd_max_rank': dmd['max_rank'],
    }

    if dim >= 1 and snaps['fields']:
        disp = dispersion_from_data(snaps['fields'][0], snaps['dx_per_axis'],
                                    snaps['dt'], horizon=horizon)
        cls = classify_wellposedness(disp)
        out.update({
            'dispersion_k': disp['k'].tolist(),
            'dispersion_Re_s': disp['Re_s'].tolist(),
            'dispersion_Im_s': disp['Im_s'].tolist(),
            'k_max': disp['k_max'],
            'sup_Re_s': disp['sup_Re_s'],
            'highk_exponent': disp['highk_exponent'],
            'highk_mean': disp['highk_mean'],
            'amplification': disp['amplification'],
            'obs_cycles': disp['obs_cycles'],
            'type': cls['type'],
            'wellposed': cls['wellposed'],
            'reason': cls['reason'],
            'caveat': 'resolved up to Nyquist k_max only; high-k behaviour is '
                      'extrapolated, not certified',
        })
    else:
        # ODE: finite-dimensional -> locally well-posed (Picard-Lindelof);
        # the abscissa reports Lyapunov-type IC sensitivity, not ill-posedness.
        out.update({
            'type': 'ode',
            'wellposed': True,
            'reason': 'finite-dim ODE; DMD abscissa = Lyapunov-type IC '
                      'sensitivity (alpha>0 => exponential separation)',
        })
    return out
