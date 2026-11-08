"""SpectralDeriv: Fourier derivatives that are exact on periodic axes (found from the data by
default), boundary-corrected on the others, unfiltered unless ``n`` is given, laid out like
``define_derivatives``.

Before the fix the default preprocessor (``n=None``) raised TypeError, an int ``n`` raised
IndexError on unequal axes, ``mixed=True`` used the removed ``np.math``, the "unfiltered" default
still damped every mode, and non-periodic samples (most EPDE data) came out with O(1) errors
because the FFT wraps the jump between the two ends into the derivative.
"""
import numpy as np
import pytest

from epde.preprocessing.deriv_calculators import AdaptiveFiniteDeriv, SpectralDeriv


def _rel(est, ref, crop=0):
    if crop:
        sl = tuple(slice(crop, -crop) for _ in range(np.ndim(ref)))
        est, ref = est[sl], ref[sl]
    return float(np.max(np.abs(est - ref)) / np.max(np.abs(ref)))


def test_a_periodic_axis_is_exact():
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    out = SpectralDeriv()(np.sin(3 * t), [t], 2, periodic=True)
    assert out.shape == (128, 2)
    assert _rel(out[:, 0], 3 * np.cos(3 * t)) < 1e-12
    assert _rel(out[:, 1], -9 * np.sin(3 * t)) < 1e-11


def test_the_default_handles_non_periodic_samples():
    t = np.linspace(0, 1, 256)
    f = np.exp(2 * t)
    out = SpectralDeriv()(f, [t], 2)
    fd = AdaptiveFiniteDeriv()(f, [t], 2)
    for k, ref in enumerate((2 * f, 4 * f)):
        assert _rel(out[:, k], ref, crop=25) < 1e-5
        assert _rel(out[:, k], ref, crop=25) < 0.1 * _rel(fd[:, k], ref, crop=25)
    assert _rel(out[:, 0], 2 * f) < 1e-5            # the ends too


def test_columns_follow_define_derivatives_and_axes_take_their_own_flags():
    T, X = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 2 * np.pi, 96, endpoint=False),
                       indexing='ij')
    u = np.exp(T) * np.cos(2 * X)
    refs = [u, -2 * np.exp(T) * np.sin(2 * X), -4 * u]          # u_t, u_x, u_xx
    sd = SpectralDeriv()
    out = sd(u, [T, X], [1, 2], periodic=[False, True])
    assert out.shape == (u.size, 3)
    errs = [_rel(out[:, k].reshape(u.shape), r, crop=6) for k, r in enumerate(refs)]
    assert errs[0] < 1e-5 and errs[1] < 1e-12 and errs[2] < 1e-11
    # 1-D coordinate vectors are the same grid as the meshgrid
    np.testing.assert_array_equal(out, sd(u, [T[:, 0], X[0]], [1, 2], periodic=[False, True]))
    # an axis with order 0 contributes no column
    only_x = sd(u, [T, X], [0, 2], periodic=[False, True])
    np.testing.assert_array_equal(only_x, out[:, 1:])


def test_no_filter_unless_n_is_given():
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    sd = SpectralDeriv()
    h = t[1] - t[0]
    # n counts distinct non-negative frequencies: the 4th is the 3rd harmonic
    assert sd.cutoff_frequency(4, 128, h) == pytest.approx(3 / (128 * h))
    assert sd.cutoff_frequency(10 ** 6, 128, h) == pytest.approx(0.5 / h)    # clipped to Nyquist
    assert sd.cutoff_frequency(None, 128, h) is None
    out = sd(np.sin(3 * t), [t], 1, n=4, steepness=1, periodic=True)
    np.testing.assert_allclose(out[:, 0], 0.5 * 3 * np.cos(3 * t), atol=1e-12)   # half power at f_n


def test_an_int_n_works_on_unequal_axes():
    T, X = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 48), indexing='ij')
    assert SpectralDeriv()(np.sin(T + X), [T, X], 1, n=40).shape == (32 * 48, 2)


@pytest.mark.parametrize('kwargs, error', [
    (dict(mixed=True), NotImplementedError),
    (dict(n=1), ValueError),
    (dict(steepness=0), ValueError),
    (dict(periodic=[True, False, True]), ValueError),
])
def test_bad_arguments_are_refused(kwargs, error):
    t = np.linspace(0, 1, 64)
    with pytest.raises(error):
        SpectralDeriv()(np.sin(t), [t], 1, **kwargs)


def test_the_grid_has_to_fit_the_data_and_short_axes_need_periodic():
    T, X = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 4), indexing='ij')
    sd = SpectralDeriv()
    with pytest.raises(ValueError):
        sd(np.sin(T), [T[:, 0]], 1)
    with pytest.raises(ValueError):
        sd(np.sin(T), [T, X], 1, periodic=False)         # 4 samples on a non-periodic axis
    assert sd(np.sin(T), [T, X], 1, periodic=[False, True]).shape == (64, 2)


# ---------------------------------------------------------------------------
# periodic='auto' (the default): data_profile's seam test, axis by axis
# ---------------------------------------------------------------------------
def test_auto_reads_both_endpoint_conventions_and_stays_exact():
    sd = SpectralDeriv()
    x = np.linspace(0, 2 * np.pi, 128, endpoint=False)                 # FFT convention
    out = sd(np.sin(3 * x), [x], 2)
    assert sd.last_periodic == [True]
    assert _rel(out[:, 0], 3 * np.cos(3 * x)) < 1e-12
    xi = np.linspace(0, 2 * np.pi, 129)                                # last sample = first
    out = sd(np.sin(3 * xi), [xi], 2)
    assert sd.last_periodic == ['included']
    assert _rel(out[:, 0], 3 * np.cos(3 * xi)) < 1e-12
    assert _rel(out[:, 1], -9 * np.sin(3 * xi)) < 1e-11


def test_auto_calls_non_periodic_samples_non_periodic():
    sd = SpectralDeriv()
    t = np.linspace(0, 1, 256)
    sd(np.exp(2 * t), [t], 1)
    assert sd.last_periodic == [False]
    # value-periodic (zero at both ends) but not slope-periodic: the Dirichlet standing wave
    T, X = np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, 1, 81), indexing='ij')
    sd(np.sin(np.pi * X) * np.cos(np.pi * T), [T, X], [2, 2])
    assert sd.last_periodic == [False, False]


def test_auto_decides_each_axis_on_its_own():
    T, X = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 2 * np.pi, 96, endpoint=False),
                       indexing='ij')
    u = np.exp(T) * np.cos(2 * X)
    sd = SpectralDeriv()
    auto = sd(u, [T, X], [1, 2])
    assert sd.last_periodic == [False, True]
    np.testing.assert_array_equal(auto, sd(u, [T, X], [1, 2], periodic=[False, True]))
    mixed = sd(u, [T, X], [1, 2], periodic=['auto', False])           # per-axis mix
    assert sd.last_periodic == [False, False]
    np.testing.assert_array_equal(mixed[:, 0], auto[:, 0])


def test_an_unknown_periodic_value_is_refused():
    t = np.linspace(0, 1, 64)
    with pytest.raises(ValueError):
        SpectralDeriv()(np.sin(t), [t], 1, periodic='yes')


@pytest.mark.parametrize('dataset', ['ac', 'burgers'])
def test_auto_finds_the_periodic_axis_of_the_repository_data(dataset):
    import os
    data = os.path.join(os.path.dirname(__file__), '..', '..', 'projects', 'pic', 'data')
    if dataset == 'ac':
        path = os.path.join(data, 'ac', 'ac_data.npy')
        if not os.path.exists(path):
            pytest.skip('ac_data.npy not present')
        u = np.load(path)
        t, x = np.linspace(0.0, 1.0, 51), np.linspace(-1.0, 0.984375, 128)
    else:
        path = os.path.join(data, 'burgers', 'burgers.mat')
        if not os.path.exists(path):
            pytest.skip('burgers.mat not present')
        from scipy.io import loadmat
        m = loadmat(path)
        t, x, u = np.ravel(m['t']), np.ravel(m['x']), np.real(m['usol']).T
    sd = SpectralDeriv()
    sd(u, [t, x], [1, 2])
    assert sd.last_periodic == [False, True]


def test_the_spectral_preprocessor_runs_with_its_defaults():
    from epde.interface.interface import EpdeSearch
    t = np.linspace(0, 4, 400)
    u = np.sin(2 * t) + 0.3 * t
    search = EpdeSearch(use_solver=False, multiobjective_mode=True,
                        verbose_params={'show_iter_idx': False}, device='cpu')
    _, domain = search.createDomain((t,), boundary_width=20, ID=0)
    search.set_preprocessor(default_preprocessor_type='spectral', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': u}, domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(2,), data_fun_pow=1, additional_tokens=[])
    derivs = np.asarray(trajectory._entries[0].derivatives)
    assert derivs.shape[1] == 2
    assert _rel(derivs[:, 0], 2 * np.cos(2 * t) + 0.3, crop=20) < 1e-6
    assert _rel(derivs[:, 1], -4 * np.sin(2 * t), crop=20) < 1e-5
