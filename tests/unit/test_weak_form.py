"""The weak-form engine (epde/preprocessing/weak_form.py): integration by parts, exact-moment kernels, window
integrals, the AR(1) noise model and the pcr window choice."""
import os

import numpy as np
import pytest

from epde.preprocessing import weak_form as WF


def _key(data, coords):
    return (tuple(sorted(data)), tuple(coords), (0,) * len(coords))


# ------------------------------------------------------------------------------------------------- IBP
def test_reduce_solves_the_self_reproducing_step():
    # u u_x = (u^2 / 2)_x  ->  -1/2 u^2 psi_x
    out = WF.reduce(_key([('u', (0,)), ('u', (1,))], (0,)))
    assert out == {((('u', (0,)), ('u', (0,))), (0,), (1,)): -0.5}


def test_reduce_keeps_a_non_divergence_remainder():
    key = _key([('u', (1,)), ('u', (1,))], (0,))
    assert WF.reduce(key) == {key: 1.0}


def test_reduce_moves_every_derivative_of_a_linear_term():
    out = WF.reduce(_key([('u', (0, 3))], (0, 0)))
    assert out == {((('u', (0, 0)),), (0, 0), (0, 3)): -1.0}


def test_reduce_respects_the_psi_cap():
    key = _key([('u', (3,))], (0,))
    capped = WF.reduce(key, psi_cap=(1,))
    assert all(k[2][0] <= 1 for k in capped)
    assert WF.max_order(next(iter(capped))) == 2          # one derivative moved, two kept pointwise


def _analytic_fields(n=(161, 201)):
    t = np.linspace(0.0, 1.0, n[0])
    x = np.linspace(0.0, 2 * np.pi, n[1])
    T, X = np.meshgrid(t, x, indexing='ij')
    a, k = 0.3, 2.0
    u = np.exp(-a * k * k * T) * np.sin(k * X) + 0.5 * np.cos(X) * np.exp(-a * T)
    return WF.Fields({'u': u}, [t, x]), a


@pytest.mark.parametrize('term', [
    WF.WeakTerm([('u', (0, 0)), ('u', (0, 1))], (0, 0)),          # u u_x
    WF.WeakTerm([('u', (0, 2))], (0, 1)),                          # x u_xx
    WF.WeakTerm([('u', (1, 0)), ('u', (0, 0)), ('u', (0, 0))], (0, 0)),   # u^2 u_t
    WF.WeakTerm([('u', (0, 1)), ('u', (0, 2))], (0, 0)),          # u_x u_xx
])
def test_ibp_columns_equal_the_pointwise_columns(term):
    """The IBP form and the plain pointwise form integrate the same function: equal to quadrature error."""
    F, _ = _analytic_fields()
    ms, ss, q, cut = (8, 8), (8, 8), (4, 4), (12, 12)
    weak, _ = WF.weak_matrix([term.monomials()], [()], F, cut, ms, ss, q)
    strong_monos = {(term.data, term.coords, (0, 0)): 1.0}
    strong, _ = WF.weak_matrix([strong_monos], [()], F, cut, ms, ss, q)
    assert np.max(np.abs(weak - strong)) <= 1e-3 * np.max(np.abs(strong))


# ----------------------------------------------------------------------------------------------- kernels
@pytest.mark.parametrize('m,q,g', [(2, 2, 0), (4, 3, 1), (8, 4, 2), (16, 5, 3)])
def test_exact_kernel_reproduces_the_legendre_moments(m, q, g):
    from numpy.polynomial import polynomial as Pl, legendre as L
    w = WF.exact_kernel(m, q, g)
    s = np.arange(-m, m + 1) / m
    c = Pl.polypow([1.0, 0.0, -1.0], q)
    c = Pl.polyder(c, g) if g else c
    for j in range(min(2 * m, 2 * q) + 1):
        prod_ = Pl.polyint(Pl.polymul(c, L.leg2poly(np.eye(j + 1)[j])))
        exact = Pl.polyval(1.0, prod_) - Pl.polyval(-1.0, prod_)
        assert abs(w @ L.legval(s, np.eye(j + 1)[j]) - exact) < 1e-10


def test_kernels_are_normalised_to_unit_mass():
    K = WF.window_kernels((4, 16), (0.1, 0.02), (3, 5), (2, 2))
    for Kd in K:
        assert abs(Kd[0].sum() - 1.0) < 1e-12


def test_a_constant_has_the_weak_column_one():
    F, _ = _analytic_fields()
    const = WF.WeakTerm([], (0, 0))
    M, shape = WF.weak_matrix([const.monomials()], [()], F, (4, 4), (8, 16), (4, 8), (3, 3))
    assert M.shape == (int(np.prod(shape)), 1)
    assert np.max(np.abs(M - 1.0)) < 1e-12


def test_wint_matches_a_brute_force_sum():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((40, 33))
    ms, ss = (4, 2), (3, 1)
    K = WF.window_kernels(ms, (1.0, 1.0), (3, 3), (1, 1))
    out = WF.wint_s(f, (1, 0), ms, ss, K)
    shape = WF.lattice_shape(f.shape, ms, ss)
    assert out.shape == shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            patch = f[i * ss[0]:i * ss[0] + 2 * ms[0] + 1, j * ss[1]:j * ss[1] + 2 * ms[1] + 1]
            assert abs(out[i, j] - K[0][1] @ patch @ K[1][0]) < 1e-12


# ------------------------------------------------------------------------------------- weak identity
def test_weak_rows_of_the_heat_equation_recover_its_coefficient():
    F, a = _analytic_fields()
    target = WF.WeakTerm([('u', (1, 0))], (0, 0))
    feat = WF.WeakTerm([('u', (0, 2))], (0, 0))
    M, _ = WF.weak_matrix([target.monomials(), feat.monomials()], [(), ()], F, (0, 0), (8, 8), (4, 4), (3, 3))
    c = np.linalg.lstsq(M[:, 1:], M[:, 0], rcond=None)[0][0]
    assert abs(c - a) < 1e-4 * a


def test_opaque_factor_is_integrated_pointwise():
    t = np.linspace(0, 10, 1001)
    F = WF.Fields({'u': np.sin(t)}, [t], opaque={'cos': np.cos(t)})
    term = WF.WeakTerm([('u', (0,))], (0,), opaque=('cos',))
    assert term.monomials() == {((('u', (0,)),), (0,), (0,)): 1.0}
    M, _ = WF.weak_matrix([term.monomials()], [term.opaque], F, (0,), (16,), (16,), (2,))
    product_field = WF.Fields({'w': np.sin(t) * np.cos(t)}, [t])
    ref, _ = WF.weak_matrix([WF.WeakTerm([('w', (0,))], (0,)).monomials()], [()], product_field, (0,), (16,),
                            (16,), (2,))
    assert np.allclose(M, ref, rtol=1e-13, atol=1e-15)


def test_a_column_beyond_the_cut_is_refused():
    term = WF.WeakTerm([('u', (1,)), ('u', (1,))], (0,))
    with pytest.raises(ValueError, match='edge cut'):
        WF.check_cut(term.monomials(), 1, (1,))


# --------------------------------------------------------------------------------------------- noise
def test_noise_model_recovers_ar1_noise():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 20, 20000)
    clean = np.sin(t) + 0.3 * np.sin(3.1 * t)
    noise = WF.ar_noise(rng, t.shape, 0.05, [0.5])
    sig, rhos, _ = WF.noise_model(clean + noise)
    assert abs(sig / 0.05 - 1) < 0.1
    assert abs(rhos[0] - 0.5) < 0.1


def test_noise_model_finds_no_noise_on_a_clean_polynomial():
    t = np.linspace(0, 1, 400)
    sig, _, _ = WF.noise_model(1 + t + t ** 2)
    assert sig < 1e-6


# ----------------------------------------------------------------------------------------- selection
def _lv_fields(noise, seed=0, n=601):
    from scipy.integrate import solve_ivp
    t = np.linspace(0, 12, n)
    sol = solve_ivp(lambda _, z: [z[0] * (1.0 - 0.5 * z[1]), z[1] * (0.25 * z[0] - 0.75)], (0, 12), [2.0, 1.0],
                    t_eval=t, rtol=1e-11, atol=1e-12).y
    rng = np.random.default_rng(seed)
    vals = {k: v + noise * v.std() * rng.standard_normal(v.shape) for k, v in zip('uv', sol)}
    return WF.Fields(vals, [t])


def _lv_library():
    terms = [WF.WeakTerm([('u', (1,))], (0,)), WF.WeakTerm([('v', (1,))], (0,))]
    for p in (1, 2):
        terms += [WF.WeakTerm([('u', (0,))] * p, (0,)), WF.WeakTerm([('v', (0,))] * p, (0,))]
    terms.append(WF.WeakTerm([('u', (0,)), ('v', (0,))], (0,)))
    return terms


def test_choose_window_is_deterministic_and_feasible():
    F = _lv_fields(0.02)
    lib = _lv_library()
    plans = [WF.choose_window(F, lib, [0], np.random.default_rng(7)) for _ in range(2)]
    assert repr(plans[0]) == repr(plans[1])
    plan = plans[0]
    assert np.isfinite(plan.score) and plan.rows > len(lib) + 1
    assert plan.q == tuple(r + plan.j for r in plan.r)
    assert all(ss <= ms for ss, ms in zip(plan.ss, plan.ms))


def test_choose_window_respects_the_row_floor():
    F = _lv_fields(0.02)
    lib = _lv_library()
    plan = WF.choose_window(F, lib, [0], np.random.default_rng(7), rows_ok=lambda shape: shape[0] >= 200)
    assert plan.rows >= 200
    with pytest.raises(ValueError, match='no feasible'):
        WF.choose_window(F, lib, [0], np.random.default_rng(7), rows_ok=lambda shape: False)


def test_pcr_window_fit_recovers_lotka_volterra():
    F = _lv_fields(0.01)
    lib = _lv_library()
    plan = WF.choose_window(F, lib, [0], np.random.default_rng(7))
    feats = [2, 6]                                           # u, u v
    M, _ = WF.weak_matrix([lib[k].monomials() for k in [0] + feats], [()] * 3, F, plan.cut, plan.ms, plan.ss,
                          plan.q)
    c = np.linalg.lstsq(M[:, 1:], M[:, 0], rcond=None)[0]
    assert np.allclose(c, [1.0, -0.5], rtol=0.05)


# ------------------------------------------------------------------------------------------ spin heading
def _spin_heading():
    path = os.path.join(os.path.dirname(__file__), 'data', 'weak_noise_spin_heading.npy')
    return np.load(path).astype(float) / 1e4


def test_noise_model_stays_within_the_data_variance():
    """sin(heading) of a spinning vibrobot at 25 frames/s (8250 samples, int16 x 1e4). Before the variance guard
    the scan accepted order 196 at rho -> 1 and returned (431.3, [0.99999999888], [196])."""
    y = _spin_heading()
    sig, rhos, ms = WF.noise_model(y)
    assert sig ** 2 <= np.var(y)
    assert sig > 0


def test_choose_window_on_a_spin_heading_is_feasible():
    """Before the variance guard this raised RuntimeError (J_GUARD reached)."""
    y = _spin_heading()
    t = np.arange(y.size) * 0.04
    fields = WF.Fields({'s': y}, [t])
    library = [WF.WeakTerm([('s', (k,))], (0,)) for k in (0, 1, 2)] + [WF.WeakTerm([], (1,))]
    plan = WF.choose_window(fields, library, [1, 2], np.random.default_rng([0, 0]))
    assert plan.j < WF.J_GUARD
    assert np.isfinite(plan.score)
