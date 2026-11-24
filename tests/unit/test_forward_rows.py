"""Public configuration, causal rows, spatial quadrature and plain-mode discovery."""
import numpy as np
import pytest

import epde
import epde.globals as state
from epde.interface.prepared_tokens import CacheStoredTokens, GridTokens, TrigonometricTokens
from epde.interface.search_config import load_search_config
from epde.operators.common.frequency_fit import ToneColumn, fit_support, row_residual_modes
from epde.preprocessing import weak_form as wf
from epde.structure.main_structures import Equation, Term


def search(**overrides):
    return epde.EpdeSearch(**dict(dict(evaluation_form='forward', system_rps_mode='plain',
                                       engine='nsga2', default_preprocessor_type='FD',
                                       population_size=4, neighbors_number=2, training_epochs=1,
                                       equation_terms_max_number=3), **overrides))


def trajectory(s, axes, fields, key=0, time_axis=0):
    grids = np.meshgrid(*axes, indexing='ij') if len(axes) > 1 else axes[0]
    domain = s.createDomain(grids, ID=key, time_axis=time_axis)[1]
    return s.createTrajectory(fields, domain, cache_id=key)[1]


def row_fit(s, trajectories, monkeypatch, **kwargs):
    monkeypatch.setattr(s, '_run_optimization', lambda *args, **kw: None)
    s.fit(trajectories, **kwargs)
    return state.samples_manager.evaluation_rows


def term(s, *labels):
    return Term(s.pool, passed_term=[s.pool.create(label=label, power=1)[1] for label in labels],
                max_factors_in_term=len(labels))


@pytest.mark.parametrize('options,match', [
    ({'forward_lag': 0}, 'positive integer'), ({'forward_lag': True}, 'positive integer'),
    ({'forward_space': 'auto'}, 'forward_space'), ({'system_rps_mode': 'assignment'}, 'plain'),
    ({'preprocessor_kwargs': {'lag': 2}}, 'forward_lag'),
    ({'system_rps_params': {'plain_target': False}}, 'plain_target'),
])
def test_invalid_config(options, match):
    with pytest.raises(ValueError, match=match):
        search(**options)


def test_forward_config_roundtrip():
    cfg = search(forward_space='weak', forward_lag=3).config
    assert load_search_config(cfg.as_dict()) == cfg


def test_irregular_origins_and_cache_isolation(monkeypatch):
    x = np.linspace(-1, 1, 17)
    t = np.cumsum(np.linspace(.06, .09, 50))
    X, T = np.meshgrid(x, t, indexing='ij')
    s = search(forward_lag=3)
    data = X**2 + T**2
    for shift in (0, 7):
        recs = [trajectory(s, [x, t], {'u': data + shift + k*T}, k, 1) for k in (0, 1)]
        rows = row_fit(s, recs, monkeypatch, max_deriv_order=(1, 2))
        target = term(s, 'du/dx1')
        for k in (0, 1):
            info = s.measurement_summary[k, 'u']
            assert info['time_axis'] == 1 and info['shape'] == (len(x)-4, len(t)-3)
            assert info['elapsed_time_range'] == pytest.approx((min(t[3:]-t[:-3]), max(t[3:]-t[:-3])))
            read = lambda *labels: rows.columns(term(s, *labels), 'u')[k].reshape(info['shape'])
            np.testing.assert_allclose(read('u'), (data + shift + k*T)[2:-2, :-3])
            np.testing.assert_allclose(read('du/dx1'), (T[:, 3:] + T[:, :-3] + k)[2:-2])
            np.testing.assert_allclose(read('du/dx0'), 2*X[2:-2, :-3], atol=1e-12)
            assert rows.is_target(target, 'u') and not rows.is_predictor(target)
            with pytest.raises(ValueError, match='increment'):
                read('u', 'du/dx1')


def test_spatial_windows_apply_to_complete_terms(monkeypatch):
    t = np.cumsum(np.linspace(.02, .04, 63))
    x = np.linspace(-2, 2, 65)
    T, X = np.meshgrid(t, x, indexing='ij')
    u = (1+T)*X**2 + np.sin(T)
    s = search(forward_space='weak', forward_lag=2, weak_window_gate='off')
    rows = row_fit(s, [trajectory(s, [t, x], {'u': u})], monkeypatch, max_deriv_order=(1, 1),
                   additional_tokens=[TrigonometricTokens()])
    plan = rows.plans[0, 'u']
    assert plan.ms[0] == 0 and plan.ss[0] == 1 and plan.shape[0] == len(t)-2
    def average(data, derivative=0):
        m = plan.ms[1]
        kernel = wf.window_kernels([m], [x[1]-x[0]], [plan.q[1]], [derivative])[0][derivative]
        return (np.lib.stride_tricks.sliding_window_view(data, 2*m+1, axis=1)[:, ::plan.ss[1]] @ kernel)
    def check(labels, expected):
        got = rows.columns(term(s, *labels), 'u')[0].reshape(plan.shape)
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)
    check(['du/dx0'], (average(u[2:])-average(u[:-2])) / (t[2:]-t[:-2])[:, None])
    check(['u'], average(u[:-2]))
    check(['du/dx1'], average((2*X*(1+T))[:-2]))
    # u u_x = (u²)_x / 2. Check the discrete by-parts quadrature itself.
    check(['u', 'du/dx1'], -.5 * average(u[:-2]**2, derivative=1))
    np.testing.assert_allclose(average(np.ones_like(u[:-2])), 1.)
    modulated = Term(s.pool, passed_term=[s.pool.create(label='u', power=1)[1],
                                          s.pool.create(label='sin', power=1, freq=2.3, dim=1)[1]],
                     max_factors_in_term=2)
    np.testing.assert_allclose(ToneColumn(rows, 0, 'u', modulated)([2.3]).reshape(plan.shape),
                               average((u*np.sin(2.3*X))[:-2]))
    # A future observation can change the target, never a predictor at the origin.
    altered = u.copy()
    altered[2] += X[2]**2
    f = rows.fields[0]
    changed = type(f)({'u': altered}, f.full_axes, f.time_axis, f.lag)
    for orders in ((0, 0), (0, 1)):
        np.testing.assert_array_equal(changed.get('u', orders)[0], f.get('u', orders)[0])
    assert not np.array_equal(changed.get('u', (1, 0))[0], f.get('u', (1, 0))[0])


@pytest.mark.parametrize('space', ['pointwise', 'weak'])
def test_irregular_fitted_tone_and_time_only_reduction(monkeypatch, space):
    t = np.cumsum(np.linspace(.06, .09, 160))
    omega = 8.32
    u = np.r_[0., np.cumsum(np.diff(t) * np.cos(omega*t[:-1]))]
    s = search(forward_space=space)
    rows = row_fit(s, [trajectory(s, [t], {'u': u})], monkeypatch, additional_tokens=[TrigonometricTokens()])
    assert rows.fit_frequencies and rows.uniform_step(0, 0) is None
    assert rows.plans[0, 'u'].shape == (len(t)-1,)
    assert rows.plans[0, 'u'].ms == (0,)
    guess, = row_residual_modes(rows, 0, 'u', np.cos(omega*t[:-1]), 0, np.pi/max(np.diff(t)))
    assert abs(guess-omega) <= np.pi/np.ptp(t[:-1])
    trig = Term(s.pool, passed_term=[s.pool.create(label='cos', power=1, freq=0., dim=0)[1]])
    eq = Equation(s.pool, [term(s, 'du/dx0'), trig], 'u', {'max_terms_number': {'value': 2},
                                                            'max_factors_in_term': {'value': 1}})
    eq.target_idx = 0
    fit = fit_support(eq, [1], False, rows)
    assert fit.structure[1].structure[0].param('freq') == pytest.approx(omega, rel=1e-5)


def test_stored_predictors_require_origin_declaration(monkeypatch):
    s = search()
    t = np.linspace(0, 3, 60)
    rec = trajectory(s, [t], {'u': np.cos(t)})
    def stored(available):
        return CacheStoredTokens('input', ['a'], {'a': t*t}, params_ranges={'power': (1, 1)},
                                 available_at_origin=available)
    with pytest.raises(ValueError, match='available_at_origin'):
        row_fit(s, [rec], monkeypatch, additional_tokens=[stored(False)])
    rows = row_fit(s, [rec], monkeypatch, additional_tokens=[stored(True)])
    np.testing.assert_allclose(rows.columns(term(s, 'a'), 'u')[0], t[:-1]**2)


def test_coupled_system_rejects_other_future_fields_and_incompatible_axes(monkeypatch):
    t = np.linspace(0, 3, 80)
    s = search()
    data = {'u': np.sin(t), 'v': np.cos(t)}
    rows = row_fit(s, [trajectory(s, [t], data)], monkeypatch)
    assert rows.is_target(term(s, 'dv/dx0'), 'v')
    with pytest.raises(ValueError, match='own field'):
        rows.columns(term(s, 'dv/dx0'), 'u')
    with pytest.raises(ValueError, match='own field'):
        rows.columns(term(s, 'dv/dx0', 'v'), 'u')
    for slot in data:
        np.testing.assert_allclose(rows.columns(term(s, 'v'), slot)[0], data['v'][:-1])
    invalid = Equation(s.pool, [term(s, 'du/dx0'), term(s, 'dv/dx0')], 'u',
                       {'max_terms_number': {'value': 2}, 'max_factors_in_term': {'value': 1}})
    invalid.target_idx = 0
    with pytest.raises(ValueError, match='future predictor'):
        rows.validate_candidate(invalid)
    fresh = Equation(s.pool, [], 'u', {'max_terms_number': {'value': 3},
                                       'max_factors_in_term': {'value': 1}})
    assert sum(rows.is_target(t, 'u') for t in fresh.structure) == 1
    assert all(rows.is_target(t, 'u') or rows.is_predictor(t) for t in fresh.structure)
    s = search(system_rps_params={'target_axes': [1]})
    with pytest.raises(ValueError, match='must select time axis'):
        row_fit(s, [trajectory(s, [t], data)], monkeypatch)


def test_explicit_margins_preserve_pointwise_spatial_stencils(monkeypatch):
    x = np.linspace(-1, 1, 25)
    t = np.linspace(0, 1, 60)
    T, X = np.meshgrid(t, x, indexing='ij')
    s = search(boundary_width=[2, 2], forward_lag=2)
    rows = row_fit(s, [trajectory(s, [t, x], {'u': X**4 + T**2})], monkeypatch)
    shape = rows.plans[0, 'u'].shape
    np.testing.assert_allclose(rows.columns(term(s, 'du/dx1'), 'u')[0].reshape(shape),
                               4*X[2:-4, 2:-2]**3, atol=1e-12)
    np.testing.assert_allclose(rows.columns(term(s, 'du/dx0'), 'u')[0].reshape(shape),
                               (T[4:-2] + T[2:-4])[:, 2:-2])


def test_pointwise_pool_constancy_cannot_remove_a_forward_target(monkeypatch):
    t = np.arange(64, dtype=float)
    u = (-1.)**t
    s = search(evaluation_form='strong')
    domain = s.createDomain(t, ID=0)[1]
    rec = s.createTrajectory({'u': u}, domain, cache_id=0, derivs=np.zeros((len(t), 1)))[1]
    row_fit(s, [rec], monkeypatch)
    assert 'du/dx0' not in s.pool.families_demand_equation[0].tokens
    s = search()
    rows = row_fit(s, [rec], monkeypatch)
    np.testing.assert_array_equal(rows.columns(term(s, 'du/dx0'), 'u')[0], np.diff(u))


def test_changing_forms_replaces_fit_local_measurements(monkeypatch):
    t = np.linspace(0, 8, 128)
    for form in ('forward', 'weak', 'strong', 'forward'):
        s = search(evaluation_form=form)
        rows = row_fit(s, [trajectory(s, [t], {'u': np.sin(t)})], monkeypatch)
        assert s.measurement_summary[0, 'u']['form'] == form
        assert (rows is None) == (form == 'strong')
        assert (state.samples_manager.weak is not None) == (form == 'weak')


@pytest.mark.parametrize('space', ['pointwise', 'weak'])
def test_seeded_forward_search(space):
    np.random.seed(14)
    t = np.linspace(0, 2, 75)
    x = np.linspace(-1, 1, 41)
    T, X = np.meshgrid(t, x, indexing='ij')
    u = (1 + .3*np.cos(3*X)) * np.exp(-T) + .1*X*T
    s = search(forward_space=space, default_preprocessor_type='poly', preprocessor_kwargs={'mp_poolsize': 1})
    tokens = [GridTokens(), TrigonometricTokens()] if space == 'pointwise' else [GridTokens()]
    s.fit([trajectory(s, [t, x], {'u': u})], additional_tokens=tokens, max_deriv_order=(2, 1),
          data_fun_pow=2)
    rows = state.samples_manager.evaluation_rows
    equations = [eq for sol in s.optimizer.pareto_levels.population for eq in sol.vals]
    assert equations
    if space == 'pointwise':
        assert rows.frequency_solves > 0
    for eq in equations:
        assert rows.is_target(eq.target, 'u')
        assert all(rows.is_predictor(f) for i, f in enumerate(eq.structure) if i != eq.target_idx)
        assert np.isfinite(eq.fitness_value)
