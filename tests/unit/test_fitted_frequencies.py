"""Recovery and native weak-column parity for continuous, equation-fitted tones."""
import numpy as np
import pytest

import epde
from epde.interface.prepared_tokens import TrigonometricTokens
from epde.operators.common.frequency_fit import _FrequencyBlock, ToneColumn, project_frequencies, residual_modes
from epde.preprocessing import weak_form as wf
from epde.structure.weak_rows import WeakRows


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('tones', [1, 2])
def test_projected_trial_matches_full_weighted_design(joint, intercept, tones):
    samples = []
    for length, shift in [(54, .7), (81, 1.2)]:
        t = np.linspace(0, 3, length)
        columns = [(lambda w, t=t: np.sin(w[0]*t), slice(0, 1)), (np.cos(2*t), None)]
        if tones == 2:
            columns.append((lambda w, t=t: np.cos(w[0]*t), slice(1, 2)))
        samples.append((np.exp(-shift*t), np.sqrt(t), columns))
    for group in ([samples] if joint else [[s] for s in samples]):
        y = np.concatenate([target * weight for target, weight, _ in group])
        block = _FrequencyBlock(group, y, intercept, lambda: None)
        for w in (np.array([1.1, .8]), np.array([0., 2.]), np.array([1.1, 2.+1e-13])):
            design = []
            for _, weight, cols in group:
                values = [c if idx is None else c(w[idx]) for c, idx in cols]
                if intercept:
                    values.append(np.ones_like(weight))
                design.append(np.column_stack(values) * weight[:, None])
            x = np.vstack(design)
            norms = np.linalg.norm(x, axis=0)
            x = x / np.where(norms > 0, norms, 1.)
            coeff, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
            actual, full_rank = block.residual(w)
            assert full_rank == (rank == x.shape[1])
            np.testing.assert_allclose(actual, y - x @ coeff, atol=2e-12)


def test_subcycle_duffing_with_linear_nuisance_and_intercept():
    t = np.linspace(0, 1, 151)
    u = np.sin(8*t)
    v = 8*np.cos(8*t)
    y = -0.5*v + u - u**3 + 0.7*np.cos(1.32*t) + 0.2
    cols = [(col, None) for col in (v, u, u**3)]
    cols.append((lambda w: np.cos(w[0]*t), slice(1)))
    weights = np.sqrt(1+t)
    w = project_frequencies([(y, weights, cols)], [0], [np.pi/(t[1]-t[0])],
                            [np.pi/2], [2*np.pi], True, lambda: None)
    np.testing.assert_allclose(w, [1.32], rtol=.01)


def test_default_fit_recovers_the_second_observed_peak_in_a_modulated_term():
    from epde.operators.common.frequency_fit import fit_support
    from epde.structure.main_structures import Equation, Term

    # du/dt = v sin(omega t). The derivative oscillates at twice the
    # required frequency, and that harmonic also outranks omega in the observations.
    t = .075 * np.arange(400)
    omega = 2.2
    data = {'u': -np.cos(2*omega*t)/(4*omega), 'v': np.cos(omega*t)}
    search = epde.EpdeSearch(use_solver=False, system_rps_mode='plain', engine='nsga2',
                             evaluation_form='weak')
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    domain = search.createDomain(t, ID=0)[1]
    trajectory = search.createTrajectory(data, domain, cache_id=0)[1]
    search.create_pool(data=[trajectory], additional_tokens=[TrigonometricTokens()], max_deriv_order=(1,))
    target = Term(search.pool, passed_term='du/dx0')
    sine = search.pool.create(label='sin', power=1, freq=0., dim=0)[1]
    feature = Term(search.pool, passed_term=[search.pool.create(label='v', power=1)[1], sine],
                   max_factors_in_term=2)
    equation = Equation(search.pool, [target, feature], 'u', {'max_terms_number': {'value': 2},
                                                              'max_factors_in_term': {'value': 2}})
    equation.target_idx = 0
    width, stride = (5,), (2,)
    plan = wf.WindowPlan(1, (2,), width, stride, (0,), wf.lattice_shape(t.shape, width, stride), 0.)
    rows = WeakRows({0: wf.Fields(data, [t])}, {(0, 'u'): plan}, [0])
    np.testing.assert_allclose(rows.observed_frequencies(0)[:2], [2*omega, omega], rtol=1e-5)
    fitted = fit_support(equation, [1], False, rows)
    recovered = next(factor for factor in fitted.structure[1].structure if factor.label == 'sin')
    assert recovered.param('freq') == pytest.approx(omega, rel=1e-4)


def test_observed_peaks_use_only_fit_trajectories_and_keep_axes_separate():
    t = np.arange(128) * .1
    x = np.arange(32) * .2
    shape = (t.size, x.size)
    u = np.broadcast_to(np.cos(3.1*t)[:, None], shape)
    other = np.broadcast_to(np.cos(8.2*x), shape)
    fields = {'train': wf.Fields({'u': u}, [t, x]), 'held': wf.Fields({'u': other}, [t, x])}
    rows = WeakRows(fields, {}, ['train'])
    assert rows.observed_frequencies(0)[0] == pytest.approx(3.1, rel=.001)
    assert rows.observed_frequencies(1) == ()


def test_two_independent_frequencies_on_one_axis():
    t = np.linspace(0, 16, 257)
    y = np.cos(1.33*t) - 0.7*np.cos(3.47*t)
    cols = [(lambda w: np.cos(w[0]*t), slice(i, i+1)) for i in range(2)]
    peaks = residual_modes(y, t.shape, 0, t[1], t[-1], np.pi/t[1], 2)
    w = project_frequencies([(y, np.ones_like(y), cols)], [0, 0], np.full(2, np.pi/t[1]),
                            np.full(2, np.pi/(2*t[-1])), peaks, False, lambda: None)
    np.testing.assert_allclose(sorted(w), [1.33, 3.47], rtol=.01)


def test_rank_deficient_support_is_refused():
    t = np.linspace(0, 1, 65)
    cols = [(t, None), (t, None), (lambda w: np.cos(w[0]*t), slice(1))]
    assert project_frequencies([(np.cos(t), np.ones_like(t), cols)], [0], [20.],
                               [1.], [6.], False, lambda: None) is None


def test_equivalent_frequencies_on_strided_rows_use_the_lower_representative():
    # Input spacing 0.1, weak-row spacing 0.2. Both frequencies are admissible
    # on the input grid and have identical cosine values on these weak rows.
    t = .2*np.arange(61)
    omega = 1.17
    upper = np.array([np.pi/.1])
    high = upper - omega
    y = np.cos(omega*t)
    np.testing.assert_allclose(np.cos(high[0]*t), y, atol=1e-13)
    columns = [(lambda w: np.cos(w[0]*t), slice(1))]
    w = project_frequencies([(y, np.ones_like(y), columns)], [0], upper,
                            [np.pi/(2*t[-1])], high, False, lambda: None, [[2*np.pi/.2]])
    np.testing.assert_allclose(w, [omega], rtol=.01)


def test_blind_axis_product_uses_native_quadrature_and_exact_frequency_keys():
    t, x = np.linspace(0, 1, 53), np.linspace(-2, 2, 59)
    grids = np.meshgrid(t, x, indexing='ij')
    field = np.cos(1.21*grids[0])*np.sin(2.34*grids[1])
    search = epde.EpdeSearch(use_solver=False, system_rps_mode='plain', engine='nsga2',
                             evaluation_form='weak', verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    domain = search.createDomain(grids, boundary_width=1, ID=0)[1]
    trajectory = search.createTrajectory({'u': field}, domain, cache_id=0)[1]
    trig = TrigonometricTokens()
    search.create_pool(data=[trajectory], additional_tokens=[trig], max_deriv_order=(1, 1))
    family, = [f for f in search.pool.families_meaningful if f.ftype == trig.token_family.ftype]
    assert family.status['fit_frequencies']
    np.testing.assert_allclose(family.status['frequency_nyquist'], np.pi / np.array([t[1], x[1]-x[0]]))
    assert not (family.param_grids or {}).get('freq') and not family.status['unique_token_type']
    factors = [family.create(label=name, all_vars=['u'], power=1, freq=2., dim=i)[1]
               for i,name in enumerate(('cos', 'sin'))]
    from epde.structure.main_structures import Term
    term = Term(search.pool, passed_term=factors, max_factors_in_term=2)
    fields = wf.Fields({'u': field}, [t, x])
    shape = wf.lattice_shape(field.shape, (3, 3), (2, 2))
    plan = wf.WindowPlan(1, (1, 1), (3, 3), (2, 2), (0, 0), shape, 0.)
    rows = WeakRows({0: fields}, {(0, 'u'): plan}, [0])
    rows.fit_frequencies = True
    import epde.globals as global_var
    from epde.operators.common import fill_prune
    from types import SimpleNamespace
    global_var.samples_manager.weak = rows
    eq = SimpleNamespace(pool=search.pool, metaparameters={'max_factors_in_term': {'value': 2}})
    target = Term(search.pool, passed_term='du/dx0')
    store = fill_prune.gram_store(eq, 0, rows.weights('u')[0], 'u', target)
    assert any({(f.label, int(f.param('dim'))) for f in t.structure} == {('cos', 0), ('sin', 1)}
               for t in store['lib'] if all(f.label in ('sin','cos') for f in t.structure))
    trial = ToneColumn(rows, 0, 'u', term)
    np.testing.assert_allclose(trial([2., 2.]), rows.columns(term, 'u')[0], atol=1e-14)
    factor = term.structure[0]
    first = term.factors_labels
    original_column = rows.columns(term, 'u')[0]
    factor.set_param(2.+1.e-10, name='freq')
    assert first != term.factors_labels
    assert not np.array_equal(rows.columns(term, 'u')[0], original_column)
    assert ('u', first) not in rows._columns
    modulation = search.pool.create(label='u', power=1)[1]
    mixed = Term(search.pool, passed_term=[modulation, factor], max_factors_in_term=2)
    np.testing.assert_allclose(ToneColumn(rows, 0, 'u', mixed)([2.+1.e-10]),
                               rows.columns(mixed, 'u')[0], atol=1e-14)
    y = rows.columns(Term(search.pool, passed_term=[modulation]), 'u')[0]
    cols = [(trial, slice(2))]
    peak = [residual_modes(y, shape, i, fields.steps[i]*2, np.ptp(a), bound)[0]
            for i,(a,bound) in enumerate(zip((t,x), family.status['frequency_nyquist']))]
    w = project_frequencies([(y, np.ones_like(y), cols)], [0, 1], family.status['frequency_nyquist'],
                            np.pi/(2*np.array([np.ptp(t), np.ptp(x)])), peak, False, lambda: None)
    np.testing.assert_allclose(w, [1.21, 2.34], rtol=.01)
    with pytest.raises(ValueError, match='Nyquist'):
        family.create(label='sin', all_vars=['u'], power=1, freq=100., dim=1)


def test_declarative_default_reuses_pool_and_enables_the_native_fit(monkeypatch):
    import epde.globals as global_var
    from test_pool_reuse import _no_search, _search, _trajectory

    search = _search(use_solver=False, system_rps_mode='plain', engine='nsga2', evaluation_form='weak',
                     tokens=[{'family': 'trigonometric', 'meaningful': False, 'allow_products': False}])
    builds = _no_search(search, monkeypatch)
    traj = _trajectory(search, 0)
    for _ in range(2):
        search.fit(data=[traj], max_deriv_order=(1,))
        assert global_var.samples_manager.weak.fit_frequencies
        family = next(f for f in search.pool.families if f.ftype == 'trigonometric')
        assert not family.status['meaningful'] and family.status['unique_token_type']
    assert len(builds) == 1


@pytest.mark.parametrize('options', [
    {'fit_frequencies': False}, {'freq': (1., 3.)}, {'freq_values': [1., 2.]}, {'freq_from_data': 1},
])
def test_explicit_frequency_policy_takes_precedence(options):
    from test_pool_reuse import _search, _trajectory

    search = _search(use_solver=False, system_rps_mode='plain', engine='nsga2', evaluation_form='weak')
    trig = TrigonometricTokens(**options)
    search.create_pool(data=[_trajectory(search, 0)], additional_tokens=[trig], max_deriv_order=(1,))
    family = trig.token_family
    assert not family.status.get('fit_frequencies') and not family.status['meaningful']
    if 'freq_from_data' in options:
        assert len(family.param_grids['freq']) == 1
    elif 'freq_values' in options:
        assert family.param_grids['freq'] == tuple(options['freq_values'])
    else:
        assert family.token_params['freq'] == options.get('freq', (np.pi/2, 2*np.pi))


@pytest.mark.parametrize('unsupported', [
    {'evaluation_form': 'strong'}, {'system_rps_params': {'plain_target': False}},
])
def test_rebinding_the_default_preserves_the_earlier_pool(unsupported):
    from test_pool_reuse import _search, _trajectory

    settings = dict(use_solver=False, system_rps_mode='plain', engine='nsga2', evaluation_form='weak')
    trig = TrigonometricTokens()
    first = _search(**settings)
    first.create_pool(data=[_trajectory(first, 0)], additional_tokens=[trig], max_deriv_order=(1,))
    previous = next(f for f in first.pool.families if f.ftype == trig.token_family.ftype)
    second = _search(**{**settings, **unsupported})
    data = [_trajectory(second, 0)]
    second.create_pool(data=data, additional_tokens=[trig], max_deriv_order=(1,))
    assert previous in first.pool.families and previous.status['fit_frequencies']
    rebound = next(f for f in second.pool.families if f.ftype == trig.token_family.ftype)
    assert not rebound.status.get('fit_frequencies')
    assert not trig.token_family.status.get('fit_frequencies')
    assert not trig.token_family.status['meaningful'] and trig.token_family.status['unique_token_type']
    assert trig.token_family.token_params['freq'] == (np.pi/2, 2*np.pi)
    assert trig.token_family.equality_ranges['freq'] > 0
    with pytest.raises(ValueError, match='requires plain weak fitting'):
        second.create_pool(data=data, additional_tokens=[TrigonometricTokens(fit_frequencies=True)],
                           max_deriv_order=(1,))


def test_frequency_enters_native_score_and_nontrig_replay_is_exact():
    import epde.globals as global_var
    from epde.operators.common import plain_fit
    from epde.structure.main_structures import Equation, Term
    from epde.structure.weak_rows import _param
    from test_plain_mode import _host_and_coeffs

    t = np.linspace(0, 7, 201)
    omega = 1.37
    search = epde.EpdeSearch(use_solver=False, system_rps_mode='plain', engine='nsga2',
                             evaluation_form='weak', verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    domain = search.createDomain(t, boundary_width=1, ID=0)[1]
    traj = search.createTrajectory({'u': np.sin(omega*t)/omega}, domain, cache_id=0)[1]
    trig = TrigonometricTokens(meaningful=True, allow_products=True, fit_frequencies=True)
    search.create_pool(data=[traj], additional_tokens=[trig], max_deriv_order=(1,))
    pool = search.pool
    fields = wf.Fields({'u': np.sin(omega*t)/omega}, [t])
    ms, ss = (5,), (2,)
    shape = wf.lattice_shape(t.shape, ms, ss)
    plan = wf.WindowPlan(1, (2,), ms, ss, (0,), shape, 0.)
    rows = WeakRows({0: fields}, {(0, 'u'): plan}, [0])
    global_var.samples_manager.weak = rows
    rows.fit_frequencies = True
    host, coefs = _host_and_coeffs()
    target = Term(pool, passed_term='du/dx0')
    def equation(feature):
        eq = Equation(pool, [target, feature], 'u', {'max_terms_number': {'value': 2},
                                                     'max_factors_in_term': {'value': 1}})
        plain_fit._install_target(eq, target)
        return eq
    cos = trig.token_family.create(label='cos', all_vars=['u'], power=1, freq=8., dim=0)[1]
    eq = equation(Term(pool, passed_term=[cos]))
    fitted = plain_fit._versions(eq, 2, coefs, {}, host, {})
    assert fitted
    for child in fitted:
        recovered = next(f for term in child.structure for f in term.structure if f.label == 'cos')
        assert _param(recovered, 'freq') == pytest.approx(omega, rel=.01)
        assert child.fitness_calculated
    assert _param(cos, 'freq') == 8.
    assert {key[-1] for key in rows.frequency_fits} == {False, True}
    solves = rows.frequency_solves
    plain_fit._versions(eq, 2, coefs, {}, host, {})
    assert rows.frequency_solves == solves > 0
    ordinary = equation(Term(pool, passed_term='u'))
    with_family = plain_fit._versions(ordinary, 2, coefs, {}, host, {})
    rows.fit_frequencies = False
    without_family = plain_fit._versions(ordinary, 2, coefs, {}, host, {})
    assert len(with_family) == len(without_family) > 0
    for a,b in zip(with_family, without_family):
        np.testing.assert_array_equal(a.weights_final, b.weights_final)
        assert (a.fitness_value, a.coefficients_stability) == (b.fitness_value, b.coefficients_stability)
