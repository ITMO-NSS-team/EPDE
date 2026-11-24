"""Public API tests for independent trajectories."""
import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import load_search_config
from epde.operators.common.sparsity import LASSOSparsity


def search(**kwargs):
    return epde.EpdeSearch(default_preprocessor_type='FD', boundary_width=3,
                           verbose_params={'show_iter_idx': False}, **kwargs)


def sample(obj, key=0, names=('u',), n=80):
    t = np.linspace(0, 5, n)
    domain = obj.createDomain(t, ID=key)[1]
    fields = {name: np.exp(-(i + 1) * t) for i, name in enumerate(names)}
    return obj.createTrajectory(fields, domain, cache_id=key)[1]


def test_joint_configuration_is_explicit_and_serializable():
    default = load_search_config()
    assert default.objectives.multisample_mode == 'mean'
    assert default.objectives.multisample_weighting == 'samples'
    cfg = load_search_config(overrides={'system_rps_mode': 'plain',
                                       'multisample_mode': 'joint',
                                       'multisample_weighting': 'points'})
    assert cfg.as_dict()['objectives']['multisample_mode'] == 'joint'
    assert load_search_config(cfg.as_dict()).objectives.multisample_weighting == 'points'


@pytest.mark.parametrize('overrides, match', [
    ({'multisample_mode': 'typo'}, 'multisample_mode'),
    ({'multisample_weighting': 'typo'}, 'multisample_weighting'),
    ({'multisample_mode': 'joint', 'system_rps_mode': 'repair', 'instability_metric': 'cv'}, 'basis-free'),
    ({'multisample_mode': 'joint', 'system_rps_mode': 'assignment',
      'sparsity_cls': 'lasso'}, 'physical|lasso|refit'),
])
def test_invalid_multisample_config_is_refused(overrides, match):
    with pytest.raises(ValueError, match=match):
        load_search_config(overrides=overrides)


def test_joint_repair_rejects_a_physical_operator_without_joint_fitting():
    class PerRecordSparsity(LASSOSparsity):
        fits_physical_scale = True

    with pytest.raises(ValueError, match='repair.*vwsr'):
        load_search_config(overrides={'multisample_mode': 'joint',
                                      'system_rps_mode': 'repair',
                                      'sparsity_cls': PerRecordSparsity})


def test_joint_repair_accepts_an_explicit_default_sparsity_operator():
    config = load_search_config(overrides={'multisample_mode': 'joint',
                                          'system_rps_mode': 'repair',
                                          'sparsity_cls': None})
    assert config.objectives.multisample_mode == 'joint'


def test_empty_sample_list_is_rejected_before_build():
    obj = search()
    with pytest.raises(ValueError, match='at least one|empty'):
        obj.create_pool([])


@pytest.mark.parametrize('names', [('v',), ('u', 'v')])
def test_records_must_have_the_same_variables(names):
    obj = search()
    a, b = sample(obj), sample(obj, 1, names)
    with pytest.raises(ValueError, match='variable'):
        obj.create_pool([a, b])
    assert not a.built and not b.built


def test_variable_order_cannot_silently_change_derivative_ids():
    obj = search()
    a, b = sample(obj, names=('u', 'v')), sample(obj, 1, ('v', 'u'))
    with pytest.raises(ValueError, match='variable.*order|same.*variables'):
        obj.create_pool([a, b])


def test_repeated_sample_is_not_silently_deduplicated():
    obj = search()
    a = sample(obj)
    with pytest.raises(ValueError, match='duplicate|Duplicate'):
        obj.create_pool([a, a])


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_invalid_field_is_rejected_with_sample_context(value):
    obj = search()
    a = sample(obj, 7)
    a._entries[0].raw_tensor[25] = value
    with pytest.raises(ValueError, match='7.*u|u.*7'):
        obj.create_pool([a])


def test_conflicting_domain_id_cannot_reuse_old_coordinates():
    obj = search()
    old = obj.createDomain(np.arange(20.0), ID=7)[1]
    with pytest.raises(ValueError, match='[Dd]omain.*7|ID.*7'):
        obj.createDomain(100.0 + np.arange(30.0), ID=7)
    np.testing.assert_array_equal(old.getGrids()[0], np.arange(20.0))


def test_invalid_domain_can_be_retried_with_the_same_id():
    with search() as obj:
        t = np.linspace(0, 5, 80)
        with pytest.raises(ValueError):
            obj.createDomain(t, ID=12, gfunction=np.zeros_like(t))
        _, domain = obj.createDomain(t + 1, ID=12)
        np.testing.assert_array_equal(domain.getGrids()[0], t + 1)


def test_multisample_adapter_uses_independent_fd_derivatives():
    grids = [np.linspace(0, 3, 70), np.linspace(10, 14, 90)]
    records = [(t, np.exp(-0.4 * (t - t[0]))) for t in grids]
    obj = epde.EpdeMultisample(records, default_preprocessor_type='FD', boundary_width=3,
                              system_rps_mode='plain', engine='nsga2',
                              verbose_params={'show_iter_idx': False})
    obj.set_samples(records, max_deriv_orders=1)
    assert len(global_var.samples_manager.trajecatoryIDs) == 2
    for trajectory in global_var.samples_manager._traj.values():
        u = trajectory.get(('u', (1.0,)))
        du = trajectory.get(('du/dx0', (1.0,)))
        np.testing.assert_allclose(du, -0.4 * u, atol=3e-5, rtol=3e-4)


def test_multisample_adapter_supports_systems_and_analytic_derivatives():
    records, derivs = [], []
    for n, initial in [(75, 1.0), (95, 2.0)]:
        t = np.linspace(0, 4, n)
        u, v = initial * np.exp(-t), initial * np.exp(-2 * t)
        records.append((t, {'u': u, 'v': v}))
        derivs.append({'u': (-u)[:, None], 'v': (-2 * v)[:, None]})
    obj = epde.EpdeMultisample(default_preprocessor_type='FD', boundary_width=3,
                              verbose_params={'show_iter_idx': False})
    obj.set_samples(records, sample_derivs=derivs, max_deriv_orders=1)
    assert [family.variable for family in obj.pool.families_demand_equation] == ['u', 'v']
    for trajectory in global_var.samples_manager._traj.values():
        np.testing.assert_allclose(trajectory.get(('dv/dx0', (1.0,))),
                                   -2 * trajectory.get(('v', (1.0,))))


def test_active_geometry_ignores_unused_cached_domains():
    obj = search()
    obj.createDomain(np.linspace(0, 5, 30), ID=55)
    T, X = np.meshgrid(np.linspace(0, 4, 25), np.linspace(0, 3, 30), indexing='ij')
    domain = obj.createDomain([T, X], ID=66)[1]
    u = np.exp(-T) * np.cos(X)
    derivatives = np.column_stack([-u.ravel(), (-np.exp(-T) * np.sin(X)).ravel()])
    trajectory = obj.createTrajectory({'u': u}, domain, cache_id=99, derivs=derivatives)[1]
    obj.create_pool([trajectory], max_deriv_order=(1, 1))
    assert global_var.samples_manager.ndim == 2
    assert global_var.samples_manager.grid_keys == ['0', '1']


def test_nonuniform_one_dimensional_grid_reports_its_measure():
    obj = search()
    t = np.linspace(0, 2, 40) ** 2
    with pytest.warns(global_var.EPDEUsageWarning, match='not uniformly sampled'):
        obj.createDomain(t, ID=10)


def _decay_records(rate, offset=0.0):
    records, derivatives = [], []
    for n, amplitude in [(72, 1.0), (96, 2.0)]:
        t = np.linspace(offset, offset + 3.0, n)
        u = amplitude * np.exp(-rate * (t - offset))
        records.append((t, u))
        derivatives.append((-rate * u)[:, None])
    return records, derivatives


def _joint_adapter(records=None, *, retain_cache=False):
    return epde.EpdeMultisample(
        records, use_solver=False, device='cpu', engine='nsga2',
        config={'runtime': {'free_tensor_cache_after_fit': not retain_cache}},
        system_rps_mode='plain', multisample_mode='joint',
        discrepancy_metric='wape', instability_metric='dyad',
        default_preprocessor_type='FD', boundary_width=0,
        equation_terms_max_number=2, equation_factors_max_number=1,
        data_fun_pow=1, deriv_fun_pow=1, population_size=4, training_epochs=2,
        system_rps_params={'variable_order_seed': 11, 'fill_probability': 0.0},
        verbose_params={'show_iter_idx': False})


def _assert_decay_front(obj, rate):
    from epde.optimizers.nsga.optimizer import NSGAOptimizer
    assert isinstance(obj.optimizer, NSGAOptimizer)
    assert obj.config.objectives.multisample_mode == 'joint'
    assert len(obj.pareto_history) == 2
    equations = obj.equations_by_variable(only_print=False)['u']
    laws = [equation for equation in equations
            if equation.target.structure[0].label == 'du/dx0'
            and [term.structure[0].label for term in equation.structure
                 if term is not equation.target] == ['u']]
    assert laws, 'the actual optimizer must report the common decay law'
    for equation in laws:
        np.testing.assert_allclose(equation.weights_final, [-rate, 0.0], atol=1e-11)
        assert equation.fitness_value < 1e-11
        assert np.isfinite(equation.coefficients_stability)


def test_multisample_adapter_fits_constructor_records_with_the_real_optimizer():
    records, derivatives = _decay_records(0.4)
    obj = _joint_adapter(records, retain_cache=True)
    try:
        np.random.seed(11)
        obj.fit(derivs=derivatives, max_deriv_order=(1,))
        _assert_decay_front(obj, 0.4)
        assert [trajectory.inner_shape.tolist()
                for trajectory in global_var.samples_manager._traj.values()] == [[72], [96]]
        for trajectory, (_, u) in zip(global_var.samples_manager._traj.values(), records):
            np.testing.assert_allclose(trajectory.get(('du/dx0', (1.0,))), -0.4 * u)
    finally:
        obj.close()


def test_multisample_adapter_refit_and_replacement_use_the_current_records():
    records, derivatives = _decay_records(0.4)
    obj = _joint_adapter()
    try:
        obj.set_samples(records, sample_derivs=derivatives, max_deriv_orders=(1,))
        np.random.seed(11)
        obj.fit()
        _assert_decay_front(obj, 0.4)
        assert all(not trajectory.tensors_cached
                   for trajectory in global_var.samples_manager._traj.values())
        initial_ids = tuple(global_var.samples_manager.trajecatoryIDs)
        np.random.seed(11)
        obj.fit()
        _assert_decay_front(obj, 0.4)
        assert tuple(global_var.samples_manager.trajecatoryIDs) == initial_ids
        assert all(not trajectory.tensors_cached
                   for trajectory in global_var.samples_manager._traj.values())

        replacement, replacement_derivatives = _decay_records(0.9, offset=20.0)
        np.random.seed(11)
        obj.fit(samples=replacement, derivs=replacement_derivatives, max_deriv_order=(1,))
        _assert_decay_front(obj, 0.9)
        active = global_var.samples_manager._traj
        assert len(active) == 2 and set(active).isdisjoint(initial_ids)
        assert all(not trajectory.tensors_cached for trajectory in active.values())
        for trajectory, (t, u) in zip(active.values(), replacement):
            np.testing.assert_array_equal(trajectory.grids()[0], t)
            np.testing.assert_array_equal(trajectory.raw_field('u'), u)
    finally:
        obj.close()
