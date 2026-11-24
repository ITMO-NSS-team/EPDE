"""Check that sparse repair can recover terms identified only across records."""
import numpy as np
import pytest

import epde
from conftest import using_config
from epde.operators.common.sparsity import PhysicsInformedLasso, VWSRSparsity


def _degenerate_records(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for key, k, n in [(7, 1, 300), (19, 2, 400)]:
        t = np.linspace(0, 2, n)
        u = np.exp(-(0.2 + 0.1 * k * k) * t) * (1.0 + 0.3 * np.sin(5 * t))
        uxx = -k * k * u
        y = -0.2 * u + 0.1 * uxx + noise * rng.normal(size=n)
        blocks.append((key, np.column_stack([u, uxx]), y, np.ones(n), (n,)))
    return blocks


@pytest.mark.parametrize('metric', ['dyad', 'chi2'])
def test_joint_sparse_fit_keeps_a_law_only_the_records_identify(metric):
    with using_config(instability_metric=metric, multisample_mode='joint'):
        estimator = PhysicsInformedLasso().fit_joint(_degenerate_records())
    np.testing.assert_allclose(estimator.full_coef_, [-0.2, 0.1, 0.0], atol=1e-8)


def test_per_record_sparse_fits_lose_that_law():
    # The legacy path: each record alone cannot separate the two columns.
    coefficients = []
    with using_config(instability_metric='dyad'):
        for key, X, y, sw, shape in _degenerate_records():
            estimator = PhysicsInformedLasso().fit(X, y, sample_key=key, grid_shape=shape,
                                                   sample_weights=sw)
            coefficients.append(estimator.full_coef_)
    averaged = np.mean(coefficients, axis=0)
    assert not np.allclose(averaged, [-0.2, 0.1, 0.0], atol=1e-3)


def test_one_record_joint_fit_matches_the_ordinary_fit():
    key, X, y, sw, shape = _degenerate_records(noise=0.01)[1]
    X = np.column_stack([X[:, 0], np.sin(np.linspace(0, 7, y.size))])
    with using_config(instability_metric='dyad'):
        single = PhysicsInformedLasso().fit(X, y, sample_key=key, grid_shape=shape, sample_weights=sw)
        joint = PhysicsInformedLasso().fit_joint([(key, X, y, sw, shape)])
    np.testing.assert_allclose(joint.full_coef_, single.full_coef_, rtol=1e-10, atol=1e-12)


def test_record_order_does_not_change_the_joint_sparse_fit():
    blocks = _degenerate_records(noise=0.01)
    with using_config(instability_metric='dyad'):
        forward = PhysicsInformedLasso().fit_joint(blocks).full_coef_
        backward = PhysicsInformedLasso().fit_joint(blocks[::-1]).full_coef_
    np.testing.assert_array_equal(forward, backward)


def test_window_metrics_are_refused_for_the_joint_sparse_fit():
    with using_config(instability_metric='cv'):
        with pytest.raises(ValueError, match='basis-free'):
            PhysicsInformedLasso().fit_joint(_degenerate_records())


def test_joint_repair_rejects_a_candidate_with_nonfinite_token_values():
    with epde.EpdeSearch(multisample_mode='joint', system_rps_mode='repair',
                         boundary_width=0, device='cpu') as search:
        t = np.linspace(0, 2, 100)
        family = epde.CacheStoredTokens(token_type='stored', token_labels=['g'],
                                        params_ranges={'power': (1, 1)}, meaningful=True)
        records = []
        for key in (7, 19):
            column = np.sin(t)
            if key == 19:
                column[10] = np.nan
            domain = search.createDomain(t, ID=key)[1]
            _, record = search.createTrajectory({'u': np.cos(t)}, domain, cache_id=key,
                                                 derivs=(-np.sin(t))[:, None],
                                                 cached_token_tensors={family: {'g': column}})
            records.append(record)
        search.create_pool(records, additional_tokens=[family])
        system = epde.translate_equation('1.0 * g{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
                                         search.pool, all_vars=['u'])
        equation = system.vals['u']
        VWSRSparsity().apply(equation, {})
        np.testing.assert_array_equal(equation.weights_internal, [0.0, 0.0])


def test_joint_repair_search_runs_end_to_end():
    import epde
    search = epde.EpdeSearch(system_rps_mode='repair', multisample_mode='joint', engine='nsga2',
                             use_solver=False, default_preprocessor_type='FD', boundary_width=2,
                             equation_terms_max_number=3, population_size=6, training_epochs=2,
                             verbose_params={'show_iter_idx': False}, device='cpu')
    trajectories = []
    for key, k in [(7, 1.0), (19, 2.0)]:
        t = np.linspace(0, 3, 120)
        u = np.exp(-k * t)
        domain = search.createDomain((t,), ID=key)[1]
        trajectories.append(search.createTrajectory({'u': u}, domain, cache_id=key,
                                                    derivs={'u': (-k * u)[:, None]})[1])
    search.fit(data=trajectories, max_deriv_order=(1,), data_fun_pow=2)
    assert list(search.optimizer.reported_front())
    search.close()


def _proportional_on_each_record(search):
    trajectories = []
    for key, k in [(7, 1.0), (19, 2.0)]:
        t = np.linspace(0, 2, 80)
        x = np.linspace(0, 2 * np.pi, 30)
        T, X = np.meshgrid(t, x, indexing='ij')
        u = np.exp(-(0.2 + 0.1 * k * k) * T) * np.sin(k * X)
        domain = search.createDomain([T, X], ID=key, boundary_width=0)[1]
        derivs = np.column_stack([((-0.2 - 0.1 * k * k) * u).ravel(), (k * np.exp(-(0.2 + 0.1 * k * k) * T)
                                  * np.cos(k * X)).ravel(), (-k * k * u).ravel()])
        trajectories.append(search.createTrajectory({'u': u}, domain, cache_id=key, derivs={'u': derivs})[1])
    return trajectories


@pytest.mark.parametrize('mode, kept', [('mean', 2), ('joint', 3)])
def test_dependence_scrub_reads_the_records_together_only_in_joint_mode(mode, kept):
    import epde
    from epde.interface.equation_translator import translate_equation
    from epde.operators.common.right_part_selection import EqRightPartSelector
    search = epde.EpdeSearch(system_rps_mode='repair', multisample_mode=mode, use_solver=False,
                             equation_terms_max_number=3, device='cpu')
    search.create_pool(data=_proportional_on_each_record(search), max_deriv_order=(1, 2), data_fun_pow=1)
    system = translate_equation({'u': '1.0 * u{power: 1.0} + 1.0 * d^2u/dx1^2{power: 1.0} + 0.0 = '
                                      'du/dx0{power: 1.0}'}, search.pool, all_vars=['u'])
    equation = system.vals['u']
    features = [term for term in equation.structure if term is not equation.structure[-1]]
    with using_config(system_rps_mode='repair', multisample_mode=mode):
        EqRightPartSelector()._drop_dependent_terms(equation, features)
    assert len(equation.structure) == kept
    search.close()
