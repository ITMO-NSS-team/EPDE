import json
import random
import subprocess
import sys
import numpy as np
import pytest

from epde import DiscoveryResult, EpdeSearch, TrajectoryData, read_trajectories_csv


def observations(offset=0, engine=None):
    t = np.linspace(0, 9, 100)
    fields = {'u': np.sin(t) + offset}
    if engine != 'single':
        fields['v'] = np.cos(t)
    if engine == 'nsga2-records':
        other = {'u': 1.1 * np.sin(t + 0.2) + offset, 'v': 1.1 * np.cos(t + 0.2)}
        return [TrajectoryData({'t': t}, fields), TrajectoryData({'t': t}, other)]
    return TrajectoryData({'t': t}, fields)


def search(**options):
    cfg = dict(engine='nsga2', system_rps_mode='plain', default_preprocessor_type='FD',
               population_size=5, training_epochs=1, equation_terms_max_number=3,
               instability_metric='dyad', boundary_width=2, verbose_params={'show_iter_idx': False})
    return EpdeSearch(**dict(cfg, **options))


def test_csv_reconstructs_axes_and_binds_columns(tmp_path):
    p = tmp_path / 'input.csv'
    p.write_text('id,t,x,y,dy,f\na,4,8,48,0,2\na,0,8,8,1,2\na,4,2,42,2,2\na,0,2,2,3,2\n')
    opts = dict(coordinates=['t', 'x'], fields={'u': 'y'}, derivatives={'u': ['dy']},
                derivative_orders=(1, 0), external_inputs=['f'], trajectory_id='id')
    d, = read_trajectories_csv(p, **opts)
    np.testing.assert_equal(d.fields['u'], [[2, 8], [42, 48]])
    np.testing.assert_equal(d.derivatives['u'][:, 0], [3, 1, 2, 0])
    assert d.trajectory_id == 'a'
    p.write_text(p.read_text().rsplit('a,0,2', 1)[0])
    with pytest.raises(ValueError, match='Incomplete'):
        read_trajectories_csv(p, **opts)
    p.write_text(p.read_text() + 'a,4,8,48,0,2\n')
    with pytest.raises(ValueError, match='Duplicate'):
        read_trajectories_csv(p, **opts)


def test_new_and_legacy_paths_and_detached_result(tmp_path):
    d = observations()
    np.random.seed(24)
    a = search()
    result = a.discover(d)
    np.random.seed(24)
    b = search()
    domain = b.createDomain(d.grids, ID=0)[1]
    trajectory = b.createTrajectory(d.fields, domain, cache_id=0)[1]
    assert b.fit([trajectory]) is None
    assert b.result == result
    b.close()
    assert a.result == result
    a.close()
    path = tmp_path / 'result.json'
    result.save(path)
    assert DiscoveryResult.load(path) == result
    assert set(result.variable_fronts) == {'u', 'v'}
    assert result.front[0]['equations'][0]['target']


def test_alternating_data_and_config():
    import epde.globals as g
    a, b = search(), search(evaluation_form='forward')
    a.create_pool(observations())
    acache = a.tensor_cache
    b.create_pool(observations(1))
    assert b.tensor_cache is not acache
    b.close()
    assert a.tensor_cache is acache
    np.testing.assert_equal(g.samples_manager[0].raw_field('u'), observations().fields['u'])


def test_forward_defers_native_derivatives_until_requested(monkeypatch):
    s = search(evaluation_form='forward', free_tensor_cache_after_fit=False)
    s.set_preprocessor()
    from unittest.mock import Mock
    run = Mock(wraps=s.preprocessor_pipeline.run)
    monkeypatch.setattr(s.preprocessor_pipeline, 'run', run)
    result = s.discover(observations())
    assert result.measurements[0]['form'] == 'forward'
    sample, = s._pool_data
    assert not run.called
    assert sample.base_derivs
    assert run.call_count == len(sample.variable_names)
    sample.base_derivs
    assert run.call_count == len(sample.variable_names)


def test_warning_preferences_are_scoped_to_the_search():
    import warnings
    from epde.interface.runtime import search_operation

    @search_operation
    def probe(search):
        warnings.warn('probe', RuntimeWarning)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        filters = warnings.filters.copy()
        quiet = search(verbose_params={'show_warnings': False})
        verbose = search(verbose_params={'show_warnings': True})
        assert warnings.filters == filters
        probe(quiet)
        probe(verbose)
        warnings.warn('outside', RuntimeWarning)
        assert [str(w.message) for w in seen if str(w.message) in ('probe', 'outside')] == ['probe', 'outside']


def test_searches_retain_random_streams_and_token_bindings():
    import epde.globals as g
    from epde import GridTokens
    np.random.seed(7)
    a = search()
    shared = GridTokens()
    a.create_pool(observations(), additional_tokens=[shared])
    family = shared.token_family
    expected = np.random.RandomState()
    expected.set_state(np.random.get_state())
    np.random.seed(31)
    b = search()
    t, x = np.arange(10.), np.arange(11.)
    b.create_pool(TrajectoryData({'t': t, 'x': x}, {'u': np.sin(t[:, None]) + x}),
                  additional_tokens=[shared])
    assert family.token_params['dim'] == (0, 0)
    assert shared.token_family.token_params['dim'] == (0, 1)
    assert a.tensor_cache is g.tensor_cache
    assert np.random.randint(1000000) == expected.randint(1000000)


def test_external_csv_inputs_are_per_trajectory(tmp_path):
    import epde.globals as g
    t = np.arange(30)
    path = tmp_path / 'trials.csv'
    rows = ['trial,t,signal,input'] + [f'{key},{n},{np.cos(n)},{n+offset}'
            for key, offset in [('a', 0), ('b', 10)] for n in reversed(t)]
    path.write_text('\n'.join(rows))
    data = read_trajectories_csv(path, coordinates=['t'], fields={'u': 'signal'},
                                 external_inputs={'f': 'input'}, trajectory_id='trial')
    path.write_text('\n'.join([rows[0], *reversed(rows[1:])]))
    reordered = read_trajectories_csv(path, coordinates=['t'], fields={'u': 'signal'},
                                      external_inputs={'f': 'input'}, trajectory_id='trial')
    assert [d.trajectory_id for d in reordered] == [d.trajectory_id for d in data]
    s = search()
    s.create_pool(data)
    for d in data:
        np.testing.assert_array_equal(g.samples_manager[d.trajectory_id].raw_token('f'), d.external_inputs['f'])


@pytest.mark.parametrize('form', ['strong', 'weak', 'forward'])
def test_pde_result_matches_trajectory_api(form):
    t, x = np.linspace(0, 2, 69), np.linspace(-1, 1, 49)
    field = np.exp(-t[:, None]) * np.cos(2*x) + t[:, None]*x/10
    d = TrajectoryData({'t': t, 'x': x}, {'u': field})
    snapshots = []
    for typed in (False, True):
        np.random.seed(23)
        random.seed(23)
        with search(evaluation_form=form, population_size=4, neighbors_number=2,
                    max_deriv_order=(1, 2), data_fun_pow=2) as s:
            if typed:
                snapshots.append(s.discover(d))
            else:
                _, domain = s.createDomain(d.grids, ID=0)
                _, trajectory = s.createTrajectory(d.fields, domain, cache_id=0)
                s.fit([trajectory])
                snapshots.append(s.result)
    assert snapshots[0] == snapshots[1]
    assert snapshots[1].front
    assert snapshots[1].measurements[0]['form'] == form


def test_solver_backed_discovery_matches_legacy(monkeypatch, tmp_path):
    from epde.integrate.basis_integration import BasisAdapter

    solves = []
    solve = BasisAdapter.solve

    def record_solve(adapter, *args, **kwargs):
        value = solve(adapter, *args, **kwargs)
        solves.append(value[1])
        return value

    monkeypatch.setattr(BasisAdapter, 'solve', record_solve)
    d = observations(engine='single')
    snapshots = []
    cfg = dict(engine='moeadd', system_rps_mode='repair', population_size=4,
               neighbors_number=2, max_deriv_order=2, data_fun_pow=2,
               use_solver=True, device='cpu', solver_backend='basis',
               basis_config={'lbfgs_maxiter': 4, 'max_refinements': 0, 'threads': 1})
    checkpoint = tmp_path / 'basis.zip'
    for typed in (False, True):
        np.random.seed(23)
        random.seed(23)
        with search(**cfg) as s:
            if typed:
                snapshots.append(s.discover(d, checkpoint=checkpoint))
            else:
                _, domain = s.createDomain(d.grids, ID=0)
                _, trajectory = s.createTrajectory(d.fields, domain, cache_id=0)
                s.fit([trajectory])
                snapshots.append(s.result)
    assert solves and np.isfinite(solves).all()
    assert snapshots[0] == snapshots[1]
    assert snapshots[0].front
    with search(**cfg) as s:
        assert s.resume(checkpoint, d) == snapshots[0]


def engine_search(engine):
    import torch
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    options = dict(training_epochs=2)
    if engine == 'nsga2-records':
        options.update(multisample_discrepancy='per_record', multisample_mode='joint',
                       discrepancy_metric='wape', use_solver=False)
    elif engine.startswith('nsga2-'):
        options.update(evaluation_form=engine.split('-')[1], early_stopping_patience=1)
    elif engine != 'nsga2':
        options.update(engine='moeadd', system_rps_mode='repair', multiobjective_mode=engine != 'single')
    if engine == 'single':
        options.update(max_deriv_order=2, data_fun_pow=2, equation_factors_max_number=2)
    return search(**options)


def test_plain_fit_cache_preserves_front_history_and_random_draws(monkeypatch):
    from epde.operators.common import plain_fit
    data = observations()
    def discover():
        with engine_search('nsga2') as s:
            result = s.discover(data)
            return result, s.pareto_history, random.random(), np.random.random()

    cached = discover()
    monkeypatch.setattr(plain_fit, '_FIT_CACHE_SIZE', 0)
    assert discover() == cached


@pytest.mark.parametrize('engine', ['nsga2', 'moeadd', 'single', 'nsga2-forward', 'nsga2-weak', 'nsga2-records'])
def test_checkpoint_continues_in_fresh_process(tmp_path, monkeypatch, engine):
    from epde.interface.checkpoint import EpochCheckpoint
    d = observations(engine=engine)
    baseline = engine_search(engine)
    expected = baseline.discover(d)
    history = dict(epochs=baseline.pareto_history, next_draw=[random.random(), np.random.random()])
    baseline.close()
    save = EpochCheckpoint.save
    path = tmp_path / 'run.zip'

    class Interrupted(Exception):
        pass

    def interrupt(self, search, epoch, stopped):
        save(self, search, epoch, stopped)
        raise Interrupted()

    monkeypatch.setattr(EpochCheckpoint, 'save', interrupt)
    with pytest.raises(Interrupted):
        engine_search(engine).discover(d, checkpoint=path)
    with pytest.raises(ValueError, match='differ'):
        engine_search(engine).resume(path, observations(1, engine))
    code = """import json, random, sys
import numpy as np
sys.path.insert(0, 'tests/unit')
from test_discovery_pipeline import engine_search, observations
s = engine_search(sys.argv[1])
random.seed(456)
np.random.seed(654)
r = s.resume(sys.argv[2], observations(engine=sys.argv[1]))
r.save(sys.argv[3])
with open(sys.argv[4], 'w') as f:
    json.dump(dict(epochs=s.pareto_history, next_draw=[random.random(), np.random.random()]), f)
"""
    result_file, history_file = tmp_path / 'result.json', tmp_path / 'history.json'
    run = subprocess.run([sys.executable, '-c', code, engine, str(path), str(result_file), str(history_file)],
                         capture_output=True, text=True, errors='replace')
    assert run.returncode == 0, run.stderr[-3000:]
    restored = DiscoveryResult.load(result_file)
    if restored != expected:
        expected.save(tmp_path / 'uninterrupted.json')
    assert restored == expected
    assert json.loads(history_file.read_text()) == history


def test_multiple_trajectories_with_supplied_derivatives_and_input():
    from epde import CacheStoredTokens
    samples = []
    for key, length in [('fast', 85), ('slow', 123)]:
        t = np.linspace(0, 8, length)
        samples.append(TrajectoryData({'t': t}, {'u': np.sin(t)}, derivatives={'u': np.cos(t)[:, None]},
                                      derivative_orders=(1,), external_inputs={'forcing': np.cos(t)},
                                      trajectory_id=key))
    snapshots = []
    for typed in (True, False):
        np.random.seed(12)
        random.seed(12)
        with search(population_size=4, neighbors_number=2) as s:
            family = CacheStoredTokens('inputs', ['forcing'], params_ranges={'power': (1, 1)}, meaningful=True)
            if typed:
                snapshots.append(s.discover(samples, additional_tokens=[family]))
            else:
                trajectories = []
                for d in samples:
                    _, domain = s.createDomain(d.grids, ID=d.trajectory_id)
                    _, trajectory = s.createTrajectory(d.fields, domain, derivs=d.derivatives,
                                                        cache_id=d.trajectory_id,
                                                        cached_token_tensors={family: d.external_inputs})
                    trajectories.append(trajectory)
                s.fit(trajectories, additional_tokens=[family])
                snapshots.append(s.result)
    assert snapshots[0] == snapshots[1]
    assert snapshots[0].front
    assert [m['trajectory'] for m in snapshots[0].measurements] == ['fast', 'slow']


def test_interrupted_checkpoint_write_preserves_previous_epoch(monkeypatch, tmp_path):
    from epde.interface.checkpoint import _Writer
    d = observations(engine='single')
    path = tmp_path / 'run.zip'
    kwargs = dict(engine='moeadd', system_rps_mode='repair', multiobjective_mode=False,
                  training_epochs=1, max_deriv_order=2, equation_factors_max_number=2, data_fun_pow=2)
    np.random.seed(9)
    with search(**kwargs) as s:
        expected = s.discover(d, checkpoint=path)
    original = path.read_bytes()

    def fail(write, state):
        raise OSError('injected checkpoint write failure')

    monkeypatch.setattr(_Writer, 'dump', fail)
    with search(**kwargs) as s:
        assert s.resume(path, d) == expected
        with pytest.raises(OSError, match='injected'):
            s.resume(path, d, total_epochs=2)
        with pytest.raises(AttributeError, match='not been discovered'):
            _ = s.result
    assert path.read_bytes() == original
    with search(**kwargs) as s:
        assert s.resume(path, d) == expected
    with search(**dict(kwargs, data_fun_pow=3)) as s, pytest.raises(ValueError, match='differ'):
        s.resume(path, d)


def test_custom_callback_can_read_pointwise_data_from_a_reused_forward_pool():
    import epde.globals as g
    d = observations()
    with search(evaluation_form='forward') as s:
        _, domain = s.createDomain(d.grids, ID=0)
        _, sample = s.createTrajectory(d.fields, domain, cache_id=0)
        s.create_pool([sample])
        assert sample._pointwise_pending
        seen = []

        def stop(snapshot, epoch):
            seen.append(('du/dx0', (1.0,)) in g.tensor_cache.memory_default[sample.ID])
            return True

        s.optimizer_exec_params['early_stopping_callback'] = stop
        s.fit([sample])
        assert seen == [True]


def test_prediction_grids_keep_trajectory_ids():
    from types import SimpleNamespace
    from epde.evaluators import simple_function_evaluator
    x = np.arange(4.)
    factor = SimpleNamespace(params_description={0: {'name': 'power'}}, params=[2],
                             predict_with_ann=lambda axes: sum(axes))
    values = simple_function_evaluator(factor, grids={'first': x, 'second': [x, x]})
    np.testing.assert_equal(values['first'], x ** 2)
    np.testing.assert_equal(values['second'], (2 * x) ** 2)
    np.testing.assert_equal(simple_function_evaluator(factor, [x])[-1], x ** 2)


def test_gpu_preprocessor_is_rejected_before_training_or_checkpoint_write(tmp_path):
    with search(default_preprocessor_type='ANN', preprocessor_kwargs={'device': 'cuda'}) as s:
        with pytest.raises(ValueError, match='CPU ANN preprocessing'):
            s.discover(observations(), checkpoint=tmp_path / 'run.zip')
    assert not (tmp_path / 'run.zip').exists()
