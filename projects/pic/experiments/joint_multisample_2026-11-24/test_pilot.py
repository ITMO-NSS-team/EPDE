"""Regression check for the shared-frequency objective used by this experiment."""
import numpy as np
from scipy.optimize import minimize_scalar


def test_frequency_profile_minimizes_the_equal_record_joint_residual(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'epoch_noise_families_2026-11-23'))
    from noise_suite import trial

    grids = [np.linspace(0, 12, n) for n in (240, 48, 64)]
    ys = [np.sin(t * omega) for t, omega in zip(grids, (1.0, 1.08, 1.08))]
    factor = SimpleNamespace(label='sin', params=np.array([0.98, 0.]),
                             params_description={0: {'name': 'freq'}, 1: {'name': 'dim'}})

    class Equation:
        def __init__(self):
            self.structure = [SimpleNamespace(structure=[factor], resetSavedState=lambda: None)]
            self.weights_final = np.array([1., 0.])

        def reset_state(self, _):
            pass

        def _invalidate_label_cache(self):
            pass

        def evaluate(self):
            omega = self.structure[0].structure[0].params[0]
            return dict(enumerate(ys)), {i: np.sin(t*omega)[:, None] for i, t in enumerate(grids)}

    def joint_sse(omega):
        xs = [np.sin(t*omega) for t in grids]
        coefficient = sum(np.mean(x*y) for x,y in zip(xs,ys)) / sum(np.mean(x*x) for x in xs)
        return sum(np.mean((y-coefficient*x)**2) for x,y in zip(xs,ys))

    search = trial.Trial(['u'], True)
    search.points = [('u',)]
    search.ranges[(0, 0.98)] = (0.9, 1.13, 0.23)
    monkeypatch.setattr(trial, 'plain_versions', lambda *args: [])
    parent = SimpleNamespace(vals={'u': Equation()})
    search.profile((None, 0, (0,), (0, 0.98), parent), ())
    profile, = search.profiles
    expected = minimize_scalar(joint_sse, bounds=(0.9, 1.13), method='bounded')
    np.testing.assert_allclose(profile['proposed'], expected.x, atol=3e-5, rtol=0)
    np.testing.assert_allclose(profile['best_sse'], joint_sse(profile['proposed']), rtol=1e-12)
    assert profile['proposed'] > 1.04


def test_cached_support_profiles_reach_late_equations(monkeypatch):
    from types import SimpleNamespace
    import support_frequency as frequency

    def term(label, omega=None):
        params = {'power': 1} if omega is None else {'power': 1, 'freq': omega, 'dim': 0}
        factor = SimpleNamespace(ftype=label, label=label, params=list(params.values()),
                                 params_description={i: {'name': k} for i, k in enumerate(params)}, param=params.get)
        return SimpleNamespace(structure=[factor])

    def equation(i, omega=1., intercept=False, reverse=False):
        target = term('u')
        features = [term('v' + str(i)), term('sin', omega)]
        return SimpleNamespace(main_var_to_explain='u', target=target, target_idx=0,
                               structure=[target, *(reversed(features) if reverse else features)],
                               weights_final=[1., 2., float(intercept)])

    profiler = frequency.SupportProfiles({(0, 1.): (.8, 1.2, .4), (0, 2.): (1.8, 2.2, .4)}, id)
    monkeypatch.setattr(profiler, 'loss', lambda *args: lambda omega: (omega - 1.1)**2)
    monkeypatch.setattr(frequency, 'fitted_versions', lambda *args: [])
    for i in range(14):
        profiler.epoch = i
        eq = equation(i)
        assert profiler.refine([eq], ()) == [eq]
    assert len(profiler.fitted) == 14
    assert len(profiler.profiles) == 28
    assert [p['epoch'] for p in profiler.profiles[-2:]] == [13, 13]
    np.testing.assert_allclose([p['proposed'] for p in profiler.profiles[-2:]], [1.1, 1.8])
    profiler.refine([equation(13, 1.1, reverse=True)], ())
    assert profiler.cache_hits == 1
    assert len(profiler.profiles) == 28
    profiler.refine([equation(13, 1.1, intercept=True)], ())
    assert len(profiler.profiles) == 30


def test_band_search_retains_seed_on_ties_and_skips_unidentifiable_fit():
    from support_frequency import bounds_minimum

    bracket = (.7, 1.3, .6)
    assert bounds_minimum(lambda w: 0., 1., bracket)[0] == 1.
    assert bounds_minimum(lambda w: np.inf, 1., bracket)[0] is None


def test_multistart_keeps_both_frequency_basins_and_one_sided_minima():
    from support_frequency import band_minima

    bracket = (0.95, 1.30, .35)
    loss = lambda w: min(2*(w-1.002)**2, .0004+(w-1.253)**2)
    minima, _ = band_minima(loss, 1.13, bracket)
    np.testing.assert_allclose(minima, [1.002, 1.253], rtol=0, atol=1e-6)
    minima, _ = band_minima(lambda w: min(w-.95, 1.30-w), 1.13, bracket)
    assert minima == [.95, 1.30]
    assert band_minima(lambda w: 1., 1.13, bracket)[0] == [1.13]
    assert band_minima(lambda w: np.nan, 1.13, bracket)[0] == []


def test_structural_stream_covers_term_combinations_and_protects_partner_target(monkeypatch):
    from itertools import combinations
    from types import SimpleNamespace
    import support_proposals as module

    def term(label):
        return SimpleNamespace(factors_labels=label, structure=(label,))

    pool = list(map(term, 'tabcdv'))
    target, *rest = pool
    eq = SimpleNamespace(target=target, structure=[target, *rest[:3]], main_var_to_explain='u', pool=pool)
    monkeypatch.setattr(module.fill_prune, 'factors_setting', lambda eq: 1)
    monkeypatch.setattr(module.fill_prune, 'library_terms', lambda pool, setting: pool)
    monkeypatch.setattr(module, '_related_to_target', lambda t, target: t.factors_labels == 'd')
    schedule = module.SupportProposals(None, None)
    added, deleted = [], set()
    for _ in range(5):
        for kind, terms in schedule.predictors(eq, 3, frozenset({'v'})):
            labels = tuple(term.factors_labels for term in terms)
            if kind == 'additive':
                added.append(labels)
            else:
                deleted.add(labels)
    assert deleted == set(combinations('abc', 2))
    assert sorted(added) == sorted(group for n in (1, 2, 3) for group in combinations('abc', n))


def test_all_resolved_frequency_fits_reach_the_candidate_list(monkeypatch):
    from types import SimpleNamespace
    import support_frequency as module

    bands = {(0, 1.2): (.9, 1.4, .5)}
    profiler = module.SupportProfiles(bands, lambda variant: variant.frequency, multistart=True)
    monkeypatch.setattr(profiler, 'loss', lambda *args: lambda w: min((w-1.04)**2, .0001+(w-1.32)**2))
    monkeypatch.setattr(module, 'fitted_versions', lambda eq, w, *args: [SimpleNamespace(frequency=w)])
    offered = profiler.fit_bands(SimpleNamespace(main_var_to_explain='u'), [], False, 0, (), 0)
    np.testing.assert_allclose([v.frequency for _, v in offered], [1.04, 1.32], atol=1e-6)
    assert profiler.profiles[0]['fitted_points'] == [i for i, _ in offered]


def test_combined_front_keeps_equal_vectors_and_recording_tradeoffs(monkeypatch):
    import sys
    from types import SimpleNamespace
    from report import combined_front

    old_vectors = [[.1, .1, .1, .2], [.3, .1, .1, .1]]
    old = dict(dataset='duffing', arm='repeat3', draw=0, frequency_policy='per_support',
               labels=[[['u', []]]], systems=[[0], [1]], front=[0, 1],
               points=[['u', 0, [[0, -1]], 0., v[-1], v[:-1]] for v in old_vectors])
    current = dict(dataset='duffing', arm='repeat3', draw=0, frequency_policy='systematic', front=[0, 1, 2])
    vectors = np.array([[.1, .1, .1, .2], [.32, .13, .1, .13], [.4, .05, .1, .1]])
    def nondominated(vectors):
        return [not np.any(np.all(vectors <= v, axis=1) & np.any(vectors < v, axis=1)) for v in vectors]
    monkeypatch.setitem(sys.modules, 'compare', SimpleNamespace(
        structural_matches=lambda *args: (np.ones(2, dtype=bool), None, None), system_front=nondominated))
    members = combined_front(old, current, vectors, np.ones(3, dtype=bool), None, {(('u',),)})
    assert {(r['policy'], r['system_index']) for r in members} == {
        ('per_support', 0), ('per_support', 1), ('systematic', 0), ('systematic', 2)}


def test_spectral_profile_matches_native_weak_rows_for_unequal_recordings(monkeypatch):
    import epde
    from epde.interface.prepared_tokens import TrigonometricTokens
    from epde.preprocessing import weak_form as wf
    from epde.structure.main_structures import Equation, Term
    from epde.structure.weak_rows import WeakRows
    import support_frequency as frequency

    with epde.EpdeSearch(use_solver=False, system_rps_mode='plain', engine='nsga2', evaluation_form='weak') as search:
        search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
        fields, plans, trajectories = {}, {}, []
        for i, n in enumerate([89, 141]):
            t = np.linspace(0, 10, n)
            data = {'u': np.cos(t*(1.12 + .04*i)) + .3*np.sin(2*t)}
            domain = search.createDomain(t, boundary_width=1, ID=i)[1]
            trajectories.append(search.createTrajectory(data, domain, cache_id=i)[1])
            fields[i] = wf.Fields(data, [t])
            plans[i, 'u'] = wf.WindowPlan(1, (1,), (3,), (2,), (1,), wf.lattice_shape((n-2,), (3,), (2,)), 0.)
        search.create_pool(data=trajectories, additional_tokens=[TrigonometricTokens(freq_values=[1., 2.])],
                           max_deriv_order=(1,))
        rows = WeakRows(fields, plans, list(fields))
        import epde.globals as global_var
        monkeypatch.setattr(global_var.samples_manager, 'weak', rows)
        monkeypatch.setattr(frequency.plain_fit, '_linear_versions', lambda eq, **kwargs: [eq])
        pool = search.pool
        sine, cosine = [pool.create(label=label, power=1, freq=1., dim=0)[1] for label in ('sin', 'cos')]
        predictors = [Term(pool, passed_term=[sine]), Term(pool, passed_term='u'),
                      Term(pool, passed_term=[pool.create(label='u', power=1)[1], cosine], max_factors_in_term=2)]
        eq = Equation(pool, [Term(pool, passed_term='du/dx0'), *predictors], 'u',
                      {'max_terms_number': {'value': 4}, 'max_factors_in_term': {'value': 2}})
        eq.target_idx = 0
        profiler = frequency.SupportProfiles({}, id)
        for intercept in (False, True):
            loss = profiler.loss(eq, predictors, intercept)
            for omega in (1., 1.11, 1.9):
                fitted, = frequency.fitted_versions(eq, omega, (), intercept)
                y, x = fitted.evaluate()
                blocks = [np.column_stack((v, np.ones(len(v)))) if intercept else v for v in x.values()]
                design = np.vstack([block / np.sqrt(len(block)) for block in blocks])
                observed = np.concatenate([v.ravel() / np.sqrt(len(v)) for v in y.values()])
                residual = observed - design @ np.linalg.lstsq(design, observed, rcond=None)[0]
                np.testing.assert_allclose(loss(omega), residual @ residual, rtol=1e-10, atol=1e-12)
