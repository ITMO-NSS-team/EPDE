"""The 'auto' settings: ``boundary_width``, token ``dimensionality``, ``population_size`` and
``training_epochs`` -- and the 'poly' preprocessor's window and degree reconciled with the
derivative orders (``PolynomialDeriv.resolve``), which ``boundary_width='auto'`` reads.

An explicit value of any of them behaves as before.
"""
import itertools
import warnings

import numpy as np
import pytest

import epde
import epde.globals as global_var
import epde.interface.search_config as search_config
from epde.interface.interface import EpdeSearch
from epde.interface.prepared_tokens import GridTokens, TrigonometricTokens
from epde.interface.search_config import collect_overrides, load_search_config
from epde.preprocessing.deriv_calculators import (AbstractDeriv, AdaptiveFiniteDeriv,
                                                  ForwardIncrementDeriv, PolynomialDeriv,
                                                  SpectralDeriv)
from epde.preprocessing.preprocessor import PreprocessingPipe
from epde.preprocessing.smoothers import GaussianSmoother, PlaceholderSmoother
from epde.structure.domain import resolve_evaluation_stride


@pytest.fixture(autouse=True)
def _forget_registered_samples():
    """Leave no trajectory in the global samples manager: tests that resolve from it (the vcoef
    microscale) would read this module's fields. The manager exists only once some search
    has initialised the caches -- not yet when this module's pure tests run first."""
    yield
    manager = getattr(global_var, 'samples_manager', None)
    if manager is not None:
        manager.reset()


def _search(preprocessor='FD', preprocessor_kwargs=None, **kwargs):
    kwargs.setdefault('verbose_params', {'show_iter_idx': False})
    search = epde.EpdeSearch(**kwargs)
    search.set_preprocessor(default_preprocessor_type=preprocessor,
                            preprocessor_kwargs=preprocessor_kwargs or {})
    return search


def _trajectory_1d(search, n=80, cache_id=0, **domain_kwargs):
    t = np.linspace(0, 4, n)
    domain = search.createDomain(t, ID=cache_id, **domain_kwargs)[1]
    return search.createTrajectory({'u': np.sin(2 * t) + 0.3 * t}, domain, cache_id=cache_id)[1]


def _trajectory_2d(search, nt=40, nx=48, cache_id=0, **domain_kwargs):
    T, X = np.meshgrid(np.linspace(0, 1, nt), np.linspace(0, 2 * np.pi, nx, endpoint=False),
                       indexing='ij')
    domain = search.createDomain([T, X], ID=cache_id, **domain_kwargs)[1]
    return search.createTrajectory({'u': np.exp(T) * np.cos(2 * X)}, domain, cache_id=cache_id)[1]


def _floor_free_rows():
    """The fewest 1-D lattice rows no shipped instability metric's row floor
    refuses at a default search's width: what a row budget must keep for the
    pool build to pass whichever metric is the default."""
    from epde.operators.common.survival import configured_width, row_floor
    width = configured_width(load_search_config().search_space)
    metrics = search_config.METRIC_MENUS['instability_metric']
    return next(n for n in itertools.count(1)
                if all(row_floor(metric, (n,), width) is None for metric in metrics))


def _controlled_pool(monkeypatch, distinct):
    """Every system drawn (``SoEq.create``, at construction and at every
    re-draw) takes the next of ``distinct`` labels in turn, and a system's
    label is that one, whatever the right-part selection leaves of its
    structure: the pool yields exactly ``distinct`` systems under any metric
    and data. The population machinery reads systems only through
    ``equations_labels``."""
    from epde.structure.main_structures import SoEq
    draws = itertools.count()
    create = SoEq.create

    def labelled_create(self, *args, **kwargs):
        out = create(self, *args, **kwargs)
        self._pool_label = next(draws) % distinct
        return out
    monkeypatch.setattr(SoEq, 'create', labelled_create)
    monkeypatch.setattr(SoEq, 'equations_labels', property(lambda self: (('system', self._pool_label),)))


def _reach(err):
    """Smallest r with the error beyond r samples from either end <= 2 x the middle-half max."""
    n = err.size
    mid = err[n // 4: n - n // 4].max()
    dist = np.minimum(np.arange(n), np.arange(n)[::-1])
    return next(r for r in range(n // 2) if err[dist >= r].max() <= 2 * mid)


# ---------------------------------------------------------------------------
# 0. poly: window and degree reconciled with the derivative orders
# ---------------------------------------------------------------------------
class TestPolyResolve:

    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
    def test_auto_keeps_the_window_of_nine_for_every_supported_order(self, order):
        assert PolynomialDeriv.resolve(order, (64,)) == (9, None)

    def test_auto_differentiates_exactly_as_the_explicit_nine(self):
        t = np.linspace(0, 1, 40)
        f = np.exp(2 * t)
        for order in (1, 2, 3):
            np.testing.assert_array_equal(PolynomialDeriv()(f, [t], order, 1, 'auto', None),
                                          PolynomialDeriv()(f, [t], order, 1, 9, None))

    def test_the_default_builder_window_is_auto(self):
        search = _search('poly')
        assert search.preprocessor_pipeline.deriv_calculator_kwargs['polynomial_window'] == 'auto'

    @pytest.mark.parametrize('kwargs, match', [
        (dict(max_order=6), 'up to order 5'),
        (dict(max_order=3, poly_order=2), 'identically zero'),
        (dict(max_order=1, polynomial_window=8), 'odd'),
        (dict(max_order=1, polynomial_window=65), 'does not fit'),
        (dict(max_order=3, polynomial_window=3), 'too narrow'),
    ])
    def test_conflicts_with_the_orders_are_refused(self, kwargs, match):
        max_order = kwargs.pop('max_order')
        with pytest.raises(ValueError, match=match):
            PolynomialDeriv.resolve(max_order, (64,), **kwargs)

    def test_a_window_equal_to_the_degree_still_drops_one_degree(self):
        # order 4 -> degree 5 = window: cheb warns and fits degree 4, still order 4
        assert PolynomialDeriv.resolve(4, (64,), polynomial_window=5) == (5, None)

    def test_short_axes_shrink_the_auto_window_or_refuse(self):
        assert PolynomialDeriv.resolve(1, (9,))[0] == 7     # 9 // 2 + 6 > 9
        assert PolynomialDeriv.resolve(2, (7,))[0] == 3
        with pytest.raises(ValueError, match='too short'):
            PolynomialDeriv.resolve(1, (6,))

    def test_an_axis_without_derivatives_constrains_nothing(self):
        assert PolynomialDeriv.resolve((0, 2), (4, 64)) == (9, None)

    def test_the_edge_band_is_the_same_at_every_order(self):
        data = np.zeros((40, 50))
        for order in (1, 3, 5):
            assert PolynomialDeriv().edge_reach(data, (0, order)) == [0, 5]


# ---------------------------------------------------------------------------
# 1a. edge reach per preprocessor
# ---------------------------------------------------------------------------
class TestEdgeReach:

    def test_finite_differences_reach_the_order(self):
        assert AdaptiveFiniteDeriv().edge_reach(np.zeros((40, 50)), (1, 3)) == [1, 3]

    @pytest.mark.parametrize('calculator, call', [
        ('FD', lambda f, t, k: AdaptiveFiniteDeriv()(f, [t], k)),
        ('poly', lambda f, t, k: PolynomialDeriv()(f, [t], k, 1, 'auto', None)),
    ])
    def test_the_reach_is_where_the_edge_error_ends(self, calculator, call):
        t = np.linspace(0, 1, 128)
        f = np.exp(2 * t)
        out = call(f, t, 3)
        for k in range(3):
            ref = 2 ** (k + 1) * f
            err = np.abs(out[:, k] - ref) / ref.max()
            claimed = (AdaptiveFiniteDeriv().edge_reach(f, k + 1)[0] if calculator == 'FD'
                       else PolynomialDeriv().edge_reach(f, k + 1)[0])
            assert _reach(err) <= claimed

    def test_forward_increments_reach_lag_times_order_in_time(self):
        reach = ForwardIncrementDeriv().edge_reach(np.zeros((40, 50)), (2, 1), lag=2, time_axis=0)
        assert reach == [4, 1]

    def test_spectral_reach_by_periodicity_and_order(self):
        T, X = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 2 * np.pi, 48, endpoint=False),
                           indexing='ij')
        u = np.exp(T) * np.cos(2 * X)
        sd = SpectralDeriv()
        assert sd.edge_reach(u, (1, 2)) == [0, 0]           # x periodic, t order 1
        assert sd.edge_reach(u, (2, 2)) == [1, 0]           # the end sample at order 2
        assert sd.edge_reach(u, (2, 2), periodic=False) == [1, 1]
        with pytest.warns(global_var.EPDEUsageWarning, match='order 3'):
            assert sd.edge_reach(u, (3, 0)) == [4, 0]       # 10% of 40 samples

    def test_smoothing_adds_its_kernel_radius(self):
        assert GaussianSmoother().edge_reach((40, 50), sigma=1, include_time=False) == [0, 4]
        assert GaussianSmoother().edge_reach((40, 50), sigma=2, include_time=True) == [8, 8]
        pipe = PreprocessingPipe()
        pipe.smoother, pipe.smoother_args = GaussianSmoother(), ()
        pipe.smoother_kwargs = {'sigma': 1, 'include_time': False}
        pipe.deriv_calculator, pipe.deriv_calculator_args = PolynomialDeriv(), ()
        pipe.deriv_calculator_kwargs = {'grid': None, 'mp_poolsize': 1,
                                        'polynomial_window': 'auto', 'poly_order': None}
        assert pipe.edge_reach(np.zeros((40, 50)), (1, 1)) == [5, 9]

    def test_a_calculator_that_does_not_say_is_unknown(self):
        class Custom(AbstractDeriv):
            pass
        pipe = PreprocessingPipe()
        pipe.smoother, pipe.smoother_args, pipe.smoother_kwargs = PlaceholderSmoother(), (), {}
        pipe.deriv_calculator, pipe.deriv_calculator_args = Custom(), ()
        assert pipe.edge_reach(np.zeros(10), 1) is None


# ---------------------------------------------------------------------------
# 1b. boundary_width='auto' through the search
# ---------------------------------------------------------------------------
class TestAutoBoundary:

    def test_the_default_is_resolved_when_the_pool_is_built(self):
        # FD's default five-point stencil reaches two samples per order (np.gradient: one).
        search = _search('FD')
        trajectory = _trajectory_1d(search)
        domain = trajectory._domain
        assert domain.boundary_auto and domain.boundary_pending
        search.create_pool(data=[trajectory], max_deriv_order=(2,), additional_tokens=[])
        assert not domain.boundary_pending
        assert domain.boundary_width_per_axis == [4]
        assert tuple(domain.inner_shape) == (72,)
        assert domain.g_func_mask.sum() == 72
        second_order = _search('FD', {'accuracy': 2})
        trajectory = _trajectory_1d(second_order)
        second_order.create_pool(data=[trajectory], max_deriv_order=(2,), additional_tokens=[])
        assert trajectory._domain.boundary_width_per_axis == [2]

    def test_an_explicit_width_is_unchanged(self):
        search = _search('FD')
        trajectory = _trajectory_1d(search, boundary_width=10)
        domain = trajectory._domain
        assert not domain.boundary_auto and not domain.boundary_pending
        assert tuple(domain.inner_shape) == (60,)
        search.create_pool(data=[trajectory], max_deriv_order=(1,), additional_tokens=[])
        assert domain.boundary_width == 10 and tuple(domain.inner_shape) == (60,)

    def test_poly_by_default_keeps_the_old_five(self):
        search = _search('poly', {'mp_poolsize': 1})
        trajectory = _trajectory_1d(search)
        search.create_pool(data=[trajectory], max_deriv_order=(1,), additional_tokens=[])
        assert trajectory._domain.boundary_width_per_axis == [5]

    def test_a_periodic_spectral_axis_keeps_every_sample(self):
        search = _search('spectral')
        trajectory = _trajectory_2d(search)
        search.create_pool(data=[trajectory], max_deriv_order=(1, 2), additional_tokens=[])
        assert trajectory._domain.boundary_width_per_axis == [0, 0]
        assert tuple(trajectory._domain.inner_shape) == (40, 48)

    def test_forward_increments(self):
        search = _search('FD', {'scheme': 'forward', 'lag': 2})
        trajectory = _trajectory_1d(search)
        search.create_pool(data=[trajectory], max_deriv_order=(1,), additional_tokens=[])
        assert trajectory._domain.boundary_width_per_axis == [2]

    def test_precomputed_derivatives_need_no_boundary(self):
        search = _search('FD')
        t = np.linspace(0, 4, 80)
        domain = search.createDomain(t, ID=0)[1]
        derivs = np.stack([2 * np.cos(2 * t) + 0.3], axis=1)
        trajectory = search.createTrajectory({'u': np.sin(2 * t) + 0.3 * t}, domain, cache_id=0,
                                             derivs=derivs)[1]
        search.create_pool(data=[trajectory], max_deriv_order=(1,), additional_tokens=[])
        assert domain.boundary_width_per_axis == [0]

    def test_a_pipeline_that_does_not_say_falls_back_to_five(self):
        search = _search('FD')
        trajectory = _trajectory_1d(search)

        class Opaque(AdaptiveFiniteDeriv):
            def edge_reach(self, *args, **kwargs):
                return None
        trajectory._preprocessor_pipeline.deriv_calculator = Opaque()
        with pytest.warns(global_var.EPDEUsageWarning, match='does not say'):
            search.create_pool(data=[trajectory], max_deriv_order=(1,), additional_tokens=[])
        assert trajectory._domain.boundary_width_per_axis == [5]

    def test_it_is_set_once_and_a_wider_need_later_is_refused(self):
        search = _search('FD')
        trajectory = _trajectory_1d(search)
        search.create_pool(data=[trajectory], max_deriv_order=(1,), additional_tokens=[])
        search.create_pool(data=[trajectory], max_deriv_order=(1,), additional_tokens=[])
        assert trajectory._domain.boundary_width_per_axis == [2]      # FD's five-point stencil, order 1
        with pytest.raises(ValueError, match='automatically from an earlier build'):
            search.create_pool(data=[trajectory], max_deriv_order=(3,), additional_tokens=[])

    def test_an_array_test_function_needs_an_explicit_width(self):
        search = _search('FD')
        t = np.linspace(0, 4, 80)
        with pytest.raises(ValueError, match="boundary_width='auto'"):
            search.createDomain(t, gfunction=np.ones(80))

    def test_an_explicit_width_narrower_than_the_reach_warns(self):
        search = _search('FD')
        trajectory = _trajectory_1d(search, boundary_width=1)
        with pytest.warns(global_var.EPDEUsageWarning, match='narrower than the edge reach'):
            search.create_pool(data=[trajectory], max_deriv_order=(2,), additional_tokens=[])

    def test_the_row_budget_lattice_follows_the_resolved_width(self):
        """The strides are laid on the interior box the resolved width leaves,
        not on the whole grid an 'auto' domain has before the pool build. The
        box holds twice the fewest rows every shipped instability metric's row
        floor takes (the pool build checks it), so stride 2 keeps exactly
        that many; on the whole grid it would keep more."""
        rows = _floor_free_rows()
        n = 2 * rows + 2 * 4              # FD (five-point), order 2: a width of 4 (see above)
        search = _search('FD', evaluation_stride=2)
        trajectory = _trajectory_1d(search, n=n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            search.create_pool(data=[trajectory], max_deriv_order=(2,), additional_tokens=[])
        domain = trajectory._domain
        assert domain.boundary_width_per_axis == [4]
        assert domain.evaluation_stride == resolve_evaluation_stride((n,), [4], 2) == (2,)
        assert tuple(int(k) for k in domain.inner_shape) == (rows,)
        assert -(-n // 2) > rows


# ---------------------------------------------------------------------------
# 2. token dimensionality='auto'
# ---------------------------------------------------------------------------
class TestAutoDimensionality:

    def _pool(self, search, trajectory, tokens):
        search.create_pool(data=[trajectory], max_deriv_order=(1,) * len(trajectory._domain.initial_shape),
                           additional_tokens=tokens)

    def test_one_dimensional_data_gets_one_axis(self):
        search = _search('FD')
        trig, grid = TrigonometricTokens(freq=(1.9, 2.1)), GridTokens(['x_0'])
        self._pool(search, _trajectory_1d(search), [trig, grid])
        assert trig.token_family.token_params['dim'] == (0, 0)
        assert grid.token_family.token_params['dim'] == (0, 0)

    def test_two_dimensional_data_matches_the_old_default(self):
        search = _search('FD')
        trig = TrigonometricTokens(freq=(1.9, 2.1))
        self._pool(search, _trajectory_2d(search), [trig])
        explicit = TrigonometricTokens(freq=(1.9, 2.1), dimensionality=1)
        assert list(trig.token_family.token_params.items()) == \
            list(explicit.token_family.token_params.items())

    def test_three_dimensional_data(self):
        search = _search('FD')
        axes = np.meshgrid(np.linspace(0, 1, 12), np.linspace(0, 1, 13), np.linspace(0, 1, 14),
                           indexing='ij')
        domain = search.createDomain(axes, ID=0)[1]
        trajectory = search.createTrajectory({'u': axes[0] + axes[1] * axes[2]}, domain, cache_id=0)[1]
        grid = GridTokens(['x'])
        self._pool(search, trajectory, [grid])
        assert grid.token_family.token_params['dim'] == (0, 2)

    def test_frequencies_from_data_look_along_every_axis(self):
        search = _search('FD')
        trig = TrigonometricTokens(freq_from_data=1)
        self._pool(search, _trajectory_2d(search), [trig])
        assert trig._freq_axes == (0, 1)

    def test_nothing_is_drawn_before_the_data_is_known(self):
        with pytest.raises(RuntimeError):
            TrigonometricTokens(freq=(1.9, 2.1)).token_family.create()

    @pytest.mark.parametrize('value', ['yes', -1])
    def test_bad_values_are_refused(self, value):
        with pytest.raises(ValueError):
            TrigonometricTokens(dimensionality=value)

    def test_too_many_grid_labels_for_the_data_are_refused(self):
        search = _search('FD')
        with pytest.raises(ValueError, match='one per axis'):
            self._pool(search, _trajectory_1d(search), [GridTokens(['x_0', 'x_1'])])

    def test_the_pool_is_reused_across_fits(self, monkeypatch):
        search = _search('FD', tokens=[{'family': 'trigonometric', 'freq': [0.5, 1.5]}])
        builds = []
        original = search._build_pool
        monkeypatch.setattr(search, '_build_pool', lambda *a, **k: (builds.append(1), original(*a, **k)))
        monkeypatch.setattr(search, '_run_optimization', lambda *a, **k: None)
        trajectory = _trajectory_1d(search)
        search.fit(data=[trajectory], max_deriv_order=(1,))
        search.fit(data=[trajectory], max_deriv_order=(1,))
        assert len(builds) == 1


# ---------------------------------------------------------------------------
# 3. the early stop (early_stopping_patience) under the user's training_epochs
# ---------------------------------------------------------------------------
class _Solution:
    def __init__(self, label):
        # The stagnation key reads the structures (the same labels outside plain mode).
        self.equations_labels = self.structure_labels = label


class _Stub:
    """Enough of EpdeSearch for _stagnation_callback / _resolved_budget."""
    def __init__(self, levels=None, multiobjective=True, niching='system', n_eqs=1):
        self.optimizer = type('O', (), {})()
        self.optimizer.pareto_levels = type('P', (), {'levels': levels or [[]]})()
        self.multiobjective_mode = multiobjective
        self._config = load_search_config(overrides=collect_overrides(niching=niching))
        self.pool = type('Pool', (), {'families_demand_equation': [None] * n_eqs})()
        self._stagnation_callback = EpdeSearch._stagnation_callback.__get__(self)


class TestAutoEpochs:

    def test_stops_after_patience_unchanged_epochs(self):
        stub = _Stub([[_Solution('a'), _Solution('b')]])
        callback = stub._stagnation_callback(3)
        assert [callback([], e) for e in range(4)] == [False, False, False, True]

    def test_a_change_restarts_the_count(self):
        stub = _Stub([[_Solution('a')]])
        callback = stub._stagnation_callback(2)
        assert not callback([], 0) and not callback([], 1)
        stub.optimizer.pareto_levels.levels = [[_Solution('c')]]
        assert not callback([], 2) and not callback([], 3) and callback([], 4)

    def test_the_user_callback_is_still_asked(self):
        stub = _Stub([[_Solution('a')]])
        asked = []
        callback = stub._stagnation_callback(5, lambda snapshot, epoch: asked.append(epoch) or epoch == 1)
        assert not callback([], 0) and callback([], 1) and asked == [0, 1]

    def test_resolved_budget(self):
        stub = _Stub()
        stub.optimizer_init_params = {'pop_size': 'auto', 'H': None, 'neighbors_number': 3}
        stub.optimizer_exec_params = {'epochs': 40, 'early_stopping_callback': None, 'patience': 5}
        init, run = EpdeSearch._resolved_budget(stub)
        assert init['pop_size'] == search_config.AUTO_POPULATION_CEILING
        assert init['H'] == search_config.AUTO_POPULATION_CEILING - 1
        assert init['population_floor'] == search_config.AUTO_POPULATION_FLOOR
        # The user's limit stands; the patience becomes the stagnation callback.
        assert run['epochs'] == 40 and callable(run['early_stopping_callback']) and 'patience' not in run
        assert stub.optimizer_init_params['pop_size'] == 'auto'          # resolved per fit

    def test_explicit_values_pass_through(self):
        stub = _Stub()
        user = lambda snapshot, epoch: False
        stub.optimizer_init_params = {'pop_size': 6, 'H': 5, 'neighbors_number': 3}
        stub.optimizer_exec_params = {'epochs': 7, 'early_stopping_callback': user, 'patience': None}
        init, run = EpdeSearch._resolved_budget(stub)
        assert init == stub.optimizer_init_params
        assert run == {'epochs': 7, 'early_stopping_callback': user}     # no patience: every epoch

    def test_single_objective(self):
        stub = _Stub(multiobjective=False)
        stub.optimizer_init_params = {'pop_size': 'auto'}
        stub.optimizer_exec_params = {'epochs': 30, 'patience': 5}
        init, run = EpdeSearch._resolved_budget(stub)
        assert run == {'epochs': 30, 'patience': 5}
        assert init['population_floor'] == search_config.AUTO_POPULATION_FLOOR

    def test_the_defaults_and_the_refusals(self):
        evolution = load_search_config().evolution
        assert (evolution.training_epochs, evolution.early_stopping_patience) == (100, 5)
        with pytest.raises(ValueError, match='training_epochs must be an int'):
            load_search_config(overrides={'training_epochs': 'auto'})
        for patience in (0, 2.5, 'auto'):
            with pytest.raises(ValueError, match='early_stopping_patience'):
                load_search_config(overrides={'early_stopping_patience': patience})
        assert load_search_config(overrides={'early_stopping_patience': None}).evolution.early_stopping_patience is None

    def test_the_setters_take_the_patience(self):
        search = EpdeSearch(verbose_params={'show_iter_idx': False}, training_epochs=12,
                            early_stopping_patience=3)
        assert search.optimizer_exec_params['epochs'] == 12 and search.optimizer_exec_params['patience'] == 3
        search.set_moeadd_params(training_epochs=8, early_stopping_patience=None)
        assert search.optimizer_exec_params['epochs'] == 8 and search.optimizer_exec_params['patience'] is None
        with pytest.raises(ValueError, match='early_stopping_patience'):
            search.set_moeadd_params(early_stopping_patience=0)

    def test_the_floor_covers_every_equation_under_per_equation_niching(self, monkeypatch):
        monkeypatch.setattr(search_config, 'AUTO_POPULATION_FLOOR', 2)
        stub = _Stub(niching='per_equation', n_eqs=3)
        stub.optimizer_init_params = {'pop_size': 'auto', 'H': None, 'neighbors_number': 1}
        stub.optimizer_exec_params = {'epochs': 3}
        assert EpdeSearch._resolved_budget(stub)[0]['population_floor'] == 6

    def test_the_single_objective_stop_condition(self):
        from epde.optimizers.single_criterion.ea_stop_conds import IterationLimit
        cond = IterationLimit()
        cond.reset(limit=50, patience=2)
        population = [_Solution('a')]
        results = []
        for _ in range(4):
            results.append(cond.check())
            cond.observe(population)
        assert results == [False, False, False, True]
        cond.reset(limit=3)                                   # no patience: the plain limit
        assert [cond.check() for _ in range(4)] == [False, False, False, True]


# ---------------------------------------------------------------------------
# 4. population_size='auto'
# ---------------------------------------------------------------------------
class TestAutoPopulation:

    def test_config_values_are_checked(self):
        for key, bad in (('population_size', 1), ('population_size', 'big'),
                         ('training_epochs', -1), ('boundary_width', 'wide'),
                         ('boundary_width', [1, -2])):
            with pytest.raises(ValueError):
                load_search_config(overrides=collect_overrides(**{key: bad}))

    @pytest.mark.parametrize('distinct', [3, 7])
    def test_the_population_ends_where_the_pool_runs_out_of_new_systems(self, monkeypatch, distinct):
        """The pool yields exactly ``distinct`` systems (:func:`_controlled_pool`):
        the population is all of them, each once. How many distinct systems a
        real pool yields is the right-part selection's business and changes
        with the instability metric and the data, so it is fixed here."""
        _controlled_pool(monkeypatch, distinct)
        monkeypatch.setattr(search_config, 'AUTO_POPULATION_FLOOR', 2)
        monkeypatch.setattr(search_config, 'AUTO_POPULATION_CEILING', 40)
        np.random.seed(0)
        search = _search('FD', training_epochs=1, equation_terms_max_number=3,
                         equation_factors_max_number=2, neighbors_number=1,
                         operators={'InitialParetoLevelSorting': {'uniqueness_attempt_limit': 10}})
        trajectory = _trajectory_1d(search)
        search.fit(data=[trajectory], max_deriv_order=(1,),
                   additional_tokens=[GridTokens(['x_0'], max_power=2)])
        optimizer = search.optimizer
        assert not getattr(optimizer.pareto_levels, '_init_collapsed', False)
        size = len(optimizer.pareto_levels.population)
        assert size == distinct
        assert optimizer.weights.shape[0] == size == len(optimizer.neighborhood_lists)
        assert len({sol.equations_labels for sol in optimizer.pareto_levels.population}) == size

    def test_a_pool_drier_than_the_floor_fills_the_floor_with_duplicates(self, monkeypatch, capsys):
        """One distinct system under a floor of 3: the duplicates fill the
        floor's slots (MOEA/D needs them), no more, and the report says how
        many of them are distinct."""
        _controlled_pool(monkeypatch, 1)
        monkeypatch.setattr(search_config, 'AUTO_POPULATION_FLOOR', 3)
        monkeypatch.setattr(search_config, 'AUTO_POPULATION_CEILING', 40)
        np.random.seed(0)
        search = _search('FD', training_epochs=0, equation_terms_max_number=3,
                         equation_factors_max_number=2, neighbors_number=1,
                         operators={'InitialParetoLevelSorting': {'uniqueness_attempt_limit': 5}})
        search.fit(data=[_trajectory_1d(search)], max_deriv_order=(1,),
                   additional_tokens=[GridTokens(['x_0'], max_power=2)])
        population = search.optimizer.pareto_levels.population
        assert len(population) == 3
        assert len({sol.equations_labels for sol in population}) == 1
        assert 'Population size: 3 systems, 1 distinct' in capsys.readouterr().out

    def test_single_objective_stops_adding_once_new_individuals_run_out(self, monkeypatch):
        from epde.optimizers.single_criterion import optimizer as so

        class Few:
            def __init__(self, **kwargs):
                self.draws = 0

            def create(self, **kwargs):
                self.draws += 1
                return ('system', self.draws % 3)          # three distinct individuals

        monkeypatch.setattr(so, 'SystemsPopulationConstructor', Few)
        built = so.SimpleOptimizer({}, pop_size=10, solution_params={}, population_floor=2)
        assert len(built.population.population) == 3
        with pytest.raises(RuntimeError):
            so.SimpleOptimizer({}, pop_size=10, solution_params={}, population_floor=5)
