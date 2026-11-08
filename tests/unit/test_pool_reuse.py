"""When a token pool may be reused, and what a rebuild must refresh.

``fit`` reuses ``self.pool`` when its key matches. The key described the pool
but not the data, so several things went wrong before:

* config-declared token families were resolved in ``fit`` and again inside
  ``create_pool``: each sat in the pool twice, and the key ``fit`` compared
  never matched the stored one;
* a second ``fit`` on DIFFERENT data with the same variable names reused the
  pool and never built or registered the new trajectories, so it optimised
  against the old samples (or, their tensors released, raised KeyError);
* the end-of-fit release emptied the tensor cache, and ``Trajectory.build`` --
  a no-op for the same request -- did not re-upload, so even a second fit on
  the SAME data raised KeyError;
* the samples manager kept every trajectory ever registered, so a later fit
  evaluated old samples beside the new ones;
* the key held only each family's ``ftype``, so families with other ranges or
  grids reused a stale pool.
"""
import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.interface import EpdeSearch
from epde.interface.prepared_tokens import (TrigonometricTokens, PhasedSine1DTokens, ConstantToken,
                                            VelocityHEQTokens)


def _search(**kwargs):
    kwargs.setdefault('verbose_params', {'show_iter_idx': False})
    search = epde.EpdeSearch(**kwargs)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _trajectory(search, cache_id, phase=0.0, n=120, domain_id=None):
    grid = np.linspace(0, 4 * np.pi, n)
    domain = search.createDomain(grid, boundary_width=10, ID=cache_id if domain_id is None else domain_id)[1]
    data = np.sin(grid + phase) + 1.3 * np.cos(grid + phase)
    return search.createTrajectory({'u': data}, domain, cache_id=cache_id)[1]


def _no_search(search, monkeypatch):
    """Count pool builds and skip the evolution itself."""
    builds = []
    original = search._build_pool

    def counted(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(search, '_build_pool', counted)
    monkeypatch.setattr(search, '_run_optimization', lambda *args, **kwargs: None)
    return builds


def _registered():
    return list(global_var.samples_manager._traj.values())


def _observed(trajectory):
    return trajectory.get(('u', (1.0,)))


# ---------------------------------------------------------------------------
# config-declared families: resolved once
# ---------------------------------------------------------------------------

class TestConfigFamilies:

    SPEC = [{'family': 'trigonometric', 'freq': [0.5, 1.5], 'dimensionality': 0}]

    def test_a_config_family_enters_the_pool_once(self, monkeypatch):
        search = _search(tokens=self.SPEC)
        _no_search(search, monkeypatch)
        search.fit(data=[_trajectory(search, 0)], max_deriv_order=(1,))
        assert sum(family.ftype == 'trigonometric' for family in search.pool.families) == 1

    def test_the_pool_is_reused_for_the_same_request(self, monkeypatch):
        search = _search(tokens=self.SPEC)
        builds = _no_search(search, monkeypatch)
        trajectory = _trajectory(search, 0)
        search.fit(data=[trajectory], max_deriv_order=(1,))
        search.fit(data=[trajectory], max_deriv_order=(1,))
        assert len(builds) == 1

    def test_a_direct_create_pool_still_adds_config_families(self):
        search = _search(tokens=self.SPEC)
        search.create_pool(data=[_trajectory(search, 0)], max_deriv_order=(1,))
        assert sum(family.ftype == 'trigonometric' for family in search.pool.families) == 1

    def test_create_pool_takes_a_bare_trajectory(self):
        search = _search()
        trajectory = _trajectory(search, 0)
        search.create_pool(data=trajectory, max_deriv_order=(1,))
        assert _registered() == [trajectory]


# ---------------------------------------------------------------------------
# new data: rebuilt and registered
# ---------------------------------------------------------------------------

class TestNewData:

    def test_a_second_fit_on_other_data_builds_and_registers_it(self, monkeypatch):
        search = _search()
        builds = _no_search(search, monkeypatch)
        first, second = _trajectory(search, 0), _trajectory(search, 1, phase=0.7)
        search.fit(data=[first], max_deriv_order=(1,))
        search.fit(data=[second], max_deriv_order=(1,))
        assert len(builds) == 2
        assert second.built
        assert _registered() == [second]                                  # the first is gone

    def test_a_new_sample_under_an_old_cache_id_is_read_as_itself(self, monkeypatch):
        """The cache never overwrites a label: the second sample's data used to
        be shadowed by the first's, or registering it raised KeyError."""
        search = _search(free_tensor_cache_after_fit=False)
        _no_search(search, monkeypatch)
        first = _trajectory(search, 0, phase=0.0)
        search.fit(data=[first], max_deriv_order=(1,))
        second = _trajectory(search, 0, phase=0.7)
        search.fit(data=[second], max_deriv_order=(1,))
        assert _registered() == [second]
        expected = second.raw_field('u')[second._domain.g_func_mask]
        assert np.allclose(_observed(second), expected)
        assert not np.allclose(_observed(second), first.raw_field('u')[first._domain.g_func_mask])

    def test_a_passed_pool_still_gets_the_data_built_and_registered(self, monkeypatch):
        search = _search()
        _no_search(search, monkeypatch)
        search.fit(data=[_trajectory(search, 0)], max_deriv_order=(1,))
        pool = search.pool
        other = _trajectory(search, 1, phase=0.3)
        search.fit(data=[other], pool=pool, max_deriv_order=(1,))
        assert search.pool is pool
        assert other.built and _registered() == [other]


# ---------------------------------------------------------------------------
# the end-of-fit release
# ---------------------------------------------------------------------------

class TestAfterTheRelease:

    def test_a_released_trajectory_is_rebuilt(self, monkeypatch):
        search = _search()
        builds = _no_search(search, monkeypatch)
        trajectory = _trajectory(search, 0)
        search.fit(data=[trajectory], max_deriv_order=(1,))
        global_var.release_tensor_cache()                                  # what the end of a real fit does
        assert not trajectory.tensors_cached
        search.fit(data=[trajectory], max_deriv_order=(1,))
        assert len(builds) == 2
        assert trajectory.tensors_cached
        assert np.allclose(_observed(trajectory), trajectory.raw_field('u')[trajectory._domain.g_func_mask])

    def test_two_real_fits_on_the_same_data(self):
        """End to end, with the default release between them: used to KeyError."""
        search = _search(multiobjective_mode=False)
        search.set_singleobjective_params(population_size=4, training_epochs=1)
        trajectory = _trajectory(search, 0, n=100)
        for _ in range(2):
            search.fit(data=[trajectory], max_deriv_order=(2,), data_fun_pow=1,
                       equation_terms_max_number=3, equation_factors_max_number=1)
        assert search.equations(only_print=False)

    def test_the_release_also_forgets_the_derivative_codes(self):
        _search()                                                          # sets the caches up
        cache = global_var.tensor_cache
        cache._deriv_codes.append((([0], [0]), ('du/dx0', (1.0,))))
        global_var.release_tensor_cache()
        assert cache._deriv_codes == []

    def test_build_is_still_a_noop_while_the_tensors_live(self):
        search = _search()
        trajectory = _trajectory(search, 0)
        search.create_pool(data=[trajectory], max_deriv_order=(2,))
        families = trajectory.families
        search.create_pool(data=[trajectory], max_deriv_order=(2,))
        assert trajectory.families is families

    def test_trajectory_zero_does_not_take_another_samples_copy_for_its_own(self):
        """Every build also copies its variables into subcache 0, so after a
        release trajectory 0 found that subcache non-empty and kept the other
        sample's data."""
        search = _search()
        t1, t0 = _trajectory(search, 1, phase=0.6), _trajectory(search, 0, phase=0.0)
        search.create_pool(data=[t1, t0], max_deriv_order=(1,))
        global_var.release_tensor_cache()
        search.create_pool(data=[t1, t0], max_deriv_order=(1,))
        for trajectory in (t0, t1):
            assert np.allclose(_observed(trajectory),
                               trajectory.raw_field('u')[trajectory._domain.g_func_mask])
            assert trajectory.get(('du/dx0', (1.0,))) is not None

    def test_samples_sharing_a_cache_id_never_read_each_other(self, monkeypatch):
        """With the release off: A, then B on A's cache id, then A again."""
        search = _search(free_tensor_cache_after_fit=False)
        _no_search(search, monkeypatch)
        a = _trajectory(search, 0, phase=0.0)
        search.fit(data=[a], max_deriv_order=(1,))
        b = _trajectory(search, 0, phase=0.9)
        search.fit(data=[b], max_deriv_order=(1,))
        search.fit(data=[a], max_deriv_order=(1,))
        assert np.allclose(_observed(a), a.raw_field('u')[a._domain.g_func_mask])

    def test_a_trajectory_without_a_cache_id_survives_a_new_request(self):
        """createTrajectory's default cache_id=None: Cache.get read 'the first
        subcache', which was the None one only until a rebuild reordered them."""
        search = _search()
        grid = np.linspace(0, 4 * np.pi, 120)
        domain = search.createDomain(grid, boundary_width=10, ID=0)[1]
        trajectory = search.createTrajectory({'u': np.sin(grid) + 1.3 * np.cos(grid)}, domain)[1]
        search.create_pool(data=[trajectory], max_deriv_order=(1,))
        search.create_pool(data=[trajectory], max_deriv_order=(2,))
        assert trajectory.get(('du/dx0', (1.0,))) is not None
        assert np.allclose(_observed(trajectory), trajectory.raw_field('u')[domain.g_func_mask])


class TestFamiliesChange:

    def _stored(self, values):
        from epde import CacheStoredTokens
        return CacheStoredTokens(token_type='stored', token_labels=['f'], token_tensors={'f': values},
                                 params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                                 dimensionality=0, meaningful=True)

    def test_new_family_tensors_replace_the_old_ones(self):
        """The cache never overwrites a label: a rebuild for new families kept
        serving the old CacheStoredTokens tensor."""
        search = _search()
        trajectory = _trajectory(search, 0)
        grid = np.linspace(0, 4 * np.pi, 120)
        search.create_pool(data=[trajectory], additional_tokens=[self._stored(np.cos(grid))],
                           max_deriv_order=(1,))
        search.create_pool(data=[trajectory], additional_tokens=[self._stored(10 * np.cos(grid))],
                           max_deriv_order=(1,))
        assert np.allclose(trajectory.get(('f', (1.0,))), 10 * np.cos(grid)[trajectory._domain.g_func_mask])

    def test_a_status_flag_is_part_of_the_key(self):
        quiet = TrigonometricTokens(freq=(0.5, 1.5), dimensionality=0, meaningful=False)
        loud = TrigonometricTokens(freq=(0.5, 1.5), dimensionality=0, meaningful=True)
        assert EpdeSearch._family_key(quiet) != EpdeSearch._family_key(loud)

    def test_a_passed_pool_leaves_no_key_behind(self, monkeypatch):
        search = _search()
        builds = _no_search(search, monkeypatch)
        trajectory = _trajectory(search, 0)
        search.fit(data=[trajectory], max_deriv_order=(1,))
        search.fit(data=[trajectory], pool=search.pool, max_deriv_order=(1,))
        assert search.pool_params is None
        search.fit(data=[trajectory], max_deriv_order=(1,))
        assert len(builds) == 2                                            # rebuilt after the passed pool


# ---------------------------------------------------------------------------
# the key: a fingerprint per family
# ---------------------------------------------------------------------------

class TestFamilyKey:

    def test_equal_families_give_equal_keys(self):
        """Config families are rebuilt on every fit; equal ones must still match."""
        a = TrigonometricTokens(freq=(0.5, 1.5), dimensionality=0)
        b = TrigonometricTokens(freq=(0.5, 1.5), dimensionality=0)
        assert EpdeSearch._family_key(a) == EpdeSearch._family_key(b)

    @pytest.mark.parametrize('other', [
        dict(freq=(0.5, 2.5)),
        dict(freq=(0.5, 1.5), freq_num=3),
        dict(freq_values=[1.0]),
    ])
    def test_other_ranges_or_grids_give_other_keys(self, other):
        base = TrigonometricTokens(freq=(0.5, 1.5), dimensionality=0)
        assert EpdeSearch._family_key(base) != EpdeSearch._family_key(TrigonometricTokens(dimensionality=0,
                                                                                            **other))

    def test_an_installed_grid_changes_the_key(self):
        tokens = TrigonometricTokens(dimensionality=0, freq_from_data=1)
        before = EpdeSearch._family_key(tokens)
        tokens._token_family.install_param_grid('freq', [2.0])
        assert EpdeSearch._family_key(tokens) != before

    def test_a_different_range_rebuilds_through_fit(self, monkeypatch):
        search = _search()
        builds = _no_search(search, monkeypatch)
        trajectory = _trajectory(search, 0)
        search.fit(data=[trajectory], max_deriv_order=(1,),
                   additional_tokens=[TrigonometricTokens(freq=(0.5, 1.5), dimensionality=0)])
        search.fit(data=[trajectory], max_deriv_order=(1,),
                   additional_tokens=[TrigonometricTokens(freq=(2.5, 3.5), dimensionality=0)])
        assert len(builds) == 2


# ---------------------------------------------------------------------------
# token definitions
# ---------------------------------------------------------------------------

def _draw_freqs(tokens, n=40):
    family = tokens._token_family
    np.random.seed(0)
    return {float(family.create(label=family.tokens[0], all_vars=['u'])[1].params[1]) for _ in range(n)}


class TestFrequencyBounds:

    def test_integer_bounds_draw_real_frequencies(self):
        """A JSON [1, 3] used to send the draw to randint: only 1, 2 and 3."""
        drawn = _draw_freqs(TrigonometricTokens(freq=(1, 3), dimensionality=0))
        assert len(drawn) > 3 and not all(f.is_integer() for f in drawn)

    def test_integer_bounds_from_a_config(self):
        from epde.interface.search_config import build_tokens
        tokens = build_tokens([{'family': 'trigonometric', 'freq': [1, 3], 'dimensionality': 0}])[0]
        assert all(type(bound) is float for bound in tokens._token_family.token_params['freq'])
        assert not all(f.is_integer() for f in _draw_freqs(tokens))

    def test_the_phased_sine_too(self):
        tokens = PhasedSine1DTokens(freq=(1, 3))
        assert all(type(bound) is float for bound in tokens._token_family.token_params['freq'])
        assert not all(f.is_integer() for f in _draw_freqs(tokens))

    def test_the_phased_sine_refuses_a_bad_interval(self):
        with pytest.raises(ValueError, match='low <= high'):
            PhasedSine1DTokens(freq=(3.0, 1.0))
        with pytest.raises(ValueError, match='low <= high'):
            PhasedSine1DTokens(freq=(1.0,))

    def test_the_default_interval_is_unchanged(self):
        assert TrigonometricTokens()._token_family.token_params['freq'] == (np.pi / 2., 2 * np.pi)


def test_the_phased_sine_latex_matches_its_evaluator():
    """phased_sine_1d_np is sin(2*pi*(freq*x_0 + phase)); the LaTeX said 'freq x_1 + phase'."""
    family = PhasedSine1DTokens()._token_family
    form = family.latex_constructor('sine', power=(1, '1'), freq=(2.0, '2.0'), phase=(0.25, '0.25'))
    assert r'2\pi' in form and r'x_{0}' in form and 'x_{1}' not in form
    assert form == r'\sin\left(2\pi\left(2.0\,x_{0} + 0.25\right)\right)'


@pytest.mark.parametrize('make, name, expected', [
    (lambda: ConstantToken(values_range=(0.0, 10.0)), 'value', 0.1),
    (lambda: VelocityHEQTokens(param_ranges=[(0.0, 2.0)] * 14 + [(1.0, 5.0)]), 'p15', 0.2),
])
def test_tolerances_are_a_fraction_of_the_range(make, name, expected, monkeypatch):
    """The trig divide-for-multiply, in two more families: the tolerance was
    100x / 20x the whole range, so every value shared one label."""
    from epde.interface.token_family import TokenFamily
    # VelocityHEQTokens test-evaluates on a 2-D grid at construction; only the tolerance matters here.
    monkeypatch.setattr(TokenFamily, 'test_evaluator', lambda self, deriv=False: None)
    assert make()._token_family.equality_ranges[name] == pytest.approx(expected)
