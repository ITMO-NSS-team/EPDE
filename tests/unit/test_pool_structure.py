"""The token pool is described on ``create_pool``/``fit``, never on a trajectory.

``max_deriv_order``, ``data_fun_pow`` and ``deriv_fun_pow`` decide which token
FAMILIES exist. A pool has one set of families however many trajectories feed
it -- ``create_pool`` literally reads them off ``data[0]`` -- so stating them
per trajectory offered a choice that does not exist: three samples with three
different ``max_deriv_order`` would silently take the first one's. They are
search-level settings, and trajectories differ in evaluation only.

The same argument applies to declarative token families: ``CacheStoredTokens``
declares its tensors with the family, and they are evaluated into each
trajectory's subcache.
"""

import warnings

import numpy as np
import pytest

import epde
from epde import CacheStoredTokens, GridTokens
from epde.structure.domain import Trajectory


def _search(**kwargs):
    kwargs.setdefault('verbose_params', {'show_iter_idx': False})
    search = epde.EpdeSearch(**kwargs)
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    return search


def _sample(search, cache_id, n=120, domain_id=None, phase=0.0):
    grid = np.linspace(0, 4 * np.pi, n)
    domain = search.createDomain(
        grid, boundary_width=10,
        ID=cache_id if domain_id is None else domain_id)[1]
    data = np.sin(grid + phase) + 1.3 * np.cos(grid + phase)
    return domain, search.createTrajectory({'u': data}, domain,
                                           cache_id=cache_id)[1]


# ---------------------------------------------------------------------------
# where the parameters live
# ---------------------------------------------------------------------------

class TestOwnership:

    def test_create_trajectory_refuses_pool_settings(self):
        search = _search()
        domain, _ = _sample(search, 0)
        grid = np.linspace(0, 4 * np.pi, 120)
        with pytest.raises(TypeError, match='max_deriv_order'):
            search.createTrajectory({'u': np.sin(grid)}, domain, cache_id=1,
                                    max_deriv_order=(2,))

    def test_create_trajectory_refuses_the_powers_too(self):
        search = _search()
        domain, _ = _sample(search, 0)
        grid = np.linspace(0, 4 * np.pi, 120)
        for name in ('data_fun_pow', 'deriv_fun_pow'):
            with pytest.raises(TypeError, match=name):
                search.createTrajectory({'u': np.sin(grid)}, domain,
                                        cache_id=1, **{name: 2})

    def test_they_are_create_pool_parameters(self):
        import inspect
        from epde.interface.interface import EpdeSearch
        for method in ('create_pool', 'fit'):
            params = inspect.signature(getattr(EpdeSearch, method)).parameters
            for name in ('max_deriv_order', 'data_fun_pow', 'deriv_fun_pow'):
                assert name in params, '%s.%s' % (method, name)

    def test_a_fresh_trajectory_has_no_families_yet(self):
        """They follow from arguments it has not been given."""
        search = _search()
        _, trajectory = _sample(search, 0)
        assert trajectory.built is False
        assert trajectory.max_deriv_order is None
        with pytest.raises(RuntimeError, match='before build'):
            trajectory.tokens

    def test_variable_names_are_available_before_build(self):
        """Pool invalidation and the legacy shim read them at that point."""
        search = _search()
        _, trajectory = _sample(search, 0)
        assert trajectory.variable_names == ['u']


# ---------------------------------------------------------------------------
# they reach the pool
# ---------------------------------------------------------------------------

class TestReachThePool:

    def test_create_pool_builds_the_trajectory(self):
        search = _search()
        _, trajectory = _sample(search, 0)
        search.create_pool(data=[trajectory], max_deriv_order=(2,))
        assert trajectory.built is True
        assert trajectory.max_deriv_order == (2,)
        assert search.pool.families_cardinality()[1] > 0

    def _labels(self, **kwargs):
        # One search at a time: the trajectory manager is a process global that
        # each EpdeSearch construction resets, so two live searches share it.
        search = _search()
        _, trajectory = _sample(search, 0)
        search.create_pool(data=[trajectory], **kwargs)
        return {label for family in search.pool.families
                for label in family.tokens}, search

    def test_order_changes_the_token_labels(self):
        low, _ = self._labels(max_deriv_order=(1,))
        high, _ = self._labels(max_deriv_order=(3,))
        assert 'd^3u/dx0^3' in high
        assert 'd^3u/dx0^3' not in low

    def test_data_fun_pow_widens_the_variable_family(self):
        def power_range(**kwargs):
            _, search = self._labels(max_deriv_order=(1,), **kwargs)
            for family in search.pool.families:
                if 'u' in family.tokens:
                    return family.token_params['power']
            raise AssertionError('no variable family')

        assert power_range(data_fun_pow=3)[1] > power_range(data_fun_pow=1)[1]

    def test_defaults_come_from_the_config(self):
        search = _search(max_deriv_order=(2,), data_fun_pow=2)
        _, trajectory = _sample(search, 0)
        search.create_pool(data=[trajectory])
        assert trajectory.max_deriv_order == (2,)
        assert search.pool_params['data_fun_pow'] == 2


# ---------------------------------------------------------------------------
# one pool, several samples
# ---------------------------------------------------------------------------

class TestSharedAcrossTrajectories:

    def test_every_trajectory_gets_the_same_orders(self):
        search = _search()
        _, t1 = _sample(search, 0, phase=0.0)
        _, t2 = _sample(search, 1, phase=0.4)
        search.create_pool(data=[t1, t2], max_deriv_order=(2,), data_fun_pow=3)
        assert t1.max_deriv_order == t2.max_deriv_order == (2,)
        assert t1.built == t2.built is True

    def test_rebuild_is_a_noop_for_the_same_request(self):
        search = _search()
        _, trajectory = _sample(search, 0)
        search.create_pool(data=[trajectory], max_deriv_order=(2,))
        families = trajectory.families
        search.create_pool(data=[trajectory], max_deriv_order=(2,))
        assert trajectory.families is families

    def test_a_different_request_rebuilds(self):
        search = _search()
        _, trajectory = _sample(search, 0)
        search.create_pool(data=[trajectory], max_deriv_order=(1,))
        search.create_pool(data=[trajectory], max_deriv_order=(2,))
        assert trajectory.max_deriv_order == (2,)


# ---------------------------------------------------------------------------
# pool invalidation
# ---------------------------------------------------------------------------

class TestPoolInvalidation:
    """All three change which families exist, so all three must invalidate.

    The key used to read ``max_deriv_order`` off the trajectory and ignore the
    powers entirely, so a second ``fit`` at a different ``data_fun_pow`` reused
    a pool built for the first one.
    """

    def _key(self, **kwargs):
        search = _search()
        _, trajectory = _sample(search, 0)
        search.create_pool(data=[trajectory], **kwargs)
        return search.pool_params

    def test_order_is_part_of_the_key(self):
        assert self._key(max_deriv_order=(1,)) != self._key(max_deriv_order=(2,))

    def test_data_fun_pow_is_part_of_the_key(self):
        assert (self._key(max_deriv_order=(1,), data_fun_pow=1)
                != self._key(max_deriv_order=(1,), data_fun_pow=3))

    def test_deriv_fun_pow_is_part_of_the_key(self):
        assert (self._key(max_deriv_order=(1,), deriv_fun_pow=1)
                != self._key(max_deriv_order=(1,), deriv_fun_pow=2))

    def test_the_same_request_gives_the_same_key(self):
        assert self._key(max_deriv_order=(2,)) == self._key(max_deriv_order=(2,))


# ---------------------------------------------------------------------------
# declarative families stay with the pool
# ---------------------------------------------------------------------------

class TestDeclarativeFamilies:
    """A family is pool structure; its tensors are evaluated per trajectory.

    ``CacheStoredTokens`` therefore declares ``token_tensors`` with the family,
    as it always did, and ``create_pool`` uploads them into each trajectory's
    subcache -- masked by that trajectory's domain, the way the variable data
    itself is masked.
    """

    def _tokens(self, grid, labels=('tt',)):
        return CacheStoredTokens(
            token_type='stored', token_labels=list(labels),
            token_tensors={label: grid for label in labels},
            params_ranges={'power': (1, 1)}, params_equality_ranges=None,
            dimensionality=0, meaningful=True)

    def test_declared_tensors_reach_the_pool(self):
        search = _search()
        grid = np.linspace(0, 4 * np.pi, 120)
        _, trajectory = _sample(search, 0)
        family = self._tokens(grid)
        search.create_pool(data=[trajectory], additional_tokens=[family],
                           max_deriv_order=(2,))
        assert 'tt' in {label for fam in search.pool.families
                        for label in fam.tokens}

    def test_the_tensor_is_masked_like_the_data(self):
        search = _search()
        grid = np.linspace(0, 4 * np.pi, 120)
        domain, trajectory = _sample(search, 0)
        family = self._tokens(grid)
        search.create_pool(data=[trajectory], additional_tokens=[family],
                           max_deriv_order=(2,))
        stored = trajectory.get(('tt', (1.0,)))
        assert stored.shape == (int(np.prod(domain.inner_shape)),)
        assert np.allclose(stored, grid[domain.g_func_mask])

    def test_every_trajectory_gets_the_tensors(self):
        search = _search()
        grid = np.linspace(0, 4 * np.pi, 120)
        _, t1 = _sample(search, 0, phase=0.0)
        _, t2 = _sample(search, 1, phase=0.4)
        family = self._tokens(grid)
        search.create_pool(data=[t1, t2], additional_tokens=[family],
                           max_deriv_order=(2,))
        assert np.allclose(t1.get(('tt', (1.0,))), t2.get(('tt', (1.0,))))

    def test_label_mismatch_is_reported(self):
        _search()          # a search has to exist for the samples manager
        grid = np.linspace(0, 4 * np.pi, 120)
        with pytest.raises(KeyError, match='do not match'):
            CacheStoredTokens(token_type='stored', token_labels=['a', 'b'],
                              token_tensors={'a': grid},
                              params_ranges={'power': (1, 1)},
                              params_equality_ranges=None, dimensionality=0)

    def test_a_family_without_tensors_still_works(self):
        """GridTokens and friends generate their own values."""
        search = _search()
        _, trajectory = _sample(search, 0)
        family = GridTokens(['x_0'], dimensionality=0, max_power=2)
        search.create_pool(data=[trajectory], additional_tokens=[family],
                           max_deriv_order=(2,))
        assert 'x_0' in {label for fam in search.pool.families
                         for label in fam.tokens}
