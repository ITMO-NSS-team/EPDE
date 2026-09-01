"""The boundary width is applied exactly once, and the solver grids agree.

``createDomain`` feeds one ``boundary_width`` to two places: ``setBoundaries``,
which records ``inner_shape``, and ``BoundaryExclusion``, which builds the test
function whose non-zero mask actually selects the data. That looks like double
application but is not -- ``setBoundaries`` excludes nothing, it only
describes. The two are one concept and must agree, which is what these tests
pin: if they ever diverge, ``inner_shape`` starts misdescribing the masked
data and nothing else notices.

``getGrids(mode='solver')`` is the third place the width is used, and it was
dead: ``setBoundaries`` assigns ``boundary_width`` while it read
``_boundary_width``, so every call raised AttributeError -- and iterating that
value would still have failed for a scalar width.
"""

import numpy as np
import pytest

import epde


@pytest.fixture(scope='module')
def search():
    return epde.EpdeSearch(verbose_params={'show_iter_idx': False})


def _domain_1d(search, boundary_width, domain_id, n=200):
    grid = np.linspace(0, 4 * np.pi, n)
    return search.createDomain(grid, boundary_width=boundary_width,
                               ID=domain_id)[1]


def _domain_2d(search, boundary_width, domain_id):
    grids = np.meshgrid(np.linspace(0, 1, 51), np.linspace(-1, 1, 128),
                        indexing='ij')
    return search.createDomain(grids, boundary_width=boundary_width,
                               ID=domain_id)[1]


class TestSingleApplication:

    def test_scalar_width_1d(self, search):
        domain = _domain_1d(search, 10, 10)
        assert list(domain.inner_shape) == [180]        # 200 - 2*10
        assert int(domain.g_func_mask.sum()) == 180

    def test_mask_matches_inner_shape_1d(self, search):
        domain = _domain_1d(search, 7, 11)
        assert int(domain.g_func_mask.sum()) == int(np.prod(domain.inner_shape))

    def test_mask_matches_inner_shape_2d_scalar(self, search):
        domain = _domain_2d(search, 5, 12)
        assert list(domain.inner_shape) == [41, 118]
        assert int(domain.g_func_mask.sum()) == int(np.prod(domain.inner_shape))

    def test_mask_matches_inner_shape_2d_per_axis(self, search):
        domain = _domain_2d(search, (5, 12), 13)
        assert list(domain.inner_shape) == [41, 104]
        assert int(domain.g_func_mask.sum()) == int(np.prod(domain.inner_shape))

    def test_zero_width_keeps_everything(self, search):
        domain = _domain_1d(search, 0, 14)
        assert list(domain.inner_shape) == [200]
        assert int(domain.g_func_mask.sum()) == 200


class TestSolverGrids:

    def test_solver_mode_used_to_raise(self, search):
        """Regression: _boundary_width vs boundary_width."""
        domain = _domain_1d(search, 10, 20)
        grids = domain.getGrids(mode='solver')          # must not raise
        assert [g.shape for g in grids] == [(180,)]

    def test_solver_grids_match_inner_shape_scalar(self, search):
        domain = _domain_2d(search, 5, 21)
        expected = tuple(int(n) for n in domain.inner_shape)
        assert all(g.shape == expected for g in domain.getGrids(mode='solver'))

    def test_solver_grids_match_inner_shape_per_axis(self, search):
        domain = _domain_2d(search, (5, 12), 22)
        expected = tuple(int(n) for n in domain.inner_shape)
        assert all(g.shape == expected for g in domain.getGrids(mode='solver'))

    def test_full_mode_is_untrimmed(self, search):
        domain = _domain_1d(search, 10, 23)
        assert [g.shape for g in domain.getGrids(mode='full')] == [(200,)]

    def test_per_axis_width_is_recorded(self, search):
        domain = _domain_2d(search, (5, 12), 24)
        assert domain.boundary_width_per_axis == [5, 12]
        assert domain.boundary_width == (5, 12)

    def test_scalar_width_is_broadcast_per_axis(self, search):
        domain = _domain_2d(search, 5, 25)
        assert domain.boundary_width_per_axis == [5, 5]


class TestValidation:

    def test_width_too_large_for_the_grid(self, search):
        with pytest.raises(IndexError, match='does not fit'):
            _domain_1d(search, 150, 30)

    def test_bad_width_type(self, search):
        with pytest.raises(TypeError, match='Incorrect type of boundaries'):
            _domain_1d(search, 'wide', 31)


class TestConfigDefault:

    def test_width_comes_from_the_config_when_omitted(self):
        search = epde.EpdeSearch(boundary_width=8,
                                 verbose_params={'show_iter_idx': False})
        domain = _domain_1d(search, epde.interface.search_config.UNSET, 40)
        assert list(domain.inner_shape) == [184]        # 200 - 2*8

    def test_explicit_width_overrides_the_config(self):
        search = epde.EpdeSearch(boundary_width=8,
                                 verbose_params={'show_iter_idx': False})
        domain = _domain_1d(search, 3, 41)
        assert list(domain.inner_shape) == [194]


class TestMaskShapes:
    """Derivatives come back flat, the data tensor keeps the grid shape.

    ``addEntryToCache`` needs both forms of the same mask. Indexing a flat
    derivative column with the grid-shaped mask happens to work in 1-D, where
    the two coincide, and raises ``IndexError: too many indices`` for every 2-D
    system -- which is most of the pic/data corpus.
    """

    def test_flat_mask_matches_the_grid_mask(self, search):
        domain = _domain_2d(search, (5, 12), 50)
        assert domain.g_func_mask_flat.shape == (domain.g_func_mask.size,)
        assert int(domain.g_func_mask_flat.sum()) == int(domain.g_func_mask.sum())

    def test_flat_mask_is_a_noop_in_1d(self, search):
        domain = _domain_1d(search, 10, 51)
        assert domain.g_func_mask_flat.shape == domain.g_func_mask.shape

    def test_two_dimensional_data_reaches_the_pool(self):
        """End-to-end regression: this raised IndexError before the fix."""
        import warnings
        search = epde.EpdeSearch(verbose_params={'show_iter_idx': False})
        search.set_preprocessor(default_preprocessor_type='FD',
                                preprocessor_kwargs={})
        grids = np.meshgrid(np.linspace(0., 1., 51),
                            np.linspace(-1., 1., 64), indexing='ij')
        data = np.sin(np.pi * grids[1]) * np.exp(-grids[0])
        domain = search.createDomain(grids, boundary_width=(5, 12), ID=60)[1]
        trajectory = search.createTrajectory({'u': data}, domain, cache_id=60)[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            search.create_pool(data=[trajectory], max_deriv_order=(2, 2))
        assert search.pool.families_cardinality()[1] > 0
