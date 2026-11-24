"""Check per-trajectory grids, row ordering, preprocessing and caches."""
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.evaluators import CustomEvaluator, simple_function_evaluator
from epde.preprocessing.preprocessor import PreprocessingPipe


def _search(**kwargs):
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, **kwargs)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _one_dimensional(search, key, grid, values=None, **kwargs):
    domain = search.createDomain(grid, ID=key, boundary_width=2)[1]
    values = np.sin(grid) if values is None else values
    return search.createTrajectory({'u': values}, domain, cache_id=key, **kwargs)[1]


def _pipe(smoother=None, calculator=None):
    pipe = PreprocessingPipe()
    pipe.smoother = smoother
    pipe.smoother_args = ()
    pipe.deriv_calculator = calculator or (
        lambda data, grid, max_order: np.repeat(
            np.cos(grid[0]).reshape(-1, 1), sum(max_order), axis=1))
    pipe.deriv_calculator_args = ()
    pipe.deriv_calculator_kwargs = {'grid': None}
    return pipe


class TestSampleRowGeometry:

    @pytest.mark.parametrize('stride', [None, (2, 3)])
    def test_two_samples_can_share_a_multidimensional_domain(self, stride):
        search = _search(evaluation_stride=stride)
        grids = np.meshgrid(np.linspace(0, 1, 21), np.linspace(-2, 2, 31), indexing='ij')
        domain = search.createDomain(grids, ID=4, boundary_width=2)[1]
        samples = [search.createTrajectory(
            {'u': np.sin(grids[0] + phase) + np.cos(grids[1])}, domain, cache_id=key)[1]
            for key, phase in [(7, 0.0), (19, 0.4)]]

        search.create_pool(samples, max_deriv_order=(1, 1))

        expected_rows = 17 * 27 if stride is None else 9 * 9
        assert global_var.samples_manager.trajecatoryIDs == [7, 19]
        for sample in samples:
            assert sample.get(('u', (1.0,))).shape == (expected_rows,)
        assert not np.allclose(samples[0].get(('u', (1.0,))),
                               samples[1].get(('u', (1.0,))))

    def test_a_nonzero_time_axis_preserves_flat_cache_row_order(self):
        search = _search()
        x, t = np.meshgrid(np.linspace(-1, 1, 17), np.linspace(0, 2, 23), indexing='ij')
        data = x ** 2 + 3 * t ** 3
        domain = search.createDomain([x, t], ID=8, boundary_width=(2, 3), time_axis=1)[1]
        derivatives = np.column_stack([(2 * x).ravel(), (9 * t ** 2).ravel()])
        sample = search.createTrajectory({'u': data}, domain, cache_id=8,
                                         derivs=derivatives)[1]

        sample.build(max_deriv_order=(1, 1))

        np.testing.assert_array_equal(sample.get(('u', (1.0,))), data[2:-2, 3:-3].ravel())
        np.testing.assert_array_equal(sample.get(('du/dx0', (1.0,))),
                                      (2 * x)[2:-2, 3:-3].ravel())
        np.testing.assert_array_equal(sample.get(('du/dx1', (1.0,))),
                                      (9 * t ** 2)[2:-2, 3:-3].ravel())

    def test_unequal_samples_are_differentiated_independently(self):
        search = _search()
        first_grid, second_grid = np.linspace(0, 2, 51), np.linspace(10, 11, 67)
        first = _one_dimensional(search, 7, first_grid, 2 * first_grid ** 2)
        second = _one_dimensional(search, 19, second_grid, 1000 - 3 * second_grid ** 2)

        search.create_pool([first, second], max_deriv_order=(1,))

        assert first.get(('u', (1.0,))).shape == (47,)
        assert second.get(('u', (1.0,))).shape == (63,)
        np.testing.assert_allclose(first.get(('du/dx0', (1.0,))),
                                   4 * first_grid[2:-2], atol=1e-10)
        np.testing.assert_allclose(second.get(('du/dx0', (1.0,))),
                                   -6 * second_grid[2:-2], atol=1e-9)


class TestPreprocessingRebuild:

    @pytest.mark.parametrize('rebuild', ['new_order', 'release', 'force'])
    def test_a_rebuild_starts_from_raw_observations(self, rebuild):
        search = _search()
        grid = np.linspace(0, 2, 60)
        raw = np.sin(grid)
        sample = _one_dimensional(search, 7, grid, raw, preprocessor=_pipe(lambda data: data + 1))
        sample.build(max_deriv_order=(1,))

        if rebuild == 'new_order':
            sample.build(max_deriv_order=(2,))
        elif rebuild == 'release':
            global_var.release_tensor_cache()
            sample.build(max_deriv_order=(1,))
        else:
            sample.build(max_deriv_order=(1,), force=True)

        np.testing.assert_allclose(sample.get(('u', (1.0,))), raw[2:-2] + 1)
        np.testing.assert_array_equal(sample.raw_field('u'), raw)

    def test_an_inplace_smoother_cannot_mutate_observations(self):
        def inplace_smoother(data):
            data += 2
            return data

        search = _search()
        grid = np.linspace(0, 2, 60)
        raw = np.sin(grid)
        observed = raw.copy()
        sample = _one_dimensional(search, 7, grid, raw,
                                  preprocessor=_pipe(inplace_smoother))
        sample.build(max_deriv_order=(1,))

        np.testing.assert_array_equal(raw, observed)
        np.testing.assert_array_equal(sample.raw_field('u'), observed)
        np.testing.assert_allclose(sample.get(('u', (1.0,))), observed[2:-2] + 2)

    @pytest.mark.parametrize('derivatives', [np.ones((60, 2)), np.ones((59, 1)),
                                             np.ones(60), np.full((60, 1), np.nan)])
    def test_bad_calculator_output_fails_before_cache_upload(self, derivatives):
        search = _search()
        pipe = _pipe(calculator=lambda data, **kwargs: derivatives)
        sample = _one_dimensional(search, 7, np.linspace(0, 2, 60), preprocessor=pipe)

        with pytest.raises(ValueError, match='derivatives.*u|u.*derivatives'):
            sample.build(max_deriv_order=(1,))

        assert not sample.tensors_cached

    def test_a_smoother_must_preserve_the_field_shape(self):
        search = _search()
        sample = _one_dimensional(search, 7, np.linspace(0, 2, 60),
                                  preprocessor=_pipe(lambda data: data[:-1]))

        with pytest.raises(ValueError, match='shape.*u|u.*shape'):
            sample.build(max_deriv_order=(1,))

    def test_partial_precomputed_derivatives_still_check_forward_boundaries(self):
        search = _search()
        search.set_preprocessor(default_preprocessor_type='FD',
                                preprocessor_kwargs={'scheme': 'forward', 'lag': 2})
        grid = np.linspace(0, 2, 60)
        domain = search.createDomain(grid, ID=7, boundary_width=0)[1]
        sample = search.createTrajectory(
            {'u': np.sin(grid), 'v': np.cos(grid)}, domain, cache_id=7,
            derivs={'u': np.cos(grid).reshape(-1, 1), 'v': None})[1]

        with pytest.raises(ValueError, match='undefined.*boundary'):
            sample.build(max_deriv_order=(1,))


class TestSampleCacheMembership:

    @pytest.mark.parametrize('normalized', [False, True])
    def test_membership_uses_only_current_samples(self, normalized):
        search = _search()
        first = _one_dimensional(search, 7, np.linspace(0, 2, 40))
        second = _one_dimensional(search, 19, np.linspace(0, 3, 50))
        first.build(max_deriv_order=(1,))
        second.build(max_deriv_order=(1,))
        search._register_trajectories([first, second])
        label = ('sample_specific_token', (1.0,))
        values = {7: np.arange(36.), 19: np.arange(46.)}

        assert global_var.samples_manager.add(label, values, normalized=normalized)

        assert (label, normalized) in global_var.samples_manager
        assert global_var.samples_manager.get(label, normalized=normalized)[19].shape == (46,)
        first._cache.memory_normalized[19].pop(label, None)
        first._cache.memory_default[19].pop(label, None)
        assert (label, normalized) not in global_var.samples_manager
        search._register_trajectories([first])
        assert (label, normalized) in global_var.samples_manager

    def test_reusing_a_cache_id_drops_old_variable_derivative_labels(self):
        search = _search()
        grid = np.linspace(0, 2, 40)
        first = _one_dimensional(search, 7, grid)
        first.build(max_deriv_order=(1,))
        second = search.createTrajectory({'v': np.cos(grid)}, first._domain, cache_id=7,
                                         derivs=-np.sin(grid).reshape(-1, 1))[1]

        second.build(max_deriv_order=(1,))

        np.testing.assert_allclose(second.get(None, deriv_code=(0, [0])), -np.sin(grid)[2:-2])
        np.testing.assert_allclose(second.get(None, deriv_code=(0, [None])), np.cos(grid)[2:-2])


class TestDomainWeights:

    @pytest.mark.parametrize('stride', [None, (2, 2)])
    @pytest.mark.parametrize('support', ['shifted', 'hole'])
    def test_equal_row_count_does_not_make_an_arbitrary_mask_rectangular(self, stride, support):
        search = _search(evaluation_stride=stride)
        try:
            x, t = np.meshgrid(np.arange(15.), np.arange(19.), indexing='ij')
            weights = np.zeros_like(x)
            if support == 'shifted':
                weights[:11, :13] = 1.0
            else:
                weights[2:-2, 3:-3] = 1.0
                weights[5, 5], weights[0, 0] = 0.0, 1.0
            assert np.count_nonzero(weights) == 11 * 13

            with pytest.raises(ValueError, match='interior box'):
                search.createDomain([x, t], ID=7, boundary_width=(2, 3), gfunction=weights)
        finally:
            search.close()

    def test_custom_rectangular_weights_match_each_records_solver_coordinates(self):
        search = _search()
        try:
            x, t = np.meshgrid(np.arange(17.), np.arange(23.), indexing='ij')
            samples = []
            for key, boundary, amplitude in [(7, (1, 2), 1.), (19, (2, 3), 3.)]:
                weights = np.zeros_like(x)
                bx, bt = boundary
                weights[bx:-bx, bt:-bt] = (1 + x / 20 + t / 30)[bx:-bx, bt:-bt]
                domain = search.createDomain([x, t], ID=key, boundary_width=boundary,
                                             gfunction=weights)[1]
                field = amplitude * (100 * x + t)
                derivatives = np.column_stack([np.full(x.size, 100 * amplitude),
                                               np.full(x.size, amplitude)])
                samples.append(search.createTrajectory({'u': field}, domain, cache_id=key,
                                                        derivs=derivatives)[1])
            search.create_pool(samples, max_deriv_order=(1, 1))

            for sample, amplitude, expected_shape in zip(samples, [1., 3.], [(15, 19), (13, 17)]):
                solver_x, solver_t = sample.grids('solver')
                assert solver_x.shape == solver_t.shape == expected_shape
                np.testing.assert_array_equal(sample.get(('u', (1.0,))),
                                              (amplitude * (100 * solver_x + solver_t)).ravel())
                np.testing.assert_array_equal(sample.gFuncs('dmf'),
                                              (1 + solver_x / 20 + solver_t / 30).ravel())
        finally:
            search.close()

    @pytest.mark.parametrize('weights', [np.ones((4, 5)), np.full(20, np.nan),
                                          np.full(20, np.inf), -np.ones(20),
                                          np.ones(20, dtype=complex) * (1 + 1j)])
    def test_weights_must_match_the_grid_and_be_finite_nonnegative_reals(self, weights):
        search = _search()

        with pytest.raises(ValueError, match='test function|Test function'):
            search.createDomain(np.linspace(0, 1, 20), ID=7, boundary_width=0,
                                 gfunction=weights)

    def test_list_weights_have_the_same_rows_as_array_weights(self):
        search = _search()
        domain = search.createDomain(np.linspace(0, 1, 20), ID=7, boundary_width=2,
                                     gfunction=[0., 0.] + [2.] * 16 + [0., 0.])[1]

        np.testing.assert_array_equal(domain.g_func_masked_flat, np.full(16, 2.))


class TestCustomSampleEvaluators:

    @staticmethod
    def factor():
        return SimpleNamespace(label='f', params_description={})

    def test_reusing_a_sample_id_reads_its_new_grid(self):
        evaluator = CustomEvaluator(lambda x: x * x, [])
        first_search = _search()
        first = _one_dimensional(first_search, 7, np.linspace(0, 2, 30))
        first.build(max_deriv_order=(1,))
        first_search._register_trajectories([first])
        evaluator.full_grid(self.factor())

        second_search = _search()
        grid = np.linspace(10, 12, 40)
        second = _one_dimensional(second_search, 7, grid)
        second.build(max_deriv_order=(1,))
        second_search._register_trajectories([second])

        np.testing.assert_allclose(evaluator.full_grid(self.factor())[7], grid ** 2)

    def test_explicit_grids_can_use_a_subset_of_sample_ids(self):
        search = _search()
        first = _one_dimensional(search, 7, np.linspace(0, 2, 30))
        second = _one_dimensional(search, 19, np.linspace(0, 3, 40))
        for sample in [first, second]:
            sample.build(max_deriv_order=(1,))
        search._register_trajectories([first, second])
        evaluator = CustomEvaluator(lambda x: x + 5, [])
        grid = np.array([2., 4., 6.])

        values = evaluator.full_grid(self.factor(), {19: [grid]})

        assert list(values) == [19]
        np.testing.assert_array_equal(values[19], [7., 9., 11.])

    def test_explicit_grid_changes_do_not_reuse_saved_coordinates(self):
        search = _search()
        sample = _one_dimensional(search, 7, np.linspace(0, 2, 30))
        sample.build(max_deriv_order=(1,))
        search._register_trajectories([sample])
        evaluator = CustomEvaluator(lambda x: x + 5, [])
        grid = np.array([2., 4., 6.])
        evaluator.full_grid(self.factor(), {7: [grid]})
        grid += 10

        np.testing.assert_array_equal(evaluator.full_grid(self.factor(), {7: [grid]})[7],
                                      [17., 19., 21.])

    def test_data_evaluator_preserves_nonzero_explicit_sample_keys(self):
        factor = SimpleNamespace(
            params_description={0: {'name': 'power'}}, params=np.array([2.]),
            predict_with_ann=lambda coords: coords[0] + 1)

        values = simple_function_evaluator(factor, grids={19: [np.array([2., 4.])]})

        assert list(values) == [19]
        np.testing.assert_array_equal(values[19], [9., 25.])


class TestSampleTokenConstancy:
    """A fixed experiment parameter can vary across independent records."""

    @staticmethod
    def family():
        return epde.CacheStoredTokens(
            token_type='source', token_labels=['p'],
            params_ranges={'power': (1, 1)}, dimensionality=0,
            meaningful=True)

    @staticmethod
    def record(search, family, key, p):
        t = np.linspace(0, 4, 80)
        domain = search.createDomain(t, ID=key, boundary_width=0)[1]
        values = np.broadcast_to(p, t.shape).copy()
        return search.createTrajectory(
            {'u': values * (1 - np.exp(-t))}, domain, cache_id=key,
            derivs=(values * np.exp(-t))[:, None],
            cached_token_tensors={family: {'p': values}})[1]

    @pytest.mark.parametrize('ids', [(0, 1), (7, 19)])
    @pytest.mark.parametrize('reverse', [False, True])
    def test_record_specific_constants_remain_available(self, ids, reverse):
        search = _search(system_rps_mode='plain', multisample_mode='joint')
        family = self.family()
        samples = [self.record(search, family, key, p)
                   for key, p in zip(ids, [1., 2.])]

        search.create_pool(samples[::-1] if reverse else samples,
                           additional_tokens=[family], max_deriv_order=(1,))

        assert 'p' in family.token_family.tokens
        for sample, p in zip(samples, [1., 2.]):
            np.testing.assert_array_equal(sample.get(('p', (1.0,))), np.full(80, p))

    @pytest.mark.parametrize('ids', [(0, 1), (7, 19)])
    def test_inactive_values_neither_block_pruning_nor_get_deleted(self, ids):
        search = _search()
        family = self.family()
        inactive = self.record(search, family, 55, np.linspace(1., 2., 80))
        inactive.build(max_deriv_order=(1,))
        before = inactive.get(('p', (1.0,))).copy()
        samples = [self.record(search, family, key, 3.) for key in ids]

        search.create_pool(samples, additional_tokens=[family], max_deriv_order=(1,))

        assert 'p' not in next(f.tokens for f in search.pool.families if f.ftype == 'source')
        assert family.token_family.tokens == ['p']
        for sample in samples:
            np.testing.assert_array_equal(sample.get(('p', (1.0,))), np.full(80, 3.))
        np.testing.assert_array_equal(inactive.get(('p', (1.0,))), before)

    def test_changing_the_active_set_restores_declared_tokens(self):
        search = _search()
        family = self.family()
        first = self.record(search, family, 0, 1.)
        second = self.record(search, family, 1, 2.)

        for samples, retained in [([first], False), ([first, second], True), ([first], False)]:
            search.create_pool(samples, additional_tokens=[family], max_deriv_order=(1,))
            active_labels = next(f.tokens for f in search.pool.families if f.ftype == 'source')
            assert ('p' in active_labels) is retained
            assert family.token_family.tokens == ['p']
            np.testing.assert_array_equal(first.get(('p', (1.0,))), np.ones(80))

    def test_invalid_cached_values_are_not_pruned_as_a_constant(self):
        search = _search()
        family = self.family()
        samples = [self.record(search, family, key, p) for key, p in [(0, 1.), (1, 2.)]]
        search.create_pool(samples, additional_tokens=[family], max_deriv_order=(1,))
        samples[1].get(('p', (1.0,)))[:] = np.nan

        search.create_pool(samples, additional_tokens=[family], max_deriv_order=(1,))

        assert 'p' in next(f.tokens for f in search.pool.families if f.ftype == 'source')
