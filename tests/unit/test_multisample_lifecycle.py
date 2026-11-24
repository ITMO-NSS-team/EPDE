"""Check that cached rows and full-grid values stay in sync across fits."""
import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.structure.domain import VariableEntry
from epde.structure.weak_rows import build_weak_rows


@pytest.fixture
def search():
    obj = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False})
    obj.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    try:
        yield obj
    finally:
        obj.close()


def _weak_sources(search):
    return build_weak_rows(search.pool, global_var.samples_manager,
                           lambda shape: True, window_gate='off')


@pytest.mark.parametrize('entries_form', ['mapping', 'entries'])
@pytest.mark.parametrize('rebuild', ['first', 'reuse', 'force', 'release'])
def test_observations_remain_the_creation_snapshot(search, entries_form, rebuild):
    grid = np.linspace(0, 2, 40)
    observed = np.sin(grid)
    expected = observed.copy()
    domain = search.createDomain(grid, ID=7, boundary_width=0)[1]
    if entries_form == 'entries':
        entry = VariableEntry('u', 0, observed)
        entries, caller_owned = [entry], entry.raw_tensor
    else:
        entries, caller_owned = {'u': observed}, observed
    sample = search.createTrajectory(entries, domain, cache_id=7,
                                     derivs=np.cos(grid)[:, None])[1]
    if rebuild != 'first':
        search.create_pool([sample], max_deriv_order=(1,))

    caller_owned += 1.0
    if rebuild == 'force':
        sample.build(max_deriv_order=(1,), force=True)
    elif rebuild == 'release':
        global_var.release_tensor_cache()
    search.create_pool([sample], max_deriv_order=(1,))

    np.testing.assert_array_equal(sample.raw_field('u'), expected)
    np.testing.assert_array_equal(sample.get(('u', (1.0,))), expected)
    np.testing.assert_array_equal(_weak_sources(search).fields[7].values['u'], expected)


def test_nonfinite_edit_to_the_original_field_does_not_corrupt_the_sample(search):
    grid = np.linspace(0, 2, 40)
    field = np.sin(grid)
    expected = field.copy()
    domain = search.createDomain(grid, ID=7, boundary_width=0)[1]
    sample = search.createTrajectory({'u': field}, domain, cache_id=7,
                                     derivs=np.cos(grid)[:, None])[1]
    field[:] = np.nan

    search.create_pool([sample], max_deriv_order=(1,))

    np.testing.assert_array_equal(sample.raw_field('u'), expected)
    np.testing.assert_array_equal(sample.get(('u', (1.0,))), expected)


@pytest.mark.parametrize('form', ['array', 'mapping', 'list', 'tuple'])
@pytest.mark.parametrize('rebuild', ['first', 'force'])
def test_precomputed_derivatives_are_snapshotted_with_the_observations(search, form, rebuild):
    grid = np.linspace(0, 2, 40)
    domain = search.createDomain(grid, ID=7, boundary_width=0)[1]
    du, dv = np.cos(grid)[:, None], -np.sin(grid)[:, None]
    expected = {'du/dx0': du[:, 0].copy(), 'dv/dx0': dv[:, 0].copy()}
    fields = {'u': np.sin(grid), 'v': np.cos(grid)}
    if form == 'array':
        derivatives, fields = du, {'u': fields['u']}
        expected.pop('dv/dx0')
    elif form == 'mapping':
        derivatives = {'u': du, 'v': dv}
    elif form == 'list':
        derivatives = [du, dv]
    else:
        derivatives = (du, dv)
    sample = search.createTrajectory(fields, domain, cache_id=7, derivs=derivatives)[1]
    if rebuild == 'force':
        search.create_pool([sample], max_deriv_order=(1,))

    du += 7.0
    dv -= 9.0
    if form == 'mapping':
        derivatives['u'] = np.full_like(du, 12.0)
    elif form == 'list':
        derivatives[0] = np.full_like(du, 12.0)
    if rebuild == 'force':
        sample.build(max_deriv_order=(1,), force=True)
    search.create_pool([sample], max_deriv_order=(1,))

    for label, values in expected.items():
        np.testing.assert_array_equal(sample.get((label, (1.0,))), values)


def test_original_coordinate_edits_do_not_change_the_domain_or_derivatives(search):
    grid = np.linspace(0, 2, 60)
    expected_grid = grid.copy()
    field = 2 * grid ** 2
    domain = search.createDomain(grid, ID=7, boundary_width=2)[1]
    sample = search.createTrajectory({'u': field}, domain, cache_id=7)[1]

    grid *= 3.0
    search.create_pool([sample], max_deriv_order=(1,))

    np.testing.assert_array_equal(sample.grids()[0], expected_grid)
    np.testing.assert_array_equal(sample.grids('solver')[0], expected_grid[2:-2])
    np.testing.assert_allclose(sample.get(('du/dx0', (1.0,))),
                               4 * expected_grid[2:-2], atol=1e-10)


@pytest.mark.parametrize('callable_weights', [False, True])
def test_original_weight_edits_do_not_change_a_validated_domain(search, callable_weights):
    grid = np.linspace(0, 2, 40)
    weights = np.r_[np.zeros(2), np.linspace(1, 2, 36), np.zeros(2)]
    expected = weights.copy()
    declared = (lambda grids: weights) if callable_weights else weights
    domain = search.createDomain(grid, ID=7, boundary_width=2, gfunction=declared)[1]

    weights[:] = np.nan

    np.testing.assert_array_equal(domain.g_func, expected)
    np.testing.assert_array_equal(domain.g_func_masked_flat, expected[2:-2])


def _stored(values=None):
    return epde.CacheStoredTokens(token_type='stored', token_labels=['g'],
                                  token_tensors=None if values is None else {'g': values},
                                  params_ranges={'power': (1, 1)},
                                  params_equality_ranges=None, meaningful=True)


@pytest.mark.parametrize('source', ['family', 'trajectory'])
@pytest.mark.parametrize('rebuild', ['reuse', 'force', 'release'])
def test_stored_token_edits_keep_raw_and_cached_sources_consistent(search, source, rebuild):
    grid = np.linspace(0, 2, 40)
    token = 2 + np.cos(grid)
    original = token.copy()
    domain = search.createDomain(grid, ID=7, boundary_width=2)[1]
    family = _stored(token if source == 'family' else None)
    per_sample = {family: {'g': token}} if source == 'trajectory' else None
    sample = search.createTrajectory({'u': np.sin(grid)}, domain, cache_id=7,
                                     derivs=np.cos(grid)[:, None],
                                     cached_token_tensors=per_sample)[1]
    search.create_pool([sample], max_deriv_order=(1,), additional_tokens=[family])

    token += 5.0
    np.testing.assert_array_equal(sample.raw_token('g'), original)
    np.testing.assert_array_equal(sample.get(('g', (1.0,))), original[2:-2])

    if rebuild == 'force':
        sample.build(max_deriv_order=(1,), force=True)
    elif rebuild == 'release':
        global_var.release_tensor_cache()
    search.create_pool([sample], max_deriv_order=(1,), additional_tokens=[family])

    # Family declarations are uploaded afresh on a real rebuild. Per-sample
    # tensors, like its observations, belong to the trajectory's creation.
    expected = token if source == 'family' and rebuild != 'reuse' else original
    np.testing.assert_array_equal(sample.raw_token('g'), expected)
    np.testing.assert_array_equal(sample.get(('g', (1.0,))), expected[2:-2])
    np.testing.assert_array_equal(_weak_sources(search).tokens[7]['g'], expected)


def test_replacing_a_stored_family_replaces_both_raw_and_masked_values(search):
    grid = np.linspace(0, 2, 40)
    domain = search.createDomain(grid, ID=7, boundary_width=2)[1]
    sample = search.createTrajectory({'u': np.sin(grid)}, domain, cache_id=7,
                                     derivs=np.cos(grid)[:, None])[1]
    search.create_pool([sample], max_deriv_order=(1,), additional_tokens=[_stored(np.cos(grid))])
    replacement = 10 * np.cos(grid)

    search.create_pool([sample], max_deriv_order=(1,), additional_tokens=[_stored(replacement)])

    np.testing.assert_array_equal(sample.raw_token('g'), replacement)
    np.testing.assert_array_equal(sample.get(('g', (1.0,))), replacement[2:-2])


def test_invalid_stored_upload_does_not_poison_a_later_valid_upload(search):
    grid = np.linspace(0, 2, 40)
    domain = search.createDomain(grid, ID=7, boundary_width=2)[1]
    sample = search.createTrajectory({'u': np.sin(grid)}, domain, cache_id=7,
                                     derivs=np.cos(grid)[:, None])[1]
    search.create_pool([sample], max_deriv_order=(1,))
    family = _stored()

    with pytest.raises((IndexError, ValueError)):
        sample.uploadTokenTensors(family, {'g': np.ones(3)})
    with pytest.raises(KeyError):
        sample.raw_token('g')

    expected = 2 + np.cos(grid)
    sample.uploadTokenTensors(family, {'g': expected})

    np.testing.assert_array_equal(sample.raw_token('g'), expected)
    np.testing.assert_array_equal(sample.get(('g', (1.0,))), expected[2:-2])


@pytest.mark.parametrize('route', ['build', 'create_pool'])
def test_mutating_derivative_order_list_is_a_new_build_request(search, route):
    grid = np.linspace(0, 2, 60)
    domain = search.createDomain(grid, ID=7, boundary_width=4)[1]
    sample = search.createTrajectory({'u': 2 * grid ** 2}, domain, cache_id=7)[1]
    orders = [1]

    def prepare():
        if route == 'build':
            sample.build(max_deriv_order=orders)
        else:
            search.create_pool([sample], max_deriv_order=orders)

    prepare()
    orders[0] = 2
    prepare()

    np.testing.assert_allclose(sample.get(('d^2u/dx0^2', (1.0,))), 4.0, atol=1e-9)
