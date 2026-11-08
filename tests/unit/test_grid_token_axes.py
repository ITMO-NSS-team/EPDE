"""GridTokens with one label per axis: label k IS axis k.

The labels used to be names only: ``dim`` was drawn freely for every label, so on 2-D data
``x`` on axis 0 was the same column as ``t`` on axis 0. Identities such as ``x u_t = t u_t``
then fitted exactly (zero discrepancy) and took the first Pareto level of ac, burgers, kdv, ks,
wave and pde_compound in the common battery.
"""
import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.prepared_tokens import GridTokens


@pytest.fixture(autouse=True)
def _forget_registered_samples():
    """Leave no trajectory in the global samples manager: tests that resolve from it (the vcoef
    microscale) would read this module's fields."""
    yield
    global_var.samples_manager.reset()


def _dims(family, n=60, label=None):
    np.random.seed(0)
    draws = [family.create(label=label)[1] for _ in range(n)]
    return [(factor.label, int(factor.param('dim'))) for factor in draws]


def test_each_label_draws_its_own_axis():
    family = GridTokens(['t', 'x'], max_power=2, dimensionality=1).token_family
    draws = _dims(family)
    assert {label for label, _ in draws} == {'t', 'x'}
    assert all(dim == ['t', 'x'].index(label) for label, dim in draws)


def test_a_single_label_still_takes_any_axis():
    family = GridTokens(['x'], dimensionality=1).token_family
    assert {dim for _, dim in _dims(family)} == {0, 1}


def test_explicit_parameters_are_kept():
    family = GridTokens(['t', 'x'], dimensionality=1).token_family
    factor = family.create(label='x', power=1, dim=0)[1]
    assert int(factor.param('dim')) == 0


def test_the_tie_follows_an_auto_dimensionality():
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    T, X = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 40), indexing='ij')
    domain = search.createDomain([T, X], ID=0)[1]
    trajectory = search.createTrajectory({'u': np.sin(T + 2 * X)}, domain, cache_id=0)[1]
    grid = GridTokens(['t', 'x'])
    search.create_pool(data=[trajectory], max_deriv_order=(1, 1), additional_tokens=[grid])
    assert grid.token_family.token_params['dim'] == (0, 1)
    assert all(dim == ['t', 'x'].index(label) for label, dim in _dims(grid.token_family))


def test_bad_ties_are_refused():
    family = GridTokens(['t', 'x'], dimensionality=1).token_family
    with pytest.raises(ValueError):
        family.fix_token_params({'y': {'dim': 0}})
    with pytest.raises(ValueError):
        family.fix_token_params({'t': {'phase': 0}})
