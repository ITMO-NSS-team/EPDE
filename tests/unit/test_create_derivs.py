"""A derivative-only draw (``create_derivs=True``) takes its family among those that can offer a
derivative token.

``TFPool.create`` used to weigh the meaningful families by their plain cardinality and only then
ask the chosen family for a derivative token, so ``Term(pool, create_derivs=True)`` without a
``mandatory_family`` failed whenever it landed on a meaningful family without derivatives: a grid
family has no ``derivs_ords`` at all (AttributeError), and the variable's own family offers only
``[None,]`` (ValueError).
"""
from collections import OrderedDict

import numpy as np
import pytest

from epde.interface.token_family import TFPool, TokenFamily
from epde.structure.main_structures import Term

DERIVS = ['du/dx0', 'd^2u/dx0^2']


def _family(ftype, tokens, *, meaningful, derivs=None, variable=None):
    family = TokenFamily(token_type=ftype, variable=variable, family_of_derivs=derivs is not None)
    family.set_status(meaningful=meaningful, unique_specific_token=True, unique_token_type=False)
    family.set_params(tokens, OrderedDict([('power', (1, 1))]), {'power': 0},
                      derivs_solver_orders=derivs)
    family.set_evaluator(lambda factor, **kwargs: np.zeros(4))
    return family


def _pool(with_derivs=True):
    families = [_family('poly of u', ['u'], meaningful=True, derivs=[[None]], variable='u'),
                _family('grid', ['t'], meaningful=True),
                _family('trig', ['sin', 'cos'], meaningful=False)]
    if with_derivs:
        families.insert(0, _family('deriv of u', DERIVS, meaningful=True, derivs=[[0], [0, 0]],
                                   variable='u'))
    return TFPool(families)


def test_only_derivative_tokens_are_derivatives():
    pool = _pool()
    deriv, poly, grid, trig = pool.families
    assert [deriv.is_derivative_token(t) for t in DERIVS] == [True, True]
    assert not poly.is_derivative_token('u')
    assert not grid.is_derivative_token('t')
    assert [f.cardinality(derivs_only=True) for f in pool.families] == [2, 0, 0, 0]
    assert [f.cardinality() for f in pool.families] == [2, 1, 1, 2]


@pytest.mark.parametrize('meaningful', [True, False])
def test_a_derivative_draw_never_lands_on_a_family_without_derivatives(meaningful):
    pool = _pool()
    np.random.seed(0)
    labels = {pool.create(create_meaningful=meaningful, create_derivs=True)[1].label
              for _ in range(300)}
    assert labels == set(DERIVS)


def test_a_term_drawn_with_create_derivs_starts_with_a_derivative(monkeypatch):
    import epde.globals as global_var
    monkeypatch.setattr(global_var, 'tensor_cache', None, raising=False)    # no cache: Term skips use_cache
    pool = _pool()
    np.random.seed(1)
    for _ in range(200):
        term = Term(pool, max_factors_in_term=1, create_derivs=True)
        assert [factor.label for factor in term.structure][0] in DERIVS


def test_no_derivative_anywhere_is_a_value_error_not_an_attribute_error():
    pool = _pool(with_derivs=False)
    with pytest.raises(ValueError):
        pool.create(create_meaningful=True, create_derivs=True)
    with pytest.raises(ValueError):
        pool.create(create_meaningful=False, create_derivs=True)


@pytest.mark.parametrize('meaningful', [True, False])
def test_plain_draws_keep_their_probabilities_and_random_stream(meaningful):
    """create_derivs=False draws exactly as before the fix: the pre-fix body of TFPool.create,
    replayed on the same seed, gives the same factors."""
    pool = _pool()
    np.random.seed(2)
    drawn = [pool.create(create_meaningful=meaningful)[1].label for _ in range(50)]
    np.random.seed(2)
    families = pool.families_meaningful if meaningful else pool.families
    all_vars = [family.variable for family in pool.families_demand_equation]
    expected = []
    for _ in range(50):
        card = np.array([len([t for t in f.tokens]) for f in families])
        family = np.random.choice(a=families, p=card / np.sum(card))
        expected.append(family.create(label=None, token_status=None, create_derivs=False,
                                      all_vars=all_vars)[1].label)
    assert drawn == expected
