"""``search_space.system_rps_params['invariant_terms']`` of plain mode: terms an exact algebraic invariant of the
variables makes redundant (``epde.operators.common.invariant_terms``).

Pinned here:

* the configuration: ``'keep'`` by default, ``'refuse'`` accepted, anything else refused;
* ``detect``: the unit circle found with exact fractions, nothing on unrelated data;
* the rule on a heading-vector pool (c = cos phi, s = sin phi, derivatives to order 2): it refuses the identity
  forms the robot spin fronts filled with and keeps the law's terms and the terms that are no identity;
* the standard monomials are linearly independent on an exact jet of a non-uniform rotation and span every
  monomial the rule refuses -- no identity can be written, nothing is lost;
* ``refused``: all False under ``'keep'`` without reading the data;
* end to end: no fitted equation of a seeded plain NSGA-II fit holds a refused term, and a seeded fit with
  ``invariant_terms='keep'`` equals the same fit without the key.
"""
import warnings
from contextlib import contextmanager
from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import (SYSTEM_RPS_PARAMS, active_config, check_plain_params,
                                          collect_overrides, load_search_config, resolve_plain_params,
                                          set_active_config)
from epde.operators.common import invariant_terms as IT
from epde.operators.common import plain_fit as PF

LIVE_FACTORS = {'factors_num': [1, 2, 3], 'probas': [0.5, 0.35, 0.15]}
C, S = 'c{power: 1.0}', 's{power: 1.0}'
DC, DS = 'dc/dx0{power: 1.0}', 'ds/dx0{power: 1.0}'
DDC, DDS = 'd^2c/dx0^2{power: 1.0}', 'd^2s/dx0^2{power: 1.0}'


def _phi(t):
    """A non-uniform rotation and its first two derivatives."""
    phi = 1.3 * t + 0.4 * np.sin(0.7 * t) + 0.2 * np.cos(1.9 * t)
    d1 = 1.3 + 0.28 * np.cos(0.7 * t) - 0.38 * np.sin(1.9 * t)
    d2 = -0.196 * np.sin(0.7 * t) - 0.722 * np.cos(1.9 * t)
    return phi, d1, d2


@contextmanager
def _active(config):
    previous = active_config()
    set_active_config(config)
    try:
        yield config
    finally:
        set_active_config(previous)


def _plain_refuse():
    return load_search_config(overrides=collect_overrides(
        system_rps_mode='plain', engine='nsga2', system_rps_params={'invariant_terms': 'refuse'}))


@pytest.fixture(scope='module')
def heading():
    """(pool, rule) of a heading-vector pool: c, s and their derivatives to order 2, powers to 3. The rule is
    built while the trajectory is registered; only the pool's tokens are read afterwards."""
    previous = active_config()
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0.0, 40.0, 1200)
    phi, _, _ = _phi(t)
    try:
        _, domain = search.createDomain(t, boundary_width=10, ID=0)
        _, trajectory = search.createTrajectory({'c': np.cos(phi), 's': np.sin(phi)}, domain, cache_id=0)
        np.random.seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            search.create_pool(data=[trajectory], max_deriv_order=(2,), data_fun_pow=3)
            IT.reset()
            rule = IT.rule(search.pool)
        return search.pool, rule
    finally:
        global_var.samples_manager.reset()
        IT.reset()
        set_active_config(previous)


def _terms(pool, text):
    """The non-target terms of the c-equation ``text`` translated on ``pool``."""
    from epde.interface.equation_translator import translate_equation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        system = translate_equation({'c': text, 's': f'0.7 * {C} + 0.0 = {DS}'}, pool, all_vars=['c', 's'])
    eq = system.vals['c']
    eq.metaparameters['max_factors_in_term']['value'] = dict(LIVE_FACTORS)
    for term in eq.structure:
        term.max_factors_in_term = dict(LIVE_FACTORS)
    target = len(eq.structure) - 1
    return [term for i, term in enumerate(eq.structure) if i != target]


# ---------------------------------------------------------------------------
# 1. The configuration.
# ---------------------------------------------------------------------------

class TestConfig:

    def test_the_default_keeps_every_term(self):
        assert SYSTEM_RPS_PARAMS['plain']['invariant_terms'] == 'keep'
        assert resolve_plain_params(load_search_config(overrides=collect_overrides(
            system_rps_mode='plain', engine='nsga2')))['invariant_terms'] == 'keep'

    def test_refuse_is_accepted(self):
        check_plain_params(dict(SYSTEM_RPS_PARAMS['plain'], invariant_terms='refuse'))
        assert resolve_plain_params(_plain_refuse())['invariant_terms'] == 'refuse'

    @pytest.mark.parametrize('value', ['auto', 'drop', True, None])
    def test_other_values_are_refused(self, value):
        with pytest.raises(ValueError, match='invariant_terms'):
            check_plain_params(dict(SYSTEM_RPS_PARAMS['plain'], invariant_terms=value))

    def test_another_mode_refuses_the_key(self):
        with pytest.raises(ValueError, match='invariant_terms'):
            load_search_config(overrides=collect_overrides(
                system_rps_mode='assignment', system_rps_params={'invariant_terms': 'refuse'}))


# ---------------------------------------------------------------------------
# 2. Detection.
# ---------------------------------------------------------------------------

class TestDetect:

    def test_the_unit_circle_with_exact_fractions(self):
        phi = np.random.default_rng(0).uniform(0, 2 * np.pi, 500)
        found = IT.detect([np.sin(phi), np.cos(phi)])         # generators from the largest: s, c
        assert found == [{(2, 0): Fraction(1), (0, 2): Fraction(1), (0, 0): Fraction(-1)}]

    def test_a_circle_of_another_radius(self):
        phi = np.random.default_rng(1).uniform(0, 2 * np.pi, 500)
        found = IT.detect([3 * np.sin(phi), 3 * np.cos(phi)])
        assert found == [{(2, 0): Fraction(1), (0, 2): Fraction(1), (0, 0): Fraction(-9)}]

    def test_nothing_on_unrelated_data(self):
        rng = np.random.default_rng(2)
        assert IT.detect([rng.standard_normal(500), rng.standard_normal(500)]) == []

    def test_noise_breaks_the_invariant(self):
        rng = np.random.default_rng(3)
        phi = rng.uniform(0, 2 * np.pi, 500)
        assert IT.detect([np.sin(phi) + 1e-6 * rng.standard_normal(500), np.cos(phi)]) == []


# ---------------------------------------------------------------------------
# 3. The rule on a heading pool.
# ---------------------------------------------------------------------------

class TestRule:

    def test_the_pool_has_the_unit_circle(self, heading):
        _, rule = heading
        assert rule is not None and rule.variables == {'c', 's'}
        assert 's**2 + c**2 - 1 = 0' in rule.text or 'c**2 + s**2 - 1 = 0' in rule.text

    @pytest.mark.parametrize('text', [
        f'1.0 * {S} + 0.0 = {DC}',
        f'1.0 * {C} + 0.0 = {DC}',
        f'1.0 * {DDC} + 0.0 = {DC}',
        f'1.0 * c{{power: 2.0}} + 0.0 = {DC}',
        f'1.0 * {C} * {S} + 0.0 = {DC}',
        f'1.0 * {C} * dc/dx0{{power: 2.0}} + 0.0 = {DC}',      # a rate proxy, no identity
        f'1.0 * {C} * {DDC} + 0.0 = {DC}',
        f'1.0 * {C} * {DS} + 0.0 = {DC}',
    ])
    def test_kept(self, heading, text):
        pool, rule = heading
        assert not any(rule.reducible(term) for term in _terms(pool, text))

    @pytest.mark.parametrize('text', [
        f'1.0 * s{{power: 2.0}} + 0.0 = {DC}',
        f'1.0 * s{{power: 3.0}} + 0.0 = {DC}',
        f'1.0 * {S} * {DS} + 0.0 = {DC}',                     # c c' + s s' = 0
        f'1.0 * {C} * {DS} * s{{power: 3.0}} + 0.0 = {DC}',
        f'1.0 * {C} * {DC} * {S} + 0.0 = {DC}',               # c c' s = -s**2 s'
        f'1.0 * ds/dx0{{power: 2.0}} + 0.0 = {DC}',           # c'^2 + c c'' + s'^2 + s s'' = 0
    ])
    def test_refused(self, heading, text):
        pool, rule = heading
        assert all(rule.reducible(term) for term in _terms(pool, text))

    def test_standard_monomials_are_independent_and_span(self, heading):
        """On an exact jet of a non-uniform rotation, every monomial of degree <= 3 in (c, s, c', s', c'', s''):
        the ones the rule keeps are linearly independent, and adding the refused ones adds no rank."""
        _, rule = heading
        t = np.linspace(0.0, 60.0, 3000)
        phi, d1, d2 = _phi(t)
        c, s = np.cos(phi), np.sin(phi)
        jet = {('c', (0,)): c, ('s', (0,)): s, ('c', (1,)): -s * d1, ('s', (1,)): c * d1,
               ('c', (2,)): -c * d1 ** 2 - s * d2, ('s', (2,)): -s * d1 ** 2 + c * d2}
        keys = sorted(rule.index, key=rule.index.get)
        assert set(keys) == set(jet)
        kept, refused = [], []
        for degree in range(4):
            for combo in combinations_with_replacement(range(len(keys)), degree):
                exps = [0] * len(keys)
                for i in combo:
                    exps[i] += 1
                col = np.prod([jet[keys[i]] ** e for i, e in enumerate(exps)], axis=0) * np.ones_like(t)
                col = col / np.linalg.norm(col)
                reducible = any(all(e >= l for e, l in zip(exps, lm)) for lm in rule.lms)
                (refused if reducible else kept).append(col)
        assert refused and kept
        K = np.stack(kept, axis=1)
        A = np.stack(kept + refused, axis=1)
        tol = 1e-9
        assert np.linalg.matrix_rank(K, tol=tol * np.linalg.norm(K, 2)) == K.shape[1]
        assert np.linalg.matrix_rank(A, tol=tol * np.linalg.norm(A, 2)) == K.shape[1]

    def test_refused_is_all_false_under_keep_without_the_data(self, heading, monkeypatch):
        pool, _ = heading
        terms = _terms(pool, f'1.0 * s{{power: 2.0}} + 1.0 * {S} * {DS} + 0.0 = {DC}')

        def boom(_pool):
            raise AssertionError('the rule was built under keep')
        monkeypatch.setattr(IT, 'rule', boom)
        with _active(load_search_config(overrides=collect_overrides(system_rps_mode='plain', engine='nsga2'))):
            assert IT.refused(_Owner(pool), terms) == [False, False]


class _Owner(object):
    def __init__(self, pool):
        self.pool = pool


# ---------------------------------------------------------------------------
# 4. End to end.
# ---------------------------------------------------------------------------

def _search(**extra):
    """A small universal search on the heading vector: four terms of up to two factors, population 8."""
    return epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2', system_rps_mode='plain',
                           discrepancy_metric='wape', instability_metric='dyad',
                           equation_terms_max_number=4, equation_factors_max_number=2,
                           data_fun_pow=3, population_size=8, training_epochs=2, **extra)


def _heading_fit(search, seed):
    """The reported front (text, objectives) and every fitted equation of the front and the final population."""
    from epde.operators.common.objectives import LOSS_NAN_VAL
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0.0, 30.0, 600)
    phi, _, _ = _phi(t)
    np.random.seed(seed)
    domain = search.createDomain(t, ID=0)[1]
    trajectory = search.createTrajectory({'c': np.cos(phi), 's': np.sin(phi)}, domain, cache_id=0)[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        search.fit(data=[trajectory], max_deriv_order=(2,))
    front = list(search.optimizer.reported_front())
    fitted = []
    for system in front + list(search.optimizer.pareto_levels.population):
        for var in ('c', 's'):
            eq = system.vals[var]
            if PF._fitted(eq) and eq.fitness_value is not None and eq.fitness_value < LOSS_NAN_VAL:
                fitted.append(eq)
    text = [(system.text_form, [float(v) for v in system.obj_fun]) for system in front]
    return text, fitted, [str(w.message) for w in caught]


def _run(seed, **extra):
    previous = active_config()
    search = _search(**extra)
    try:
        text, fitted, caught = _heading_fit(search, seed)
        rule = IT.rule(search.pool) if PF.plain_params()['invariant_terms'] == 'refuse' else None
        refused = [term.name for eq in fitted for i, term in enumerate(eq.structure)
                   if i != eq.target_idx and rule is not None and rule.reducible(term)]
        return text, fitted, caught, refused
    finally:
        global_var.samples_manager.reset()
        PF.release()
        set_active_config(previous)


def test_no_fitted_equation_holds_a_refused_term(monkeypatch):
    seen = []
    real = IT.refused

    def counting(equation, terms):
        out = real(equation, terms)
        seen.extend(out)
        return out
    monkeypatch.setattr(IT, 'refused', counting)
    _, fitted, caught, refused = _run(0, system_rps_params={'invariant_terms': 'refuse'})
    assert fitted
    assert any(seen), 'the fit met no term to refuse; the test would pass vacuously'
    assert refused == []
    assert any("invariant_terms='refuse': the data satisfy" in m for m in caught)


def test_keep_equals_the_fit_without_the_key():
    a, _, _, _ = _run(1)
    b, _, _, _ = _run(1, system_rps_params={'invariant_terms': 'keep'})
    assert a == b
