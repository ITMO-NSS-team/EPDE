"""The code paths the universal search's settings switch on, on real systems and real rows.

The universal search is plain-mode NSGA-II (``system_rps_mode='plain'``, ``engine='nsga2'``): its settings
resolve to ``plain_target``, ``variable_order='random'``, the fill-and-prune move, the factor moves, the
product-rule decline on weak rows and per-variable niching. Pinned here:

* the plain fit -- the target candidates under ``plain_target`` and without it, the plain-target repair and
  its roll-back, the variable order (identity for step 1 when the claims cannot change, the module's own
  permutation for the redundancy test, the global stream untouched), the redundancy test under a permuted
  order, and the targets a real fit installs (``epde.operators.common.plain_fit``);
* the fill-and-prune move on weak rows -- the library, the install on a real system, the Gram store and its
  reuse, the beam on clean data, the fallback outside a system, and the plain mutation that runs it
  (``epde.operators.common.fill_prune``, ``epde.operators.multiobjective.plain_moves``);
* the per-variable pipeline on real fitted systems -- the mating plan, the assembled child, the truncation
  (``epde.optimizers.nsga.per_variable``);
* the product-rule decline on real terms (``epde.operators.common.product_rule``);
* end to end: a weak-form fit with the fill always drawn.
"""
import warnings
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.prepared_tokens import GridTokens
from epde.interface.search_config import (active_config, collect_overrides, load_search_config,
                                          plain_kept_terms_cap, set_active_config)
from epde.operators.common import fill_prune as FP
from epde.operators.common import plain_fit as PF
from epde.operators.common import product_rule as PR
from epde.operators.common import system_redundancy as SR
from epde.operators.multiobjective import plain_moves as PM
from epde.optimizers.nsga import per_variable as PV

U, V = 'u{power: 1.0}', 'v{power: 1.0}'
DU, DV = 'du/dx0{power: 1.0}', 'dv/dx0{power: 1.0}'
UV = f'{U} * {V}'
#: The factors setting the translated equations get (what a live search draws with).
LIVE_FACTORS = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
RANGES = [(0.0, 4 * np.pi)]
#: Plain-mode NSGA-II, the universal search, and the bare plain mode of the default engine.
UNIVERSAL = {'system_rps_mode': 'plain', 'engine': 'nsga2'}
BARE = {'system_rps_mode': 'plain'}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _config(**overrides):
    return load_search_config(overrides=collect_overrides(**overrides))


@contextmanager
def _active(config):
    """Run the block under ``config``, putting back the configuration active before."""
    previous = active_config()
    set_active_config(config)
    try:
        yield config
    finally:
        set_active_config(previous)


def _search(form='strong', factors=1, **extra):
    """A small universal search: two variables, four terms, population 8, two epochs."""
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2', system_rps_mode='plain',
                             discrepancy_metric='wape', instability_metric='dyad', evaluation_form=form,
                             equation_terms_max_number=4, equation_factors_max_number=factors,
                             data_fun_pow=1, population_size=8, training_epochs=2, **extra)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _trajectory(search):
    """u = sin t, v = cos t: u' = v, v' = -u."""
    t = np.linspace(0, 4 * np.pi, 300)
    domain = search.createDomain(t, ID=0)[1]
    return search.createTrajectory({'u': np.sin(t), 'v': np.cos(t)}, domain, cache_id=0)[1]


def _fit(search, seed):
    np.random.seed(seed)
    trajectory = _trajectory(search)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.fit(data=[trajectory], max_deriv_order=(1,))


def _system(pool, texts, weights):
    """``texts`` translated on ``pool``, pinned to a fitted state: target last, the given weights."""
    from epde.interface.equation_translator import translate_equation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        system = translate_equation(texts, pool, all_vars=list(texts))
    for var in texts:
        eq = system.vals[var]
        eq.main_var_to_explain = var
        eq.metaparameters['max_factors_in_term']['value'] = dict(LIVE_FACTORS)
        for term in eq.structure:
            term.max_factors_in_term = dict(LIVE_FACTORS)
        w = np.array(weights[var], dtype=float)
        eq.weights_internal = w
        eq.weights_final = w.copy()
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        eq.target_idx = len(eq.structure) - 1
    return system


def _fitted_systems(systems):
    """The systems every equation of which is fitted."""
    return [system for system in systems if all(PF._fitted(eq) for eq in system.vals)]


def _index(population, member):
    return next(i for i, system in enumerate(population) if system is member)


@pytest.fixture(autouse=True)
def _plain_state(monkeypatch):
    """Every cache plain mode keeps between calls starts empty and is put back after the test."""
    for module, name, empty in ((PF, '_SIBLINGS', {}), (PF, '_PLAIN_TOKENS', {}), (FP, '_LIBRARY', {}),
                                (FP, '_STORES', {}), (FP, '_WARNED', set()), (PR, '_GROUPS', {}),
                                (SR, '_REPORTED', set())):
        monkeypatch.setattr(module, name, type(empty)())
    monkeypatch.setitem(PF._ORDER, 'rng', None)
    yield


@pytest.fixture(scope='module')
def strong_fit():
    """A finished universal fit on strong rows. Its systems are read only, so the samples manager is
    released right after the fit."""
    previous = active_config()
    search = _search()
    try:
        _fit(search, seed=2)
        result = SimpleNamespace(pool=search.pool, levels=search.optimizer.pareto_levels,
                                 population=list(search.optimizer.pareto_levels.population),
                                 front=list(search.optimizer.reported_front()))
    finally:
        global_var.samples_manager.reset()
        PF.release()
        set_active_config(previous)
    return result


@pytest.fixture(scope='class')
def weak_fit():
    """A finished universal fit on weak rows with the fill always drawn; the samples manager stays live
    for the class (the fill reads the rows)."""
    previous = active_config()
    search = _search('weak', factors=2, system_rps_params={'fill_probability': 1.0})
    try:
        _fit(search, seed=0)
        yield SimpleNamespace(search=search, config=active_config(),
                              population=list(search.optimizer.pareto_levels.population))
    finally:
        global_var.samples_manager.reset()
        PF.release()
        set_active_config(previous)


# ---------------------------------------------------------------------------
# A. The plain fit under the universal settings.
# ---------------------------------------------------------------------------

class TestPlainFitUnderUniversal:

    def test_only_a_plain_own_derivative_is_a_target_candidate(self, strong_fit):
        texts = {'u': f'0.5 * {U} * {DU} + 0.3 * {V} + 0.0 = {DU}', 'v': f'1.0 * {U} + 0.0 = {DV}'}
        eq = _system(strong_fit.pool, texts, {'u': [0.5, 0.3, 0.0], 'v': [1.0, 0.0]}).vals['u']
        decorated, other, plain = eq.structure
        assert PF.is_plain_term(plain, 'u')
        assert not PF.is_plain_term(decorated, 'u') and decorated.contains_deriv('u')
        assert not PF.is_plain_term(other, 'u') and not other.contains_deriv('u')
        with _active(_config(**UNIVERSAL)):
            assert PF.plain_params()['plain_target'] is True
            assert PF.own_derivative_terms(eq) == [plain]
        with _active(_config(**BARE)):
            assert PF.plain_params()['plain_target'] is False
            assert PF.own_derivative_terms(eq) == [decorated, plain]

    def _without_plain_target(self, pool):
        texts = {'u': f'0.5 * {U} * {DU} + 0.0 = {V}', 'v': f'1.0 * {U} + 0.0 = {DV}'}
        eq = _system(pool, texts, {'u': [0.5, 0.0], 'v': [1.0, 0.0]}).vals['u']
        assert not any(PF.is_plain_term(term, 'u') for term in eq.structure)
        return eq

    def test_the_plain_target_repair_brings_a_plain_term(self, strong_fit):
        eq = self._without_plain_target(strong_fit.pool)
        np.random.seed(0)
        assert PF._ensure_target(eq, plain=True)
        assert any(PF.is_plain_term(term, 'u') for term in eq.structure)

    def test_a_failed_plain_target_repair_rolls_the_structure_back(self, strong_fit, monkeypatch):
        eq = self._without_plain_target(strong_fit.pool)
        calls = []

        def no_plain_term(self, deriv=False, mandatory_family=False, t_derivative=False):
            calls.append(deriv)
            self.structure = [self.structure[0]]          # a change that brings no plain term
        monkeypatch.setattr(type(eq), 'restore_property', no_plain_term)
        before = list(eq.structure)
        assert PF._ensure_target(eq, plain=True) is False
        assert len(calls) == PF.ENSURE_MAX_ITER and all(calls)
        assert len(eq.structure) == len(before)
        assert all(a is b for a, b in zip(eq.structure, before))

    def test_the_variable_order_is_the_identity_without_a_choice(self, strong_fit):
        system = _fitted_systems(strong_fit.population)[0]
        equations = list(system)
        fitted = [PF._fitted(eq) for eq in equations]
        with _active(_config(**UNIVERSAL)):
            params = PF.plain_params()
            assert params['variable_order'] == 'random'
            assert PF._variable_order(equations[:1], fitted[:1], params) == ([0], None)
            assert PF._variable_order(equations, fitted, dict(params, variable_order='fixed')) == ([0, 1], None)

    @pytest.mark.parametrize('claims', ['fitted', 'genes'])
    def test_settled_claims_keep_step_one_in_order_and_permute_the_test(self, strong_fit, claims):
        system = deepcopy(_fitted_systems(strong_fit.population)[0])
        equations = list(system)
        if claims == 'genes':
            for eq in equations:
                eq.reset_state(True)
                assert eq._plain_genes is not None and PF.gene_term(eq) is not None
        fitted = [PF._fitted(eq) for eq in equations]
        assert all(fitted) == (claims == 'fitted') and (any(fitted) or claims == 'genes')
        # A seed whose permutation of two positions is not the identity.
        seed = next(s for s in range(100) if np.random.default_rng([s, 0x5256]).permutation(2).tolist() == [1, 0])
        expected = np.random.default_rng([seed, 0x5256]).permutation(len(equations)).tolist()
        with _active(_config(**UNIVERSAL)):
            params = PF.plain_params()
            PF.start_variable_order(seed)
            np.random.seed(5)
            state = np.random.get_state()
            order, test_order = PF._variable_order(equations, fitted, params)
            after = np.random.get_state()
        assert order == [0, 1]
        assert test_order == expected == [1, 0]
        assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]

    def test_a_restated_relation_loses_the_statement_tested_first(self, strong_fit):
        # u' + v' = 20 (u - v), solved once for u' and once for v': each follows from the other.
        texts = {'u': f'-1.0 * {DV} + 20.0 * {U} + -20.0 * {V} + 0.0 = {DU}',
                 'v': f'-1.0 * {DU} + 20.0 * {U} + -20.0 * {V} + 0.0 = {DV}'}
        weights = {'u': [-1.0, 20.0, -20.0, 0.0], 'v': [-1.0, 20.0, -20.0, 0.0]}
        system = _system(strong_fit.pool, texts, weights)
        equations = [system.vals['u'], system.vals['v']]
        assert SR.redundant_equations(equations, RANGES) == [0]
        assert SR.redundant_equations(equations, RANGES, order=[0, 1]) == [0]
        assert SR.redundant_equations(equations, RANGES, order=[1, 0]) == [1]

    def test_independent_laws_follow_in_no_order(self, strong_fit):
        texts = {'u': f'20.0 * {U} + -20.0 * {UV} + 0.0 = {DU}', 'v': f'-20.0 * {V} + 20.0 * {UV} + 0.0 = {DV}'}
        system = _system(strong_fit.pool, texts, {'u': [20.0, -20.0, 0.0], 'v': [-20.0, 20.0, 0.0]})
        equations = [system.vals['u'], system.vals['v']]
        assert SR.redundant_equations(equations, RANGES) == []
        assert SR.redundant_equations(equations, RANGES, order=[1, 0]) == []

    def test_a_real_fit_installs_distinct_plain_targets(self, strong_fit):
        systems = strong_fit.population + strong_fit.front
        assert _fitted_systems(systems)
        for system in systems:
            signatures = []
            for var in system.vars_to_describe:
                eq = system.vals[var]
                if PF._fitted(eq):
                    assert PF.is_plain_term(eq.target, var)
                    signatures.append(eq.target.factors_labels)
            assert len(set(signatures)) == len(signatures)


# ---------------------------------------------------------------------------
# B. The fill-and-prune move on weak rows.
# ---------------------------------------------------------------------------

def _name(var):
    return '%s{power: 1.0}' % var


class TestFillOnWeakRows:

    def _system(self, weak_fit):
        """A copy of a fitted system of the fit (the tests change what they take)."""
        return deepcopy(_fitted_systems(weak_fit.population)[0])

    def _fill_until_installed(self, system, var, params, tries=50):
        eq = system.vals[var]
        target = PF.gene_term(eq)
        for _ in range(tries):
            features = [term for term in eq.structure if term is not target]
            if FP.fill(eq, target, features, system, params):
                return eq, target
        raise AssertionError('the fill installed nothing in %d tries' % tries)

    def test_the_library_is_the_pool_without_repeats(self, weak_fit):
        eq = weak_fit.population[0].vals['u']
        library = FP.library_terms(eq.pool, FP.factors_setting(eq))
        labels = [term.factors_labels for term in library]
        assert len(set(labels)) == len(labels)
        cap = FP.factor_cap(eq)
        assert cap == 2
        assert all(1 <= len(term.structure) <= cap for term in library)
        assert any(len(term.structure) == cap for term in library)
        for var in ('u', 'v'):
            assert any(PF.is_plain_term(term, var) for term in library)

    def test_the_fill_installs_on_a_real_system(self, weak_fit):
        with _active(weak_fit.config):
            params = dict(PF.plain_params(), fill_probability=1.0)
            cap = plain_kept_terms_cap(active_config().search_space.equation_terms_max_number)
            system = self._system(weak_fit)
            other_target = system.vals['v'].target.factors_labels
            np.random.seed(0)
            with PM.mutating_system(system):
                eq, target = self._fill_until_installed(system, 'u', params)
                assert any(term is target for term in eq.structure)
                assert PF.gene_term(eq) is target
                assert len(eq.structure) <= cap + 1
                labels = [term.factors_labels for term in eq.structure]
                assert len(set(labels)) == len(labels)
                assert other_target not in labels
                store = FP._STORES['u']
                Gn, ok = store['Gn'], store['ok']
                assert ok.any()
                np.testing.assert_allclose(np.diag(Gn)[ok], 1.0, rtol=0, atol=1e-12)
                np.testing.assert_allclose(Gn, Gn.T, rtol=0, atol=1e-12)
                assert np.all(np.abs(Gn[np.ix_(ok, ok)]) <= 1.0 + 1e-9)
                FP.fill(eq, target, [term for term in eq.structure if term is not target], system, params)
                assert FP._STORES['u'] is store

    @pytest.mark.parametrize('var, law', [('u', 'v'), ('v', 'u')])
    def test_the_beam_finds_the_law_on_clean_data(self, weak_fit, var, law):
        with _active(weak_fit.config):
            params = dict(PF.plain_params(), fill_probability=1.0)
            system = self._system(weak_fit)
            eq = system.vals[var]
            target = PF.gene_term(eq)
            assert target.name == 'd%s/dx0{power: 1.0}' % var
            np.random.seed(0)
            with PM.mutating_system(system):
                FP.fill(eq, target, [term for term in eq.structure if term is not target], system, params)
        store = FP._STORES[var]
        keys = [key for key in store['beams'] if key[0] == var and key[1] == target.factors_labels]
        assert len(keys) == 1
        status, levels = store['beams'][keys[0]]
        assert status == 'ok'
        best, rss = levels[1][0]
        assert [store['lib'][i].name for i in best] == [_name(law)]
        assert rss < 1e-8

    def test_outside_a_mutated_system_the_fill_changes_nothing(self, weak_fit):
        with _active(weak_fit.config):
            params = dict(PF.plain_params(), fill_probability=1.0)
            system = self._system(weak_fit)
            eq = system.vals['u']
            target = PF.gene_term(eq)
            features = [term for term in eq.structure if term is not target]
            before = list(eq.structure)
            assert PM._MUTATING['system'] is None
            np.random.seed(0)
            for _ in range(10):
                assert FP.fill(eq, target, features, PM._MUTATING['system'], params) is False
            # A system that does not hold this equation is no system for it either.
            assert FP.fill(eq, target, features, deepcopy(system), params) is False
        assert len(eq.structure) == len(before) and all(a is b for a, b in zip(eq.structure, before))
        assert FP._STORES == {}

    def test_a_fill_mutation_resets_the_equation_and_keeps_the_gene(self, weak_fit, monkeypatch):
        filled = []
        real_fill = FP.fill
        monkeypatch.setattr(FP, 'fill', lambda *args: filled.append(real_fill(*args)) or filled[-1])
        with _active(weak_fit.config):
            assert PF.plain_params()['fill_probability'] == 1.0
            reset = 0
            for seed in range(8):
                system = self._system(weak_fit)
                eq = system.vals['u']
                assert PF._fitted(eq)
                del filled[:]
                np.random.seed(seed)
                with PM.mutating_system(system):
                    PM.plain_mutation(eq)
                assert len(eq.terms_labels) == len(eq.structure)
                if filled and filled[-1]:
                    reset += 1
                    assert not PF._fitted(eq)
                    assert eq.target is None
                    assert not (eq.right_part_selected or eq.weights_internal_evald or eq.weights_final_evald
                                or eq.fitness_calculated)
                    found = PF.gene_term(eq)
                    assert found is not None and any(term is found for term in eq.structure)
        assert reset >= 1


# ---------------------------------------------------------------------------
# C. The per-variable pipeline on real fitted systems.
# ---------------------------------------------------------------------------

class TestPerVariableOnFittedSystems:

    def test_the_mating_plan_pairs_two_planes(self, strong_fit):
        levels = deepcopy(strong_fit.levels)
        population = levels.population
        operator = SimpleNamespace(params={'parents_fraction': 1.0})
        n = len(population)
        start = levels.__dict__.get('_pv_turn', 0)          # the fit's own selections turned it already
        np.random.seed(1)
        for call in (1, 2):
            PV.select(operator, levels)
            assert len(levels.pv_plan) == (n + n % 2) // 2
            assert len(levels.pv_coins) == len(levels.pv_plan)
            for A, kA, B, kB in levels.pv_plan:
                assert kA != kB and {kA, kB} <= {'u', 'v'}
                assert any(A is s for s in population) and any(B is s for s in population)
            assert levels._pv_turn == start + call
        assert not hasattr(operator, '_pv_turn')

    def test_the_assembled_child_takes_one_equation_from_each_parent(self, strong_fit):
        A, B = _fitted_systems(strong_fit.population)[:2]
        keys = list(A.vals.equation_keys)
        before = [(s.equations_labels, [PF._fitted(eq) for eq in s.vals]) for s in (A, B)]
        child = PV.assemble(A, 'u', B, 'v', keys, frozenset())
        assert child.vals['v'].terms_labels == B.vals['v'].terms_labels
        assert child.vals['u'].terms_labels == A.vals['u'].terms_labels
        assert child.vals['v'] is not B.vals['v'] and child.vals['u'] is not A.vals['u']
        assert not any(PF._fitted(eq) for eq in child.vals)
        # Unfitted, the child's structure labels read no constant until its own fit.
        assert child.vals['v'].structure_label == (B.vals['v'].terms_labels, False)
        assert child.vals['u'].structure_label == (A.vals['u'].terms_labels, False)
        assert PV.skip_first_mutation(child) is True
        assert PV.skip_first_mutation(child) is False
        assert [(s.equations_labels, [PF._fitted(eq) for eq in s.vals]) for s in (A, B)] == before

    def test_the_truncation_keeps_order_levels_and_first_level_holders(self, strong_fit):
        population = list(strong_fit.population)
        mu = len(population)
        extra = [s for s in strong_fit.front if not any(s is p for p in population)]
        candidates = population + extra + [deepcopy(population[0])]
        assert len(candidates) > mu
        T = PV.plane_table(candidates)
        first = [{T.labels[e][i] for i in range(T.n) if T.lev[e, i] == 0} for e in range(T.n_eq)]
        assert sum(len(labels) for labels in first) <= mu
        optimizer = SimpleNamespace(pareto_levels=SimpleNamespace(population=list(candidates), levels=[]), _mu=mu)
        PV.truncate(optimizer)
        kept = optimizer.pareto_levels.population
        assert len(kept) == mu
        positions = [_index(candidates, s) for s in kept]
        assert positions == sorted(positions) and len(set(positions)) == mu
        layered = [s for level in optimizer.pareto_levels.levels for s in level]
        assert len(layered) == mu and sorted(_index(kept, s) for s in layered) == list(range(mu))
        held = PV.plane_table(kept).labels
        for e, labels in enumerate(first):
            for label in labels:
                assert label in held[e]


# ---------------------------------------------------------------------------
# The product-rule decline on real terms.
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_library():
    """The library terms (two factors at most) of a pool with u, rho, their first and second derivatives and
    the grid token t."""
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0.0, 4 * np.pi, 200)
    try:
        domain = search.createDomain(t, ID=0, boundary_width=10)[1]
        trajectory = search.createTrajectory({'u': np.sin(t), 'rho': np.cos(t) + 2.0}, domain, cache_id=0)[1]
        np.random.seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            search.create_pool(data=[trajectory], max_deriv_order=(2,), data_fun_pow=1,
                               additional_tokens=[GridTokens(['t'])])
        library = FP.library_terms(search.pool, 2)
    finally:
        global_var.samples_manager.reset()
    return {PR.term_key(term): term for term in library}


def _version(target, feature, coefficient):
    """A fitted version as the decline reads it: ``target = coefficient * feature``, no constant."""
    return SimpleNamespace(structure=[target, feature], target_idx=0,
                           weights_internal=np.array([coefficient, 0.0]),
                           weight_index=lambda i, t: i - (i > t))


class TestProductRuleOnRealTerms:

    def test_the_term_keys_parse_back(self, grid_library):
        assert {'du/dx0', 'd^2u/dx0^2*t@0', 'drho/dx0', 'd^2rho/dx0^2*t@0'} <= set(grid_library)
        for key, term in grid_library.items():
            assert PR.key_of(PR.mono(key)) == key

    @pytest.mark.parametrize('mode', ['plain', 'trig'])
    def test_t_times_u_xx_beside_u_x_is_declined_by_its_sign(self, grid_library, mode):
        # F = t u_x along t: D(t u_x) = u_x + t u_xx, so u_x = c t u_xx is the pattern when c < 0.
        target, feature = grid_library['du/dx0'], grid_library['d^2u/dx0^2*t@0']
        assert PR.declines(_version(target, feature, -0.7), mode)
        assert not PR.declines(_version(target, feature, 0.7), mode)
        assert not PR.declines(_version(target, feature, -0.7), 'off')

    @pytest.mark.parametrize('coefficient', [-0.7, 0.7])
    def test_a_variable_named_with_several_letters_is_never_declined(self, grid_library, coefficient):
        # The rule reads one-letter variable names only: rho's derivatives are opaque to it.
        target, feature = grid_library['drho/dx0'], grid_library['d^2rho/dx0^2*t@0']
        for mode in ('plain', 'trig'):
            assert not PR.declines(_version(target, feature, coefficient), mode)


# ---------------------------------------------------------------------------
# D. End to end on weak rows.
# ---------------------------------------------------------------------------

class TestWeakEndToEnd:

    def test_a_weak_fit_with_the_fill_always_drawn(self):
        previous = active_config()
        search = _search('weak', factors=2, system_rps_params={'fill_probability': 1.0})
        try:
            _fit(search, seed=1)
            assert PF.plain_params()['fill_probability'] == 1.0
            assert len(search.pareto_history) == 2
            front = search.optimizer.reported_front()
            assert front
            for system in front:
                for var in system.vars_to_describe:
                    eq = system.vals[var]
                    if eq.target is not None:
                        assert PF.is_plain_term(eq.target, var)
            assert FP._STORES
            assert set(FP._STORES) <= {'u', 'v'}
        finally:
            global_var.samples_manager.reset()
            set_active_config(previous)
