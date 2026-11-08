"""Plain-mode NSGA-II's universal search: its settings and each of its parts.

* settings -- the 'auto' values resolve to the universal search under
  ``engine='nsga2'`` in plain mode and to the bare mode otherwise; the
  refusals;
* the product-rule decline (``epde.operators.common.product_rule``) on term
  keys;
* the fill-and-prune move's exact beam and its draws
  (``epde.operators.common.fill_prune``);
* the per-variable truncation (``epde.optimizers.nsga.per_variable``);
* the variable order's generator (``plain_fit.start_variable_order``);
* end to end: a two-equation system under the defaults.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import (PLAIN_AUTO, SYSTEM_RPS_PARAMS, collect_overrides,
                                          load_search_config, resolve_niching, resolve_patience,
                                          resolve_plain_params)
from epde.operators.common import fill_prune as FP
from epde.operators.common import plain_fit as PF
from epde.operators.common import product_rule as PR
from epde.optimizers.nsga import per_variable as PV
from epde.operators.multiobjective.moeadd_specific import plane_levels


def _load(**overrides):
    return load_search_config(overrides=collect_overrides(**overrides))


def _resolved(cfg):
    """(niching, early_stopping_patience, plain settings) of ``cfg``, 'auto' resolved as the readers do."""
    evolution, space = cfg.evolution, cfg.search_space
    return (resolve_niching(evolution.niching, space.system_rps_mode, evolution.engine),
            resolve_patience(evolution.early_stopping_patience, space.system_rps_mode, evolution.engine),
            resolve_plain_params(cfg))


# ---------------------------------------------------------------------------
# Settings.
# ---------------------------------------------------------------------------

class TestSettings:

    def test_plain_nsga2_resolves_to_the_universal_search(self):
        cfg = _load(system_rps_mode='plain', engine='nsga2', evaluation_form='weak')
        # The config keeps the declared 'auto'; the readers resolve it.
        assert (cfg.evolution.niching, cfg.evolution.early_stopping_patience) == ('auto', 'auto')
        assert cfg.search_space.system_rps_params == SYSTEM_RPS_PARAMS['plain']
        niching, patience, plain = _resolved(cfg)
        assert (niching, patience) == ('per_variable', None)
        for key, value in PLAIN_AUTO['universal'].items():
            assert plain[key] == value
        assert plain['product_rule_decline'] == 'trig'
        assert (plain['fill_beam_width'], plain['fill_top'], plain['fill_prune_alpha'],
                plain['fill_levels'], plain['variable_order_seed']) == (1000, 20, 0.3, 'all', None)

    def test_strong_rows_have_no_product_rule_decline(self):
        cfg = _load(system_rps_mode='plain', engine='nsga2')
        assert resolve_plain_params(cfg)['product_rule_decline'] == 'off'
        assert resolve_plain_params(cfg)['fill_probability'] == 0.25

    @pytest.mark.parametrize('overrides', [
        {}, {'system_rps_mode': 'plain'}, {'engine': 'nsga2'},
        {'system_rps_mode': 'plain', 'evaluation_form': 'weak'}])
    def test_anything_else_keeps_the_bare_mode(self, overrides):
        niching, patience, plain = _resolved(_load(**overrides))
        assert (niching, patience) == ('system', 5)
        for key, value in PLAIN_AUTO['bare'].items():
            assert plain[key] == value
        assert plain['product_rule_decline'] == 'off'

    def test_every_part_can_be_switched_off(self):
        cfg = _load(system_rps_mode='plain', engine='nsga2', evaluation_form='weak',
                    niching='system', early_stopping_patience=3,
                    system_rps_params=dict(fill_probability=0.0, product_rule_decline='off',
                                           variable_order='fixed', plain_target=False,
                                           factor_move_probability=0.0))
        niching, patience, plain = _resolved(cfg)
        assert (niching, patience) == ('system', 3)
        assert {k: plain[k] for k in PLAIN_AUTO['bare']} == PLAIN_AUTO['bare']
        assert plain['product_rule_decline'] == 'off'

    def test_the_declared_defaults_are_auto(self):
        params = SYSTEM_RPS_PARAMS['plain']
        for key in ('fill_probability', 'product_rule_decline', 'variable_order', 'plain_target',
                    'factor_move_probability'):
            assert params[key] == 'auto'

    @pytest.mark.parametrize('overrides, match', [
        ({'niching': 'per_variable'}, 'per_variable'),
        ({'system_rps_mode': 'plain', 'niching': 'per_variable'}, 'per_variable'),
        ({'engine': 'nsga2', 'niching': 'per_variable'}, 'per_variable'),
        ({'system_rps_mode': 'plain', 'engine': 'nsga2', 'complementary_mating_prob': 0.5},
         'complementary_mating_prob'),
        ({'system_rps_mode': 'plain', 'engine': 'nsga2',
          'system_rps_params': {'product_rule_decline': 'trig'}}, 'weak'),
        ({'system_rps_mode': 'plain', 'system_rps_params': {'fill_probability': 1.5}}, 'fill_probability'),
        ({'system_rps_mode': 'plain', 'system_rps_params': {'factor_move_probability': -0.1}},
         'factor_move_probability'),
        ({'system_rps_mode': 'plain', 'system_rps_params': {'fill_beam_width': 0}}, 'fill_beam_width'),
        ({'system_rps_mode': 'plain', 'system_rps_params': {'fill_top': None}}, 'fill_top'),
        ({'system_rps_mode': 'plain', 'system_rps_params': {'fill_prune_alpha': 0}}, 'fill_prune_alpha'),
        ({'system_rps_mode': 'plain', 'system_rps_params': {'fill_levels': 'some'}}, 'fill_levels'),
        ({'system_rps_mode': 'plain', 'system_rps_params': {'variable_order': 'sorted'}}, 'variable_order'),
        ({'system_rps_mode': 'plain', 'system_rps_params': {'plain_target': 1}}, 'plain_target'),
        ({'system_rps_mode': 'plain', 'system_rps_params': {'variable_order_seed': 1.5}},
         'variable_order_seed'),
        ({'system_rps_mode': 'assignment', 'system_rps_params': {'fill_probability': 0.1}}, 'plain'),
    ])
    def test_refusals(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            _load(**overrides)

    def test_complementary_mating_keeps_working_with_system_niching(self):
        cfg = _load(system_rps_mode='plain', engine='nsga2', niching='system', complementary_mating_prob=0.5)
        assert cfg.evolution.complementary_mating_prob == 0.5


# ---------------------------------------------------------------------------
# The product-rule decline, on term keys.
# ---------------------------------------------------------------------------

U_X, U_XX, U_XXX = 'du/dx1', 'd^2u/dx1^2', 'd^3u/dx1^3'


class TestProductRule:

    def test_the_group_of_u_times_u_x(self):
        keys = [U_X + '^2', U_XX + '*u', 'du/dx0']
        groups = PR.groups_in(keys)
        assert groups == (('du/dx1*u', 1, ((U_XX + '*u', 1), (U_X + '^2', 1))),)

    def test_a_square_counts_its_multiplicity(self):
        # F = u^2 along x1: D(u^2) = 2 u u_x, one member only: no group.
        assert PR.group_of({('u', ''): 2}, 1) is None
        # F = u^2 u_x: members 2 u u_x^2 and u^2 u_xx.
        group = PR.group_of({('u', ''): 2, (U_X, ''): 1}, 1)
        assert group == {U_X + '^2*u': 2, U_XX + '*u^2': 1}

    def test_the_kdv_relation_is_declined_and_a_law_is_not(self):
        # Weak rows: u_xx^2 = -0.99999 u_x u_xxx (+ 0.0316 u_tt) is integration by parts, not a law.
        keys = [U_XX + '^2', U_XXX + '*' + U_X, 'd^2u/dx0^2']
        target = U_XX + '^2'
        (group,) = PR.target_groups(keys, target)
        relation = PR.relation_from_fit(target, {U_XXX + '*' + U_X: -0.99999, 'd^2u/dx0^2': 0.0316})
        assert PR.sign_pattern(group, relation)
        # The opposite sign is no product-rule pattern.
        relation = PR.relation_from_fit(target, {U_XXX + '*' + U_X: 0.5, 'd^2u/dx0^2': 0.0316})
        assert not PR.sign_pattern(group, relation)
        # A zero or a missing member is not the pattern.
        assert not PR.sign_pattern(group, PR.relation_from_fit(target, {U_XXX + '*' + U_X: 0.0}))
        assert not PR.sign_pattern(group, PR.relation_from_fit(target, {}))

    def test_a_target_outside_every_group_is_never_declined(self):
        keys = [U_XX + '^2', U_XXX + '*' + U_X, 'du/dx0']
        assert PR.target_groups(keys, 'du/dx0') == ()

    def test_sin_and_cos_need_the_trig_mode(self):
        keys = ['du/dx1*sin~2@1', 'cos~2@1*u']
        assert PR.groups_in(keys, trig=False) == ()
        (group,) = PR.groups_in(keys, trig=True)
        assert dict(group[2]) == {'du/dx1*sin~2@1': 1, 'cos~2@1*u': 2}

    def test_an_opaque_factor_belongs_to_no_group(self):
        assert PR.groups_in(['du/dx1^1.5', U_XX + '*u^0.5']) == ()

    def test_declines_reads_the_fitted_version(self, monkeypatch):
        monkeypatch.setattr(PR, 'term_key', lambda term: term.key)
        structure = [SimpleNamespace(key=U_XX + '^2'), SimpleNamespace(key=U_XXX + '*' + U_X)]

        def version(coef, const=0.0):
            return SimpleNamespace(structure=structure, target_idx=0,
                                   weights_internal=np.array([coef, const]),
                                   weight_index=lambda i, t: i - (i > t))
        assert PR.declines(version(-1.0), 'plain')
        assert not PR.declines(version(1.0), 'plain')
        assert not PR.declines(version(-1.0), 'off')
        with pytest.raises(ValueError, match='product_rule_decline'):
            PR.declines(version(-1.0), 'sometimes')


# ---------------------------------------------------------------------------
# The fill-and-prune move: the exact beam and its draws.
# ---------------------------------------------------------------------------

def _gram(n=200, m=8, support=(1, 4), seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    y = X[:, list(support)] @ np.array([1.5, -0.7][:len(support)]) + noise * rng.standard_normal(n)
    return FP.Gram(X.T @ X, X.T @ y, float(y @ y)), X, y


def _relative_rss(X, y, S):
    if not S:
        return 1.0
    c, *_ = np.linalg.lstsq(X[:, list(S)], y, rcond=None)
    r = y - X[:, list(S)] @ c
    return float(r @ r / (y @ y))


class TestFillAndPrune:

    def test_the_gram_fit_is_least_squares(self):
        gm, X, y = _gram(noise=0.3)
        for S in [(0,), (1, 4), (0, 2, 5, 7)]:
            assert gm.rss(S) == pytest.approx(_relative_rss(X, y, S), abs=1e-10)

    def test_the_gains_are_the_exact_residual_decrease(self):
        gm, X, y = _gram(noise=0.3)
        S = (2,)
        gains, bad = gm.gains_mask(S)
        for z in range(gm.M):
            if z in S:
                assert gains[z] == 0.0
                continue
            assert not bad[z]
            assert gains[z] == pytest.approx(gm.rss(S) - gm.rss(S + (z,)), abs=1e-10)

    def test_the_beam_holds_the_exact_best_supports(self):
        gm, X, y = _gram(noise=0.05)
        levels = FP.beam_search(gm, width=1000, kmax=3)
        # A width above every candidate count makes the beam exhaustive.
        from itertools import combinations
        for k in (1, 2, 3):
            got = levels[k]
            assert [r for _, r in got] == sorted(r for _, r in got)
            assert len(got) == len(list(combinations(range(gm.M), k)))
            best = min(combinations(range(gm.M), k), key=lambda S: _relative_rss(X, y, S))
            assert got[0][0] == best
        assert levels[2][0][0] == (1, 4)

    def test_the_beam_keeps_the_width_and_excludes_supports_in_span(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 5))
        X = np.column_stack([X, X[:, 0] + X[:, 1]])            # column 5 in span{0, 1}
        y = X[:, 2] - X[:, 3]
        gm = FP.Gram(X.T @ X, X.T @ y, float(y @ y))
        levels = FP.beam_search(gm, width=4, kmax=3)
        assert all(len(level) <= 4 for level in levels.values())
        everything = FP.beam_search(gm, width=1000, kmax=3)
        assert not any(set(S) >= {0, 1, 5} for S, _ in everything[3])

    def test_the_fill_draw_is_weighted_by_the_inverse_residual(self):
        top = [((0,), 0.01), ((1,), 0.1), ((2,), 1.0)]
        np.random.seed(0)
        draws = np.bincount([FP.draw_fill(top) for _ in range(3000)], minlength=3) / 3000
        w = np.array([100.0, 10.0, 1.0])
        assert draws == pytest.approx(w / w.sum(), abs=0.03)

    def test_the_prune_chain_shrinks_to_two_terms(self):
        gm, _, _ = _gram(m=8, noise=0.2)
        np.random.seed(3)
        stages = FP.prune_chain(gm, list(range(8)), alpha=0.3)
        assert stages[0] == tuple(range(8))
        assert len(stages[-1]) == 2
        for before, after in zip(stages, stages[1:]):
            assert set(after) < set(before)
        assert FP.prune_chain(gm, [3, 5], alpha=0.3) == [(3, 5)]

    def test_a_zero_probability_draws_nothing(self):
        state = np.random.get_state()
        assert FP.fill(None, None, [], None, {'fill_probability': 0.0}) is False
        assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                   for a, b in zip(state, np.random.get_state()))


# ---------------------------------------------------------------------------
# Per-variable truncation.
# ---------------------------------------------------------------------------

class _Vals(dict):
    @property
    def equation_keys(self):
        return list(self.keys())


def _system(labels, objectives):
    """A system as the plane ranking reads it: objectives grouped by type, then by equation."""
    vals = _Vals({key: SimpleNamespace(structure_label=label, terms_labels=label)
                  for key, label in zip('uv', labels)})
    system = SimpleNamespace(vals=vals, obj_fun=np.asarray(objectives, dtype=float))
    system.equations_labels = tuple(labels)
    return system


class TestPerVariable:

    @pytest.mark.parametrize('seed', range(5))
    def test_the_fast_plane_levels_are_the_reference_ones(self, seed):
        rng = np.random.default_rng(seed)
        points = rng.integers(0, 5, size=(30, 2)).astype(float)
        labels = [str(int(x)) for x in rng.integers(0, 8, size=30)]
        assert np.array_equal(PV.plane_levels_fast(points, labels), plane_levels(points, labels))

    def test_one_copy_of_every_front_equation_comes_first(self):
        # u's front: label A (systems 0, 1 hold it), v's front: label X (system 2). The deeper systems 3, 4
        # dominate nothing; system 1 duplicates system 0's u law and represents it only if it wins the tie.
        population = [_system(['A', 'Y'], [0.1, 0.6, 0.1, 0.6]),   # (u_disc, v_disc, u_inst, v_inst)
                      _system(['A', 'Z'], [0.1, 0.7, 0.1, 0.7]),
                      _system(['B', 'X'], [0.5, 0.1, 0.5, 0.1]),
                      _system(['C', 'W'], [0.9, 0.9, 0.9, 0.9]),
                      _system(['D', 'V'], [0.8, 0.8, 0.8, 0.8])]
        T = PV.plane_table(population)
        assert T.front[0].tolist() == [True, False, False, False, False]      # one representative of A
        assert T.front[1].tolist() == [False, False, True, False, False]
        kept = PV.survivors(T, 2, start=0)
        assert sorted(kept) == [0, 2]
        assert sorted(PV.survivors(T, 2, start=1)) == [0, 2]
        assert len(PV.survivors(T, 3)) == 3
        assert PV.survivors(T, 10) == list(range(5))

    def test_the_first_level_count_is_per_plane(self):
        count = PV.PerPlaneFirstLevelCount()
        # Each system is on level 0 of one plane only, so no plane holds more than half of four.
        for labels, objectives in ((['A', 'X'], [0.1, 0.9, 0.1, 0.9]), (['B', 'Y'], [0.9, 0.1, 0.9, 0.1]),
                                   (['C', 'Z'], [0.5, 0.95, 0.5, 0.95]), (['D', 'W'], [0.95, 0.5, 0.95, 0.5])):
            count.place(_system(labels, objectives))
        assert count.plane_sizes() == [1, 1] and count.holds()

    def test_an_assembled_child_skips_its_first_mutation_once(self):
        system = SimpleNamespace()
        system.__dict__[PV.SKIP_FIRST_MUTATION] = True
        assert PV.skip_first_mutation(system)
        assert not PV.skip_first_mutation(system)


# ---------------------------------------------------------------------------
# The variable order.
# ---------------------------------------------------------------------------

class TestVariableOrder:

    @pytest.fixture(autouse=True)
    def _own_generator(self, monkeypatch):
        monkeypatch.setitem(PF._ORDER, 'rng', None)

    def test_a_seed_fixes_the_permutations(self):
        PF.start_variable_order(7)
        first = [PF._ORDER['rng'].permutation(4).tolist() for _ in range(5)]
        PF.start_variable_order(7)
        assert [PF._ORDER['rng'].permutation(4).tolist() for _ in range(5)] == first

    def test_no_seed_reads_the_global_state_without_drawing(self):
        np.random.seed(11)
        state = np.random.get_state()
        PF.start_variable_order(None)
        after = np.random.get_state()
        assert np.array_equal(state[1], after[1]) and state[2] == after[2]
        a = PF._ORDER['rng'].permutation(6).tolist()
        np.random.seed(11)
        PF.start_variable_order(None)
        assert PF._ORDER['rng'].permutation(6).tolist() == a


# ---------------------------------------------------------------------------
# End to end: the defaults on a two-equation system.
# ---------------------------------------------------------------------------

def test_plain_nsga2_runs_the_universal_search_on_a_system():
    np.random.seed(0)
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2', system_rps_mode='plain',
                             discrepancy_metric='wape', instability_metric='dyad',
                             equation_terms_max_number=4, equation_factors_max_number=1,
                             data_fun_pow=1, population_size=8, training_epochs=2)
    assert _resolved(search._config)[0] == 'per_variable'
    assert search.optimizer_exec_params['patience'] is None
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0, 4 * np.pi, 300)
    domain = search.createDomain(t, ID=0)[1]
    trajectory = search.createTrajectory({'u': np.sin(t), 'v': np.cos(t)}, domain, cache_id=0)[1]
    try:
        search.fit(data=[trajectory], max_deriv_order=(1,))
        assert len(search.pareto_history) == 2              # every epoch: no early stop by default
        front = search.optimizer.reported_front()
        assert front
        for system in front:
            for var in system.vars_to_describe:
                eq = system.vals[var]
                if eq.target is not None:
                    assert PF.is_plain_term(eq.target, var)
    finally:
        global_var.samples_manager.reset()
