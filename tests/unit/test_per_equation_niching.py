#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``evolution.niching='per_equation'`` and ``complementary_mating_prob``.

Each equation of a system is ranked in its own (discrepancy, instability)
plane. Pinned here, on hand-built systems:

* the plane layout read from ``SoEq.obj_fun`` (grouped by type, then equation);
* plane ranks and fronts follow the same-structure rule (one structure, one
  equation: never compared, one holder);
* survival protects the systems holding an equation's front and evicts, among
  the rest, the system whose best per-equation rank is the worst -- the
  exact-u-but-junk-v system survives; with everything protected, the system on
  the fewest fronts and most crowded in its plane goes;
* the population updater uses this only when the setting is on and the system
  has more than one equation;
* mating domains are taken in the served equation's plane, and the partner
  draw pairs complementary front holders without touching the random stream
  when it is off.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.interface.search_config import load_search_config, resolve_niching
from epde.operators.multiobjective import moeadd_specific as MS
from epde.operators.multiobjective import selections as SEL

from conftest import using_config


class _Vals:
    def __init__(self, labels):
        self.equation_keys = list(labels)
        self._eqs = {k: SimpleNamespace(terms_labels=frozenset({lab})) for k, lab in labels.items()}

    def __len__(self):
        return len(self.equation_keys)

    def __getitem__(self, key):
        return self._eqs[key]


def _system(labels, disc, inst):
    """A stand-in system: ``labels`` {var: structure}, per-equation values in
    var order; obj_fun laid out as SoEq does (all discrepancies, then all
    instabilities)."""
    return SimpleNamespace(vals=_Vals(labels), obj_fun=np.array(list(disc) + list(inst), float))


def _levels(population, levels=None):
    return SimpleNamespace(population=population, levels=levels or [population])


# Three-equation fixture in the spirit of the Lorenz split: S1 exact u, S2 exact
# v, S3 exact w, S4 mediocre everywhere, S5 worse than S4 everywhere.
S1 = _system({'u': 'U*', 'v': 'v1', 'w': 'w1'}, (0.01, 0.5, 0.4), (0.01, 0.9, 0.8))
S2 = _system({'u': 'u2', 'v': 'V*', 'w': 'w2'}, (0.3, 0.02, 0.6), (0.7, 0.02, 0.5))
S3 = _system({'u': 'u3', 'v': 'v3', 'w': 'W*'}, (0.3, 0.4, 0.01), (0.6, 0.4, 0.01))
S4 = _system({'u': 'u4', 'v': 'v4', 'w': 'w4'}, (0.2, 0.2, 0.2), (0.3, 0.3, 0.3))
S5 = _system({'u': 'u5', 'v': 'v5', 'w': 'w5'}, (0.25, 0.25, 0.25), (0.35, 0.35, 0.35))


class TestThePlanes:

    def test_the_layout_is_type_then_equation(self):
        planes, labels = MS.equation_planes([S1, S2])
        np.testing.assert_array_equal(planes[1][0], [0.5, 0.9])     # S1's v plane
        np.testing.assert_array_equal(planes[2][1], [0.6, 0.5])     # S2's w plane
        assert labels[0] == [frozenset({'U*'}), frozenset({'u2'})]

    def test_plane_levels(self):
        points = [[1, 1], [2, 2], [0.5, 3], [3, 3]]
        assert MS.plane_levels(points, ['a', 'b', 'c', 'd']).tolist() == [0, 1, 0, 2]

    def test_the_same_structure_is_not_ranked_against_itself(self):
        points = [[1, 1], [2, 2]]
        assert MS.plane_levels(points, ['a', 'a']).tolist() == [0, 0]

    def test_one_holder_per_structure(self):
        twin_better = _system({'u': 'U*', 'v': 'x'}, (0.01, 0.9), (0.01, 0.9))
        twin_worse = _system({'u': 'U*', 'v': 'y'}, (0.02, 0.8), (0.02, 0.8))
        holders = MS.equation_front_holders([twin_worse, twin_better],
                                            system_level=lambda i: (1, 0)[i])
        # Both are on u's front (same structure); the shallower system holds it.
        assert holders[0] == {1}


class TestTheSurvival:

    def test_each_exact_equation_is_protected(self):
        holders = MS.equation_front_holders([S1, S2, S3, S4, S5])
        protected = set().union(*holders)
        assert {0, 1, 2} <= protected and 4 not in protected

    def test_the_mediocre_system_goes_before_the_specialists(self):
        victim = MS.per_equation_worst(_levels([S1, S2, S3, S4, S5]))
        assert victim is S5

    def test_ranks_decide_not_error_sums(self):
        # A holds u's front, B holds v's. Unprotected: K is the runner-up in u
        # (rank 1) and the worst in v; D is the runner-up in v; N is third in
        # both (sorted ranks (2, 2)). An error-SUM rule evicts K (sum 10.04 vs
        # N's 2.0); the rank rule keeps both runners-up and evicts N.
        A = _system({'u': 'A', 'v': 'a'}, (0.01, 0.9), (0.01, 0.9))
        B = _system({'u': 'b', 'v': 'B'}, (0.9, 0.01), (0.9, 0.01))
        D = _system({'u': 'd', 'v': 'dv'}, (0.95, 0.2), (0.95, 0.2))
        K = _system({'u': 'k', 'v': 'kv'}, (0.02, 5.0), (0.02, 5.0))
        N = _system({'u': 'n', 'v': 'nv'}, (0.5, 0.5), (0.5, 0.5))
        assert MS.per_equation_worst(_levels([A, B, D, K, N])) is N

    def test_a_clone_of_a_held_structure_goes_first(self):
        S1_clone = _system({'u': 'U*', 'v': 'v9', 'w': 'w9'}, (0.02, 0.6, 0.6), (0.02, 0.9, 0.9))
        victim = MS.per_equation_worst(_levels([S1, S2, S3, S4, S1_clone]))
        assert victim is S1_clone

    def test_the_updater_branch_deletes_the_per_equation_victim(self):
        from epde.optimizers.moeadd.moeadd import ParetoLevels
        population = [S1, S2, S3, S4]
        deleted = []

        class Levels:
            def __init__(self):
                self.population = list(population)
                self.levels = [list(population)]
                self.normalizer = None

            def update(self, point):
                self.population.append(point)
                self.levels[-1].append(point)

            def delete_point(self, point):
                deleted.append(point)

        updater = MS.PopulationUpdater(['PBI_penalty'])
        updater.params = {'PBI_penalty': 5.0}
        with using_config(niching='per_equation'):
            updater.apply((S5, Levels()), {'weights': np.eye(2), 'best_obj': np.zeros(2)})
        assert deleted == [S5]

    def test_with_everything_protected_the_fewest_fronts_goes(self):
        # Two u-specialists on u's front only, both at its ends -> crowding inf;
        # one middle point on u's front is the most crowded.
        a = _system({'u': 'a', 'v': 'V*'}, (0.1, 0.01), (0.9, 0.01))
        b = _system({'u': 'b', 'v': 'vb'}, (0.5, 0.5), (0.5, 0.5))
        c = _system({'u': 'c', 'v': 'vc'}, (0.9, 0.6), (0.1, 0.6))
        victim = MS.per_equation_worst(_levels([a, b, c]))
        assert victim is b

    def test_the_updater_uses_it_only_when_on_and_multi_equation(self):
        assert not MS.niching_per_equation([S1, S2])
        with using_config(niching='per_equation'):
            assert MS.niching_per_equation([S1, S2])
            single = _system({'u': 'a'}, (1.0,), (1.0,))
            assert not MS.niching_per_equation([single])


class TestTheMating:

    def test_domains_are_taken_in_the_served_equations_plane(self):
        weights = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        # u plane lies on the discrepancy axis, v on the diagonal, w on the instability axis.
        system = _system({'u': 'a', 'v': 'b', 'w': 'c'}, (1.0, 1.0, 0.0), (0.0, 1.0, 1.0))
        assert [SEL._plane_domains([system], e, weights, None)[0] for e in range(3)] == [0, 1, 2]

    def test_the_partner_is_complementary(self):
        population = [S1, S2, S3, S4, S5]
        np.random.seed(0)
        with using_config(complementary_mating_prob=1.0):
            for _ in range(20):
                idx, pair = SEL._complementary_partner(population, np.array([0, 4]))
                assert int(idx[0]) == 0 and int(idx[1]) in (1, 2)
                assert pair == (0, int(idx[1]))

    def test_off_draws_nothing(self):
        np.random.seed(5)
        before = np.random.get_state()[1].copy()
        out, pair = SEL._complementary_partner([S1, S2], np.array([0, 1]))
        assert out.tolist() == [0, 1] and pair is None
        assert np.array_equal(np.random.get_state()[1], before)

    def test_the_crossover_pairs_the_forced_mates(self):
        from epde.operators.multiobjective.variation import ParetoLevelsCrossover

        class Sol:
            def __init__(self, name):
                self.name = name
                self.times = 1

            def crossover_times(self):
                return self.times

            def reset_counter(self):
                self.times = 0

        sols = [Sol(n) for n in 'abcd']
        levels = SimpleNamespace(population=sols, forced_pair=(sols[0], sols[3]),
                                 unplaced_candidates=[])
        pairs = []

        class Recorder:
            def apply(self, objective, arguments):
                pairs.append((objective[0].name, objective[1].name))
                return objective

        crossover = ParetoLevelsCrossover()
        crossover._suboperators = {'chromosome_crossover': Recorder()}
        crossover.parse_suboperator_args = lambda arguments: ({}, {'chromosome_crossover': {}})
        import copy as _copy
        original = _copy.deepcopy
        try:
            np.random.seed(1)
            # Bypass the deepcopy/reset_state/assert tail: only the pairing matters here.
            for sol in sols:
                sol.reset_state = lambda *a, **k: None
                sol.vals = SimpleNamespace(equation_keys=[])
            crossover.apply(levels, {})
        finally:
            _copy.deepcopy = original
        assert pairs[0] == ('a', 'd') and levels.forced_pair is None


class TestTheSettings:

    def test_defaults(self):
        cfg = load_search_config()
        evo = cfg.evolution
        assert evo.niching == 'auto' and evo.complementary_mating_prob == 0.0
        # 'auto' is 'system' everywhere except plain-mode NSGA-II.
        assert resolve_niching(evo.niching, cfg.search_space.system_rps_mode,
                               evo.engine) == 'system'

    @pytest.mark.parametrize('overrides', [{'niching': 'islands'},
                                           {'complementary_mating_prob': 1.5},
                                           {'complementary_mating_prob': True}])
    def test_bad_values_are_rejected(self, overrides):
        with pytest.raises(ValueError):
            load_search_config(overrides=overrides)
