"""Unit tests for the MOEA/DD PBI/niching layer and degeneracy helpers.

First direct unit coverage for epde/operators/multiobjective/: pins the
pure-math layer (penalty_based_intersection, crowding, worst-finders),
the LOSS_NAN_VAL degeneracy reads, SimpleNeighborSelector's clamp,
MetaparameterMutation's reflected Gaussian, and TermParamCrossover's
dim-fallback semantics -- all with lightweight stubs, no SoEq/pool
construction.
"""
import numpy as np
import pytest

from epde.operators.common.objectives import LOSS_NAN_VAL
from epde.operators.multiobjective.moeadd_specific import (
    penalty_based_intersection,
    population_to_sectors,
    _most_crowded_domain,
    _worst_by_pbi,
    decomposition_based_worst,
    locate_pareto_worst,
    _equation_is_degenerate,
    has_degenerate_equation,
    SimpleNeighborSelector,
)
from epde.operators.multiobjective.mutations import MetaparameterMutation
from epde.operators.multiobjective.variation import TermParamCrossover


class StubSolution:
    """Solution stub exposing exactly what the PBI layer reads."""

    def __init__(self, obj_fun, domain=None, n_eqs=1):
        self.obj_fun = np.asarray(obj_fun, dtype=float)
        self._domain = domain
        self.vals = [object()] * n_eqs

    def get_domain(self, weights):
        return self._domain


class StubLevels:
    def __init__(self, population, levels, normalizer=None):
        self.population = population
        self.levels = levels
        self.normalizer = normalizer


class TestPenaltyBasedIntersection:

    def test_legacy_full_space_branch(self):
        # n_eqs=2 with weight.size == n_obj: weight.size * n_eqs != n_obj,
        # so the weight is treated as already living in the full space.
        sol = StubSolution([3.0, 4.0], n_eqs=2)
        pbi = penalty_based_intersection(sol, [1.0, 0.0], [0.0, 0.0],
                                         penalty_factor=1.0)
        # d1 = <[3,4],[1,0]> / 1 = 3; d2 = ||[3,4] - [3,0]|| = 4.
        assert pbi == pytest.approx(7.0)

    def test_per_objective_type_expansion(self):
        # weight.size * n_eqs == n_obj: the MOEA/D weight is per objective
        # TYPE and must be repeated per equation.
        sol = StubSolution([3.0, 4.0, 1.0, 2.0], n_eqs=2)
        pbi = penalty_based_intersection(sol, [1.0, 0.0], [0.0, 0.0],
                                         penalty_factor=1.0)
        ref = StubSolution([3.0, 4.0, 1.0, 2.0], n_eqs=1)
        expected = penalty_based_intersection(
            ref, [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
            penalty_factor=1.0)
        assert pbi == pytest.approx(expected)

    def test_penalty_factor_scales_d2(self):
        sol = StubSolution([3.0, 4.0], n_eqs=2)
        p1 = penalty_based_intersection(sol, [1.0, 0.0], [0.0, 0.0], 1.0)
        p2 = penalty_based_intersection(sol, [1.0, 0.0], [0.0, 0.0], 2.0)
        assert p2 - p1 == pytest.approx(4.0)   # + penalty * d2 with d2 = 4

    def test_normalizer_applied_to_solution_and_ideal(self):
        sol = StubSolution([3.0, 4.0], n_eqs=2)
        half = lambda arr: np.asarray(arr, dtype=float) / 2.0
        pbi = penalty_based_intersection(sol, [1.0, 0.0], [0.0, 0.0], 1.0,
                                         obj_normalizer=half)
        assert pbi == pytest.approx(3.5)       # halved objective, zero ideal


class TestCrowdingAndWorstFinders:

    def _sectors(self):
        # Sector 0: one small solution; sectors 1 and 2: tied niche count,
        # sector 2 carrying the larger PBI sum.
        s0 = [StubSolution([1.0, 1.0], domain=0)]
        s1 = [StubSolution([1.0, 0.0], domain=1),
              StubSolution([2.0, 0.0], domain=1)]
        s2 = [StubSolution([5.0, 0.0], domain=2),
              StubSolution([6.0, 0.0], domain=2)]
        weights = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        return [s0, s1, s2], weights

    def test_most_crowded_tie_broken_by_pbi_sum(self):
        sectors, weights = self._sectors()
        idx = _most_crowded_domain(sectors, weights, np.zeros(2), 1.0, None)
        assert idx == 2

    def test_candidate_idxs_restricts_search(self):
        sectors, weights = self._sectors()
        idx = _most_crowded_domain(sectors, weights, np.zeros(2), 1.0, None,
                                   candidate_idxs=[0])
        assert idx == 0

    def test_worst_by_pbi_argmax_and_first_max_tie_break(self):
        lo = StubSolution([1.0, 0.0])
        hi_a = StubSolution([6.0, 0.0])
        hi_b = StubSolution([6.0, 0.0])
        worst = _worst_by_pbi([lo, hi_a, hi_b], np.array([1.0, 0.0]),
                              np.zeros(2), 1.0, None)
        assert worst is hi_a                   # np.argmax keeps the first max

    def test_decomposition_based_worst(self):
        sectors, weights = self._sectors()
        population = [s for sec in sectors for s in sec]
        worst = decomposition_based_worst(population, weights, np.zeros(2),
                                          1.0, None)
        assert worst is sectors[2][1]          # max PBI in most crowded sector

    def test_locate_pareto_worst_prefers_last_front(self):
        sectors, weights = self._sectors()
        population = [s for sec in sectors for s in sec]
        # Put the SMALLER of sector 2's solutions on the last front: the
        # NDL filter must beat the raw PBI argmax.
        levels = StubLevels(population,
                            levels=[[s for s in population if s is not sectors[2][0]],
                                    [sectors[2][0]]])
        worst = locate_pareto_worst(levels, weights, np.zeros(2), 1.0)
        assert worst is sectors[2][0]

    def test_locate_pareto_worst_desync_raises(self):
        sectors, weights = self._sectors()
        population = [s for sec in sectors for s in sec]
        levels = StubLevels(population, levels=[[]])   # nothing on any level
        with pytest.raises(RuntimeError, match='out of sync'):
            locate_pareto_worst(levels, weights, np.zeros(2), 1.0)


class TestDegeneracyHelpers:

    class _Eq:
        def __init__(self, fv):
            if fv is not None:
                self.fitness_value = fv

    class _Offspring:
        def __init__(self, eqs):
            self.vals = eqs

    def test_equation_is_degenerate_threshold(self):
        assert _equation_is_degenerate(self._Eq(LOSS_NAN_VAL))
        assert _equation_is_degenerate(self._Eq(LOSS_NAN_VAL * 10))
        assert not _equation_is_degenerate(self._Eq(LOSS_NAN_VAL * 0.99))
        assert not _equation_is_degenerate(self._Eq(None))   # unfitted: no attr

    def test_has_degenerate_equation_any_semantics(self):
        good, bad = self._Eq(0.5), self._Eq(LOSS_NAN_VAL)
        assert has_degenerate_equation(self._Offspring([good, bad]))
        assert not has_degenerate_equation(self._Offspring([good, good]))


class TestSimpleNeighborSelector:

    def test_clamps_to_available_neighbors(self):
        op = SimpleNeighborSelector(['number_of_neighbors'])
        op.params = {'number_of_neighbors': 4}
        np.random.seed(0)
        pool = ['a', 'b']
        picked = op.apply(pool, arguments={})
        assert sorted(picked) == pool          # size clamped to len(pool)

    def test_subset_without_replacement(self):
        op = SimpleNeighborSelector(['number_of_neighbors'])
        op.params = {'number_of_neighbors': 2}
        np.random.seed(1)
        picked = op.apply(list('abcde'), arguments={})
        assert len(picked) == 2 and len(set(picked)) == 2


class TestMetaparameterMutation:

    def _op(self):
        op = MetaparameterMutation(['std', 'mean'])
        op.params = {'mean': 0.0, 'std': 1.0}
        return op

    def test_matches_relative_gaussian(self):
        op = self._op()
        np.random.seed(7)
        result = op.apply(2.0, arguments={})
        np.random.seed(7)
        expected = 2.0 + np.random.normal(0.0, 1.0 * 2.0)
        if expected < 0:
            expected = -expected
        assert result == pytest.approx(expected)
        assert isinstance(result, np.float64)

    def test_reflects_negative_results_at_zero(self):
        op = self._op()
        # Find a seed whose raw draw sends the value negative.
        for seed in range(200):
            np.random.seed(seed)
            raw = 1.0 + np.random.normal(0.0, 1.0)
            if raw < 0:
                np.random.seed(seed)
                assert op.apply(1.0, arguments={}) == pytest.approx(-raw)
                break
        else:
            pytest.fail('no seed produced a negative draw')


class StubFactor:
    def __init__(self, label, names_values):
        self.label = label
        self.params = np.array([v for _, v in names_values], dtype=float)
        self.params_description = {i: {'name': n}
                                   for i, (n, _) in enumerate(names_values)}

    def set_param(self, value, idx):
        self.params[idx] = value


class StubTerm:
    def __init__(self, factors):
        self.structure = factors

    def reset_saved_state(self):
        pass

    def reset_occupied_tokens(self):
        pass


class TestTermParamCrossoverDimFallback:

    def _op(self, proportion=0.4):
        op = TermParamCrossover(['term_param_proportion'])
        op.params = {'term_param_proportion': proportion}
        return op

    def test_dim_falls_back_to_power_and_blends_free_param(self):
        # Single token without 'dim': fallback makes dim_param_idx == power
        # index, so only the free 'freq' param is blended.
        t1 = StubTerm([StubFactor('sin', [('power', 1.0), ('freq', 2.0)])])
        t2 = StubTerm([StubFactor('sin', [('power', 1.0), ('freq', 6.0)])])
        self._op().apply((t1, t2), arguments={})
        assert t1.structure[0].params[1] == pytest.approx(2.0 + 0.4 * 4.0)
        assert t2.structure[0].params[1] == pytest.approx(2.0 + 0.6 * 4.0)
        assert t1.structure[0].params[0] == 1.0    # power untouched
        # Blend invariant: v1' + v2' == v1 + v2.
        assert (t1.structure[0].params[1] + t2.structure[0].params[1]
                == pytest.approx(8.0))

    def test_dim_index_carries_across_tokens(self):
        # Token A carries 'dim' at index 1; token B has no 'dim', and the
        # stale index 1 from token A masks B's index-1 param from blending
        # (load-bearing legacy semantics pinned by the sentinel rewrite).
        t1 = StubTerm([
            StubFactor('du/dx', [('power', 1.0), ('dim', 0.0), ('freq', 2.0)]),
            StubFactor('sin',   [('power', 1.0), ('freq', 10.0)]),
        ])
        t2 = StubTerm([
            StubFactor('du/dx', [('power', 1.0), ('dim', 0.0), ('freq', 6.0)]),
            StubFactor('sin',   [('power', 1.0), ('freq', 30.0)]),
        ])
        self._op().apply((t1, t2), arguments={})
        # Token A: only 'freq' (index 2) blends.
        assert t1.structure[0].params[2] == pytest.approx(2.0 + 0.4 * 4.0)
        # Token B: index 1 ('freq') is masked by the stale dim index.
        assert t1.structure[1].params[1] == 10.0
        assert t2.structure[1].params[1] == 30.0

    def test_unequal_factor_counts_raise(self):
        t1 = StubTerm([StubFactor('u', [('power', 1.0)])])
        t2 = StubTerm([StubFactor('u', [('power', 1.0)]),
                       StubFactor('v', [('power', 1.0)])])
        with pytest.raises(Exception, match='Wrong terms'):
            self._op().apply((t1, t2), arguments={})
