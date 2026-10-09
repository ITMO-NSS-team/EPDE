"""Single-objective ranking and parent selection on edge-case objectives.

Two defects of the single-objective path, pinned on stub populations:

* B20 -- ``RouletteWheelSelection`` drew parents with ``p = (1/v) / sum(1/v)``.
  An exact-0 objective (the true equation under six of the eight instability
  metrics on exact data, or WAPE on an exact fit) made ``1/0 = inf``, so ``p``
  held NaN and ``np.random.choice`` aborted the search at the first selection.
* N3 -- the failure stamp ``LOSS_NAN_VAL = 1e7`` is "worst" only for bounded
  metrics. vcoef / cv / chi2 / chi2_centered / het_raw can score a legitimate
  wrong form above 1e7, so ranking by the raw value let a stamped (failed)
  form outrank a real one in ``simple_sorting`` (and so in ``SizeRestriction``
  and ``FractionElitism``) and take the largest roulette share. A stamped form
  is detected the way MOEA/D detects it -- some equation's discrepancy
  ``fitness_value >= LOSS_NAN_VAL`` -- and must rank last whatever the metric.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import epde.globals as global_var
from epde.operators.common.objectives import LOSS_NAN_VAL
from epde.operators.singleobjective.selections import RouletteWheelSelection
from epde.operators.singleobjective.so_specific import (FractionElitism,
                                                       SizeRestriction)
from epde.optimizers.moeadd.solution_template import MOEADDSolution
from epde.optimizers.single_criterion.optimizer import Population
from epde.optimizers.single_criterion.supplementary import simple_sorting


class _Stub(MOEADDSolution):
    """A real ``MOEADDSolution`` with a fixed objective vector.

    ``vals`` holds one gene per equation carrying what the single-objective
    consumers read: ``fitness_value`` (the discrepancy, where the fitness host
    writes its failure stamp) and ``fitness_calculated``.
    """

    def __init__(self, tag, obj, fitness_values=(0.5,)):
        genes = [SimpleNamespace(tag=f'{tag}_{i}', fitness_value=fv,
                                 fitness_calculated=True)
                 for i, fv in enumerate(fitness_values)]
        super().__init__(genes, obj_funs=[])
        self.tag = tag
        self._obj_fun = np.asarray(obj, dtype=float).reshape(-1)
        self.precomputed_value = True

    def __hash__(self):
        return id(self)

    def __iter__(self):
        # ``SoEq`` iterates over its equations (SizeRestriction logs the best
        # system's discrepancy that way).
        return iter(self.vals)

    def __repr__(self):
        return f'<{self.tag} {self._obj_fun}>'


def _stamped(tag):
    """A form the fitness host stamped: every stamped objective is 1e7."""
    return _Stub(tag, [LOSS_NAN_VAL], fitness_values=(LOSS_NAN_VAL,))


def _roulette_p(population, monkeypatch, seed=0):
    """Run the production roulette and capture the ``p`` it hands to numpy."""
    captured = {}
    original = np.random.choice

    def spy(*args, **kwargs):
        captured['p'] = np.asarray(kwargs['p'], dtype=float)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.random, 'choice', spy)
    op = RouletteWheelSelection(['parents_fraction'])
    op.params = {'parents_fraction': 0.4}
    np.random.seed(seed)
    op.apply(Population(population), {})
    return captured['p']


def _assert_distribution(p, n):
    assert p.shape == (n,)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0)
    assert np.isclose(p.sum(), 1.0)


# --------------------------------------------------------------------------
# B20: an exact-0 objective in the roulette
# --------------------------------------------------------------------------
class TestRouletteZeroObjective:
    def test_exact_zero_objective_selects_without_raising(self, monkeypatch):
        pop = [_Stub('truth', [0.0]), _Stub('a', [1e-3]), _Stub('b', [0.1]),
               _Stub('c', [1.0])]
        p = _roulette_p(pop, monkeypatch)
        _assert_distribution(p, len(pop))
        # The exact 0 is the best objective, so it keeps the largest share --
        # as a 1e-21 truth already does under the 1/v rule -- while the
        # positive candidates keep a non-zero chance.
        assert p[0] == p.max()
        assert np.all(p[1:] > 0)
        assert sum(s.crossover_times() for s in pop) > 0

    def test_tied_exact_zeros_share_equally(self, monkeypatch):
        pop = [_Stub('t1', [0.0]), _Stub('a', [0.2]), _Stub('t2', [0.0])]
        p = _roulette_p(pop, monkeypatch)
        _assert_distribution(p, len(pop))
        assert p[0] == p[2]
        assert p[0] > p[1]

    def test_all_zero_population_is_uniform(self, monkeypatch):
        pop = [_Stub(f's{i}', [0.0]) for i in range(4)]
        p = _roulette_p(pop, monkeypatch)
        _assert_distribution(p, len(pop))
        assert np.allclose(p, 0.25)

    def test_floor_is_scale_invariant(self, monkeypatch):
        values = [0.0, 3e-4, 0.07, 2.0]
        p1 = _roulette_p([_Stub(f's{i}', [v]) for i, v in enumerate(values)],
                         monkeypatch)
        p2 = _roulette_p([_Stub(f's{i}', [1e6 * v]) for i, v in enumerate(values)],
                         monkeypatch)
        assert np.allclose(p1, p2, rtol=1e-12, atol=0.0)

    def test_all_infinite_objectives_are_uniform(self, monkeypatch):
        # 1/inf = 0 for every candidate used to normalise 0/0 into NaN.
        pop = [_Stub(f's{i}', [np.inf]) for i in range(3)]
        p = _roulette_p(pop, monkeypatch)
        _assert_distribution(p, len(pop))
        assert np.allclose(p, 1 / 3)

    def test_near_underflow_population_stays_finite(self, monkeypatch):
        pop = [_Stub('zero', [0.0]), _Stub('tiny', [1e-310]), _Stub('one', [1.0])]
        p = _roulette_p(pop, monkeypatch)
        _assert_distribution(p, len(pop))
        assert p[0] == p.max()

    def test_nan_objective_is_still_refused(self, monkeypatch):
        # The floor is for exact zeros only; a NaN is a broken score and must
        # keep failing loudly rather than be smoothed into a uniform draw.
        pop = [_Stub('nan', [np.nan]), _Stub('ok', [1.0])]
        with pytest.raises(ValueError, match='NaN'):
            _roulette_p(pop, monkeypatch)

    def test_positive_population_probabilities_unchanged(self, monkeypatch):
        # The floor must not touch an all-positive population: p stays
        # bit-identical to the former (1/v) / sum(1/v), so the parent draws of
        # every run that never hit a zero are reproduced exactly.
        values = np.array([3e-21, 1e-3, 0.4, 7.5, 1.2e8])
        p = _roulette_p([_Stub(f's{i}', [v]) for i, v in enumerate(values)],
                        monkeypatch)
        legacy = np.array([1 / np.array([v]) for v in values])
        legacy = (legacy / np.sum(legacy)).squeeze()
        assert np.array_equal(p, legacy)


# --------------------------------------------------------------------------
# N3: a stamped (failed) form must rank last whatever the metric
# --------------------------------------------------------------------------
class TestStampedFormsRankLast:
    def _population(self):
        # A legitimate wrong form above 1e7 (reachable under vcoef / chi2 /
        # het_raw with sparsity_cls='knee'), a stamped form, and two others.
        return [_Stub('big_legit', [5e7]), _stamped('stamped'),
                _Stub('good', [0.01]), _Stub('mid', [3.0])]

    def test_simple_sorting_puts_stamped_last(self):
        order = [s.tag for s in simple_sorting(self._population())]
        assert order == ['good', 'mid', 'big_legit', 'stamped']

    def test_stamped_ranks_after_an_infinite_legit_objective(self):
        pop = [_stamped('stamped'), _Stub('inf_legit', [np.inf]), _Stub('ok', [1.0])]
        assert [s.tag for s in simple_sorting(pop)] == ['ok', 'inf_legit', 'stamped']

    def test_any_stamped_equation_stamps_the_system(self):
        # Two-equation system: only its second equation failed, so its summed
        # objective is small, but the system as a whole is stamped.
        coupled = _Stub('coupled', [0.1, 0.2], fitness_values=(0.1, LOSS_NAN_VAL))
        legit = _Stub('legit', [4e7, 4e7], fitness_values=(0.3, 0.4))
        assert [s.tag for s in simple_sorting([coupled, legit])] == ['legit', 'coupled']

    def test_ranking_without_stamps_is_unchanged(self):
        pop = [_Stub('c', [2.0]), _Stub('a', [0.5]), _Stub('b', [0.5]), _Stub('d', [9e9])]
        assert [s.tag for s in simple_sorting(pop)] == ['a', 'b', 'c', 'd']

    def test_size_restriction_drops_the_stamped_form(self, monkeypatch):
        monkeypatch.setattr(global_var, 'history', global_var.TrainHistory(),
                            raising=False)
        population = Population(self._population())
        population.length = 3
        SizeRestriction().apply(population, {})
        assert sorted(s.tag for s in population.population) == ['big_legit', 'good', 'mid']

    def test_fraction_elitism_never_crowns_a_stamped_form(self):
        pop = [_stamped('stamped'), _Stub('big_legit', [5e7])]
        population = Population(pop)
        FractionElitism().apply(population, {})
        elite = [s.tag for s in population.population if s.elite == 'immutable']
        assert elite == ['big_legit']

    def test_roulette_gives_a_stamped_form_zero_probability(self, monkeypatch):
        pop = self._population()
        p = _roulette_p(pop, monkeypatch)
        _assert_distribution(p, len(pop))
        assert p[1] == 0.0
        assert np.all(p[[0, 2, 3]] > 0)
        assert pop[1].crossover_times() == 0

    def test_roulette_stamped_with_exact_zero_elsewhere(self, monkeypatch):
        pop = [_stamped('stamped'), _Stub('truth', [0.0]), _Stub('other', [0.3])]
        p = _roulette_p(pop, monkeypatch)
        _assert_distribution(p, len(pop))
        assert p[0] == 0.0

    def test_roulette_all_stamped_is_uniform(self, monkeypatch):
        pop = [_stamped(f's{i}') for i in range(3)]
        p = _roulette_p(pop, monkeypatch)
        _assert_distribution(p, len(pop))
        assert np.allclose(p, 1 / 3)
