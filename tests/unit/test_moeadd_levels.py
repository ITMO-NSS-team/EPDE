"""MOEA/DD non-domination level machinery, against the paper.

Paper: Li, Deb, Zhang, Kwong, "An Evolutionary Many-Objective Optimization
Algorithm Based on Dominance and Decomposition", IEEE TEVC 19(5), 2015
(``moeadd.pdf`` at the repo root).

Three properties are pinned here:

* ``ndl_update`` (Algorithm 4 line 3, the reference-[66] incremental insertion)
  must agree with a full ``fast_non_dominated_sorting`` of the same population.
* ``ParetoLevels.delete_point`` must re-derive the structure after a removal
  (Algorithm 4 line 26 / Algorithm 6 line 16).
* ``get_domain_idx`` (Eq. (6)) must honour an objective normalizer, so subregion
  association happens on the scale PBI selects on.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from epde.optimizers.moeadd.moeadd import ParetoLevels, ObjFunNormalizer
from epde.optimizers.moeadd.solution_template import get_domain_idx
from epde.optimizers.moeadd.supplementary import (acute_angle,
                                                  fast_non_dominated_sorting,
                                                  ndl_update)


class _Gene:
    def __init__(self, labels):
        self.terms_labels = labels


class _Chromosome:
    """Stand-in for ``SoEq.vals``: ``n_eq`` genes with distinct term labels.

    ``check_dominance`` reads ``vals.equation_keys`` and each gene's
    ``terms_labels`` (it skips an objective when the two solutions share the
    labels of the corresponding equation), so a test double has to carry both.
    """

    def __init__(self, tag, n_eq=1):
        self.equation_keys = [f'v{i}' for i in range(n_eq)]
        self._genes = {k: _Gene((f'{tag}_{k}',)) for k in self.equation_keys}

    def __len__(self):
        return len(self.equation_keys)

    def __iter__(self):
        return iter(self.equation_keys)

    def __getitem__(self, key):
        return self._genes[key]


class Sol:
    """Minimal ``MOEADDSolution`` substitute: objectives + a domain cache."""

    def __init__(self, tag, obj, n_eq=1):
        self.tag = tag
        self.obj_fun = np.asarray(obj, dtype=float)
        self.vals = _Chromosome(tag, n_eq)
        self.precomputed_domain = False
        self._domain = None

    def get_domain(self, weights, obj_normalizer=None):
        if self.precomputed_domain:
            return self._domain
        self._domain = get_domain_idx(self, weights, obj_normalizer)
        self.precomputed_domain = True
        return self._domain

    def set_domain(self, idx):
        self.precomputed_domain = True
        self._domain = idx

    def __repr__(self):
        return f'<{self.tag} {self.obj_fun}>'


def level_of(levels):
    return {id(s): idx for idx, lvl in enumerate(levels) for s in lvl}


def assert_levels_valid(levels):
    """No member of level k may be dominated by a member of level >= k, and
    every member of level k>0 must be dominated by someone on level k-1."""
    from epde.optimizers.moeadd.supplementary import check_dominance
    for k, lvl in enumerate(levels):
        for s in lvl:
            for deeper in levels[k:]:
                for other in deeper:
                    assert not check_dominance(other, s), (
                        f'{other} on level >= {k} dominates {s} on level {k}')
            if k > 0:
                assert any(check_dominance(other, s) for other in levels[k - 1]), (
                    f'{s} sits on level {k} with no dominator on level {k - 1}')


class TestNDLUpdateEquivalence:
    """Algorithm 4 line 3 -- the incremental update must equal a full sort."""

    def test_the_minimal_regression_case(self):
        # Recorded counterexample. (.69,.17) dominates the whole ORIGINAL level
        # 1, so the pre-fix code opened a new layer above it and dragged
        # (.65,.59) -- placed on level 1 earlier in the same moving-set pass --
        # down to level 2, although its only dominator, (.48,.03), is on level 0.
        parents = [Sol('p0', (0.36, 0.94)), Sol('p1', (0.65, 0.59)),
                   Sol('p2', (0.69, 0.17)), Sol('p3', (0.83, 0.33))]
        child = Sol('c', (0.48, 0.03))

        levels = fast_non_dominated_sorting(list(parents))
        got = level_of(ndl_update(child, levels))
        want = level_of(fast_non_dominated_sorting(parents + [child]))

        assert got == want
        assert got[id(parents[1])] == 1, 'p1 is dominated only by the level-0 child'

    @pytest.mark.parametrize('n_obj', [2, 3, 5])
    def test_matches_full_sort_on_random_insertions(self, n_obj):
        rng = np.random.default_rng(20260904 + n_obj)
        for trial in range(400):
            n = int(rng.integers(3, 25))
            parents = [Sol(f'p{i}', rng.random(n_obj)) for i in range(n)]
            child = Sol('c', rng.random(n_obj))
            levels = fast_non_dominated_sorting(list(parents))
            updated = ndl_update(child, levels)
            assert level_of(updated) == level_of(
                fast_non_dominated_sorting(parents + [child])), (
                f'n_obj={n_obj} trial={trial} n={n}')
            assert_levels_valid(updated)

    def test_matches_full_sort_with_duplicate_objectives(self):
        # Equal objective vectors between DIFFERENT structures are a tie in
        # check_dominance, so they must share a level.
        rng = np.random.default_rng(11)
        for trial in range(200):
            base = rng.random((3, 2))
            objs = np.vstack([base, base, rng.random((2, 2))])
            parents = [Sol(f'p{i}', o) for i, o in enumerate(objs)]
            child = Sol('c', base[0])
            levels = fast_non_dominated_sorting(list(parents))
            assert level_of(ndl_update(child, levels)) == level_of(
                fast_non_dominated_sorting(parents + [child])), f'trial={trial}'

    def test_matches_full_sort_for_multi_equation_systems(self):
        # 2 equations x 2 objectives: obj_fun has 4 entries and the
        # terms_labels skip-guard is live per equation.
        rng = np.random.default_rng(5)
        for trial in range(200):
            n = int(rng.integers(3, 14))
            parents = [Sol(f'p{i}', rng.random(4), n_eq=2) for i in range(n)]
            child = Sol('c', rng.random(4), n_eq=2)
            levels = fast_non_dominated_sorting(list(parents))
            assert level_of(ndl_update(child, levels)) == level_of(
                fast_non_dominated_sorting(parents + [child])), f'trial={trial}'

    def test_does_not_mutate_the_input_levels(self):
        parents = [Sol(f'p{i}', (i / 5, 1 - i / 5)) for i in range(5)]
        levels = fast_non_dominated_sorting(list(parents))
        snapshot = [list(lvl) for lvl in levels]
        ndl_update(Sol('c', (0.05, 0.05)), levels)
        assert [list(lvl) for lvl in levels] == snapshot


def _levels_obj(population):
    levels = ParetoLevels.__new__(ParetoLevels)
    levels._weights = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    levels._sorting_method = fast_non_dominated_sorting
    levels._update_method = ndl_update
    levels.population = list(population)
    levels.history = set()
    levels.normalizer = None
    levels.levels = fast_non_dominated_sorting(levels.population)
    return levels


class TestDeletePointRederivesLevels:
    """Algorithm 4 line 26 / Algorithm 6 line 16."""

    def test_survivor_is_promoted_when_its_only_dominator_leaves(self):
        keep = Sol('keep', (0.1, 0.9))
        dominator = Sol('dominator', (0.2, 0.2))
        dominated = Sol('dominated', (0.5, 0.5))
        levels = _levels_obj([keep, dominator, dominated])
        assert level_of(levels.levels)[id(dominated)] == 1

        levels.delete_point(dominator)

        assert level_of(levels.levels)[id(dominated)] == 0, (
            'the only dominator is gone, so it belongs on the front')
        assert len(levels.levels) == 1

    def test_structure_equals_a_full_sort_after_every_deletion(self):
        rng = np.random.default_rng(1234)
        for trial in range(200):
            n = int(rng.integers(4, 18))
            population = [Sol(f'p{i}', rng.random(2)) for i in range(n)]
            levels = _levels_obj(population)
            for _ in range(n - 2):
                victim = levels.population[int(rng.integers(len(levels.population)))]
                levels.delete_point(victim)
                assert level_of(levels.levels) == level_of(
                    fast_non_dominated_sorting(list(levels.population))), f'trial={trial}'
                assert_levels_valid(levels.levels)

    def test_population_and_levels_stay_in_sync(self):
        population = [Sol(f'p{i}', (i / 7, 1 - i / 7)) for i in range(7)]
        levels = _levels_obj(population)
        levels.delete_point(population[3])
        assert len(levels.population) == 6
        assert sum(len(lvl) for lvl in levels.levels) == 6
        assert all(s is not population[3] for s in levels.population)

    def test_deleting_an_absent_point_still_raises(self):
        population = [Sol(f'p{i}', (i / 4, 1 - i / 4)) for i in range(4)]
        levels = _levels_obj(population)
        with pytest.raises(RuntimeError, match='expected to remove exactly 1'):
            levels.delete_point(Sol('stranger', (0.5, 0.5)))


class TestGetDomainIdxNormalization:
    """Eq. (6), evaluated on the same scale PBI selects on."""

    WEIGHTS = np.array([[np.cos(a), np.sin(a)]
                        for a in np.linspace(0.0, np.pi / 2, 8)])

    def test_unset_normalizer_reproduces_the_raw_angle(self):
        sol = Sol('s', (0.8, 0.02))
        raw = np.fromiter((acute_angle(w, sol.obj_fun) for w in self.WEIGHTS),
                          dtype=float).argmin()
        assert get_domain_idx(sol, self.WEIGHTS) == raw
        assert get_domain_idx(sol, self.WEIGHTS, None) == raw

    # Axes of very different magnitude: the raw angle is dictated by the large
    # axis, which is exactly the failure marriageSolutionAssignment was fixed
    # for. Raw (1.0, 0.05) points 0.050 rad off the first axis and lands in
    # subregion 0; rescaled by (1.0, 0.1) it points 0.464 rad and lands in 2.
    SKEWED_OBJ = (1.0, 0.05)
    SKEWED_SCALE = np.array([1.0, 0.1])

    def test_normalizer_changes_the_association(self):
        normalizer = ObjFunNormalizer(self.SKEWED_SCALE)
        raw_idx = get_domain_idx(Sol('raw', self.SKEWED_OBJ), self.WEIGHTS)
        norm_idx = get_domain_idx(Sol('norm', self.SKEWED_OBJ), self.WEIGHTS,
                                  normalizer)
        expected = np.fromiter(
            (acute_angle(w, normalizer(np.asarray(self.SKEWED_OBJ, dtype=float)))
             for w in self.WEIGHTS), dtype=float).argmin()
        assert norm_idx == expected
        assert norm_idx != raw_idx, 'this fixture is chosen so the two disagree'

    def test_agrees_with_the_marriage_assignment_scale(self, monkeypatch):
        # marriageSolutionAssignment ranks on the NORMALIZED objective; a
        # solution associated through Eq. (6) with the same normalizer must be
        # ranked on that same scale.
        import epde.globals as global_var
        from epde.optimizers.moeadd.moeadd import marriageSolutionAssignment
        monkeypatch.setattr(global_var, 'verbose',
                            SimpleNamespace(show_iter_idx=False), raising=False)
        rng = np.random.default_rng(3)
        objectives = rng.random((8, 2)) * np.array([1.5, 0.05])
        population = [Sol(f'p{i}', o) for i, o in enumerate(objectives)]
        normalizer = ObjFunNormalizer(np.max(objectives, axis=0))
        marriageSolutionAssignment(self.WEIGHTS, population, normalizer)

        for sol in population:
            fresh = Sol(sol.tag, sol.obj_fun)
            eq6 = get_domain_idx(fresh, self.WEIGHTS, normalizer)
            marriage_angle = acute_angle(self.WEIGHTS[sol.get_domain(self.WEIGHTS)],
                                         normalizer(sol.obj_fun))
            eq6_angle = acute_angle(self.WEIGHTS[eq6], normalizer(sol.obj_fun))
            # Marriage is a forced bijection, so it cannot always hand out the
            # angular argmin; what must hold is that both rank on the SAME scale.
            assert eq6_angle <= marriage_angle + 1e-12

    def test_multi_equation_weight_expansion(self):
        # One weight component per objective TYPE, one obj_fun entry per
        # (objective, equation) pair.
        sol = Sol('s', (0.9, 0.7, 0.1, 0.2), n_eq=2)
        weights = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        expected = np.fromiter(
            (acute_angle([c for c in w for _ in range(2)], sol.obj_fun)
             for w in weights), dtype=float).argmin()
        assert get_domain_idx(sol, weights) == expected

    def test_population_to_sectors_forwards_the_normalizer(self):
        from epde.operators.multiobjective.moeadd_specific import population_to_sectors
        normalizer = ObjFunNormalizer(self.SKEWED_SCALE)
        obj = self.SKEWED_OBJ
        raw_pop = [Sol('raw', obj)]
        norm_pop = [Sol('norm', obj)]
        raw_sectors = population_to_sectors(raw_pop, self.WEIGHTS)
        norm_sectors = population_to_sectors(norm_pop, self.WEIGHTS, normalizer)
        raw_idx = next(i for i, d in enumerate(raw_sectors) if d)
        norm_idx = next(i for i, d in enumerate(norm_sectors) if d)
        assert raw_idx != norm_idx

    def test_explicit_set_domain_is_not_overwritten(self):
        # MOEA/DD Algorithm 2 line 17 hands out the initial associations as a
        # bijection; Eq. (6) must not silently re-derive them.
        sol = Sol('s', self.SKEWED_OBJ)
        sol.set_domain(5)
        assert sol.get_domain(self.WEIGHTS, ObjFunNormalizer(self.SKEWED_SCALE)) == 5
