"""The two operators ``evolution.engine='nsga2'`` swaps into the MOEA/DD chain:
the mating selection and the level updater. Everything else -- fitness,
right-part selection, crossover, mutation, the initial sorting -- is shared."""
import numpy as np

from epde.operators.utils.template import CompoundOperator


def plane_crowding(systems) -> np.ndarray:
    """NSGA-II crowding distance of ``systems`` (one level), per equation: in
    each equation's own (discrepancy, instability) plane, then the smallest
    over the equations. A system is as crowded as its most crowded equation;
    distances of different equations are never added."""
    from epde.operators.multiobjective.moeadd_specific import (
        _crowding_distances, equation_planes)
    if not systems:
        return np.zeros(0)
    planes, _ = equation_planes(systems)
    return np.min(np.stack([_crowding_distances(points) for points in planes]), axis=0)


def plane_ranks(systems) -> list:
    """Each system's non-domination ranks in every equation's own plane, sorted
    best first (``evolution.niching='per_equation'``): compared as tuples, a
    system good in one equation beats one mediocre in all, and nothing is
    added across equations."""
    from epde.operators.multiobjective.moeadd_specific import equation_planes, plane_levels
    planes, labels = equation_planes(systems)
    ranks = np.array([plane_levels(points, labs) for points, labs in zip(planes, labels)]).T
    return [tuple(sorted(int(r) for r in row)) for row in ranks]


class NSGASelection(CompoundOperator):
    """Binary tournaments on (Pareto level, crowding): two systems drawn
    uniformly with replacement, the one on the shallower level wins, and on
    one level the one with the larger crowding distance (the first drawn on a
    full tie). ``parents_fraction`` of the population is drawn; each winner's
    ``incr_counter`` puts it into ``ParetoLevelsCrossover``'s mating pool.

    Under ``evolution.niching='per_equation'`` the level is replaced by the
    system's per-equation plane ranks (:func:`plane_ranks`). With
    ``evolution.complementary_mating_prob`` the second winner is, with that
    probability, redrawn from the systems holding some equation's own front
    where the first does not, and the two are paired first (MOEA/DD's
    ``_complementary_partner``).

    The block keeps MOEA/DD's label ``'selection'`` and carries an unused
    ``neighborhood_selector``, so the settings ``EpdeSearch.set_moeadd_params``
    writes onto that block land somewhere harmless.
    """
    key = 'NSGASelection'

    def apply(self, objective, arguments: dict):
        from epde.operators.multiobjective.moeadd_specific import niching_per_equation
        from epde.operators.multiobjective.selections import _complementary_partner
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)
        population = objective.population
        rank, crowding = {}, {}
        for level_idx, level in enumerate(objective.levels):
            for member, distance in zip(level, plane_crowding(level)):
                rank[id(member)] = level_idx
                crowding[id(member)] = distance
        per_equation = plane_ranks(population) if niching_per_equation(population) else None

        def key(idx):
            member = population[idx]
            first = per_equation[idx] if per_equation is not None else rank.get(id(member), np.inf)
            return (first, -crowding.get(id(member), 0.0))

        n = len(population)
        parents = int(n * self.params['parents_fraction'])
        parents = min(max(2, parents + parents % 2), n + n % 2)
        winners = []
        for _ in range(parents):
            first, second = int(np.random.randint(n)), int(np.random.randint(n))
            winners.append(second if key(second) < key(first) else first)
        winners, pair = _complementary_partner(population, np.array(winners))
        objective.forced_pair = None if pair is None else (population[pair[0]], population[pair[1]])
        for winner in winners.reshape(-1):
            population[int(winner)].incr_counter()
        return objective

    def use_default_tags(self):
        self._tags = {'selection', 'population level', 'auxilary', 'suboperators', 'standard'}


class AppendUpdater(CompoundOperator):
    """Adds an offspring to the population and nothing else: NSGA-II sorts and
    cuts the parents and offspring together once per generation
    (``NSGAOptimizer._truncate``), so nothing reads the levels in between."""
    key = 'AppendUpdater'

    def apply(self, objective, arguments: dict):
        offspring, levels_obj = objective
        levels_obj.population.append(offspring)

    def use_default_tags(self):
        self._tags = {'pareto level update', 'custom level', 'no suboperators', 'inplace'}
