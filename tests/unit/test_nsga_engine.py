#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``evolution.engine='nsga2'`` and the cycle-safe sort.

Pinned here:

* the setting and its refusals;
* the sort never loses a member of a dominance cycle, and equals Deb's sort
  without one; the incremental update likewise;
* crowding is measured per equation and the smallest taken;
* the tournament prefers the shallower level; the updater only appends;
* truncation keeps whole levels, then the least crowded of the overflowing one;
* the archive keeps every system nothing evaluated dominates;
* per-equation niching: the truncation keeps every equation's front holders
  first (more fronts first on an overflow), the tournament reads the plane
  ranks, and complementary mating forces a pair from another front;
* one epoch makes the offspring of one MOEA/DD epoch;
* end to end, with 'plain' and with the default right-part selection, the
  post-fit accessors work.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import collect_overrides, load_search_config
from epde.operators.multiobjective.moeadd_specific import _crowding_distances
from epde.optimizers.moeadd.supplementary import (check_dominance, cycle_safe_ndl_update,
                                                  cycle_safe_nds, fast_non_dominated_sorting,
                                                  ndl_update)
from epde.optimizers.nsga import NSGAOptimizer, moeadd_parents_per_sector
from epde.optimizers.nsga.operators import AppendUpdater, NSGASelection, plane_crowding
from epde.optimizers.nsga.operators import plane_ranks as ops_plane_ranks

from conftest import using_config


class _Vals(dict):
    @property
    def equation_keys(self):
        return list(self.keys())


class _Stub(SimpleNamespace):
    """A system as ``check_dominance`` reads it: equation labels and an
    objective vector grouped by type, then by equation."""

    def __init__(self, labels, objectives, name=''):
        super().__init__(vals=_Vals({key: SimpleNamespace(structure_label=label, terms_labels=label)
                                     for key, label in zip('uvw', labels)}),
                         obj_fun=np.asarray(objectives, dtype=float), name=name, counter=0)
        self.equations_labels = tuple(labels)

    def incr_counter(self):
        self.counter += 1


def _cycle():
    """A > B > C > A. A beats B on v (u skipped: same structure), B beats C
    everywhere, C beats A on u (v skipped)."""
    a = _Stub(['S', 'T'], [3, 1, 0, 0], 'A')     # obj: (u_disc, v_disc, u_inst, v_inst)
    b = _Stub(['S', 'T2'], [1, 2, 0, 0], 'B')
    c = _Stub(['S3', 'T'], [2, 3, 0, 0], 'C')
    return a, b, c


# ---------------------------------------------------------------------------
# Settings.
# ---------------------------------------------------------------------------

class TestTheSetting:

    def test_default_and_accepted(self):
        assert load_search_config().evolution.engine == 'moeadd'
        assert load_search_config(overrides={'engine': 'nsga2'}).evolution.engine == 'nsga2'

    @pytest.mark.parametrize('overrides, match', [
        ({'engine': 'spea2'}, 'engine'),
        ({'engine': 'nsga2', 'multiobjective_mode': False}, 'multiobjective'),
        ({'engine': 'nsga2', 'system_rps_mode': 'assignment',
          'system_rps_params': {'variant_pick': 'sector'}}, 'variant_pick'),
    ])
    def test_refusals(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            load_search_config(overrides=collect_overrides(**overrides))

    def test_per_equation_niching_and_complementary_mating_are_accepted(self):
        cfg = load_search_config(overrides=collect_overrides(
            engine='nsga2', system_rps_mode='plain', niching='per_equation', complementary_mating_prob=0.5))
        assert (cfg.evolution.niching, cfg.evolution.complementary_mating_prob) == ('per_equation', 0.5)


# ---------------------------------------------------------------------------
# The cycle-safe sort.
# ---------------------------------------------------------------------------

class TestTheCycleSafeSort:

    def test_the_fixture_is_a_cycle(self):
        a, b, c = _cycle()
        assert check_dominance(a, b) and check_dominance(b, c) and check_dominance(c, a)
        assert sum(len(level) for level in fast_non_dominated_sorting([a, b, c])) < 3

    def test_no_member_of_a_cycle_is_lost(self):
        a, b, c = _cycle()
        d = _Stub(['S9', 'T9'], [9, 9, 9, 9], 'D')     # dominated by everyone
        levels = cycle_safe_nds([a, b, c, d])
        assert sorted(m.name for level in levels for m in level) == ['A', 'B', 'C', 'D']

    def test_without_a_cycle_it_is_debs_sort(self):
        rng = np.random.default_rng(1)
        systems = [_Stub([f'a{k}', f'b{k}'], rng.random(4)) for k in range(30)]
        fast = fast_non_dominated_sorting(systems)
        safe = cycle_safe_nds(systems)
        assert [[id(m) for m in level] for level in safe] == [[id(m) for m in level] for level in fast]

    def test_the_incremental_update_keeps_every_member(self):
        a, b, c = _cycle()
        levels = cycle_safe_ndl_update(c, cycle_safe_nds([a, b]))
        assert sorted(m.name for level in levels for m in level) == ['A', 'B', 'C']
        rng = np.random.default_rng(2)
        systems = [_Stub([f'a{k}', f'b{k}'], rng.random(4)) for k in range(12)]
        levels = fast_non_dominated_sorting(systems[:-1])
        assert [[id(m) for m in lv] for lv in cycle_safe_ndl_update(systems[-1], levels)] == \
            [[id(m) for m in lv] for lv in ndl_update(systems[-1], levels)]


def _pairwise_sort(population):
    """The pairwise ``fast_non_dominated_sorting`` (Deb's loop), verbatim."""
    from epde.optimizers.moeadd.supplementary import _dominance_view, _view_dominates
    levels, count = [], np.zeros(len(population))
    dominated = [[] for _ in population]
    current = []
    views = [_dominance_view(member) for member in population]
    for main in range(len(population)):
        for other in range(len(population)):
            if main == other:
                continue
            if _view_dominates(views[other], views[main]):
                count[main] += 1
            elif _view_dominates(views[main], views[other]):
                dominated[main].append(other)
        if count[main] == 0:
            current.append(main)
    levels.append([population[i] for i in current])
    while current:
        new = []
        for main in current:
            for i in dominated[main]:
                count[i] -= 1
                if count[i] == 0:
                    new.append(i)
        if new:
            levels.append([population[i] for i in new])
        current = new
    return levels


def _pairwise_cycle_safe(population):
    """The pairwise ``cycle_safe_nds``, verbatim."""
    from epde.optimizers.moeadd.supplementary import _dominance_view, _view_dominates
    levels = _pairwise_sort(population)
    placed = {id(member) for level in levels for member in level}
    rest = [member for member in population if id(member) not in placed]
    if not rest:
        return levels
    levels = [level for level in levels if level]
    views = {id(member): _dominance_view(member) for member in rest}
    while rest:
        counts = [sum(_view_dominates(views[id(other)], views[id(member)]) for other in rest if other is not member)
                  for member in rest]
        fewest = min(counts)
        levels.append([member for member, count in zip(rest, counts) if count == fewest])
        rest = [member for member, count in zip(rest, counts) if count != fewest]
    return levels


def _random_population(rng, size, n_eq, per_eq, kinds=3):
    """Systems with few structure labels per equation (so labels are often
    shared and cycles form), ties, NaN, inf and -0.0, frozenset and
    (frozenset, bool) labels, and one object listed twice."""
    values = np.array([0.0, -0.0, 0.5, 1.0, 1.0, 2.0, np.inf, np.nan])
    population = []
    for k in range(size):
        labels = []
        for e in range(n_eq):
            label = frozenset({f'{"uvw"[e]}{rng.integers(kinds)}'})
            labels.append((label, bool(rng.integers(2))) if rng.random() < 0.5 else label)
        n_obj = per_eq * n_eq
        objectives = np.where(rng.random(n_obj) < 0.5, rng.choice(values, n_obj), rng.random(n_obj))
        population.append(_Stub(labels, objectives, f's{k}'))
    if size > 2:
        population.append(population[int(rng.integers(size))])
    return population


class TestTheMatrixSort:
    """The sorts read a dominance matrix from 8 members on; the levels must be
    the pairwise loops' member for member and in order."""

    @pytest.mark.parametrize('min_members', [1, 8])
    @pytest.mark.parametrize('n_eq, per_eq', [(1, 2), (2, 2), (3, 2), (2, 1), (2, 3), (3, 3)])
    def test_the_levels_are_the_pairwise_ones(self, monkeypatch, min_members, n_eq, per_eq):
        import epde.optimizers.moeadd.supplementary as supplementary
        monkeypatch.setattr(supplementary, '_MATRIX_MIN_MEMBERS', min_members)
        built = []
        real_matrix = supplementary._dominance_matrix
        monkeypatch.setattr(supplementary, '_dominance_matrix',
                            lambda views: built.append(real_matrix(views) is not None) or real_matrix(views))
        rng = np.random.default_rng(n_eq * 10 + per_eq)
        cycles = 0
        for size in [0, 1, 2, 3] + [int(s) for s in rng.integers(20, 160, 8)]:
            population = _random_population(rng, size, n_eq, per_eq)
            del built[:]
            expected = _pairwise_sort(population)
            assert [[id(m) for m in level] for level in fast_non_dominated_sorting(population)] == \
                [[id(m) for m in level] for level in expected]
            safe = _pairwise_cycle_safe(population)
            assert [[id(m) for m in level] for level in cycle_safe_nds(population)] == \
                [[id(m) for m in level] for level in safe]
            # The matrix path did run (both sorts) from its floor on: the oracle cannot tell.
            assert built == [len(population) >= max(min_members, 1)] * 2
            cycles += sum(map(len, expected)) < len(population)
        if n_eq > 1:
            assert cycles          # the peel was exercised

    def test_a_small_block_gives_the_same_matrix(self, monkeypatch):
        import epde.optimizers.moeadd.supplementary as supplementary
        population = _random_population(np.random.default_rng(5), 60, 2, 2)
        views = [supplementary._dominance_view(member) for member in population]
        whole = supplementary._dominance_matrix(views)
        monkeypatch.setattr(supplementary, '_MATRIX_BLOCK', 1)
        np.testing.assert_array_equal(supplementary._dominance_matrix(views), whole)
        assert not np.any(whole & whole.T) and not np.any(np.diag(whole))

    def test_views_that_do_not_share_their_shape_take_the_pairwise_path(self):
        import epde.optimizers.moeadd.supplementary as supplementary
        population = _random_population(np.random.default_rng(6), 20, 2, 2)
        population.append(_Stub(['u0', 'v0'], [0, 0], 'short'))
        assert supplementary._dominance_matrix([supplementary._dominance_view(m) for m in population]) is None

    def test_the_other_fallbacks_take_the_pairwise_path(self, monkeypatch):
        import epde.optimizers.moeadd.supplementary as supplementary
        population = _random_population(np.random.default_rng(7), 20, 2, 2)
        views = [supplementary._dominance_view(m) for m in population]
        assert supplementary._dominance_matrix(views) is not None
        unhashable = population + [_Stub([['u0'], ['v0']], [0.0] * 4, 'unhashable')]
        assert supplementary._dominance_matrix([supplementary._dominance_view(m) for m in unhashable]) is None
        monkeypatch.setattr(supplementary, '_MATRIX_MAX_MEMBERS', len(views) - 1)
        assert supplementary._dominance_matrix(views) is None


# ---------------------------------------------------------------------------
# Operators.
# ---------------------------------------------------------------------------

class TestTheOperators:

    def test_crowding_is_per_equation_and_the_smallest(self):
        rng = np.random.default_rng(3)
        systems = [_Stub([f'a{k}', f'b{k}'], rng.random(4)) for k in range(6)]
        objs = np.array([s.obj_fun for s in systems])
        expected = np.minimum(_crowding_distances(objs[:, [0, 2]]), _crowding_distances(objs[:, [1, 3]]))
        np.testing.assert_array_equal(plane_crowding(systems), expected)
        single = [_Stub([f'a{k}'], rng.random(2)) for k in range(5)]
        np.testing.assert_array_equal(plane_crowding(single),
                                      _crowding_distances(np.array([s.obj_fun for s in single])))

    def test_the_tournament_prefers_the_shallower_level(self):
        best, worse = _Stub(['a'], [0, 0]), _Stub(['b'], [1, 1])
        levels = SimpleNamespace(population=[best, worse], levels=[[best], [worse]], forced_pair='x')
        selection = NSGASelection(['parents_fraction'])
        selection.params = {'parents_fraction': 1.0}
        np.random.seed(0)
        for _ in range(200):          # two tournaments per call: as many as members
            selection.apply(levels, {})
        assert best.counter + worse.counter == 400
        assert best.counter > 2 * worse.counter      # worse wins only when drawn twice
        assert levels.forced_pair is None

    def test_the_updater_only_appends(self):
        population = SimpleNamespace(population=[1, 2], levels=[[1, 2]])
        AppendUpdater().apply((3, population), {})
        assert population.population == [1, 2, 3] and population.levels == [[1, 2]]

    @pytest.mark.parametrize('mu, parents', [(32, 6), (16, 4), (8, 2), (3, 2), (40, 8)])
    def test_an_epoch_makes_the_offspring_of_a_moeadd_epoch(self, mu, parents):
        assert moeadd_parents_per_sector(mu) == parents


# ---------------------------------------------------------------------------
# Truncation and archive (on a bare optimizer).
# ---------------------------------------------------------------------------

def _bare(population, mu):
    optimizer = NSGAOptimizer.__new__(NSGAOptimizer)
    optimizer.pareto_levels = SimpleNamespace(population=list(population), levels=[])
    optimizer._mu = mu
    optimizer._archive = []
    return optimizer


class TestTruncationAndArchive:

    def test_whole_levels_then_the_least_crowded(self):
        front = [_Stub([f'f{k}'], [k, 4 - k]) for k in range(5)]      # one front of five
        behind = [_Stub(['z'], [9, 9])]
        optimizer = _bare(front + behind, mu=3)
        optimizer._truncate()
        kept = optimizer.pareto_levels.population
        assert len(kept) == 3 and behind[0] not in kept
        # Crowding keeps the two extremes (infinite distance) and one inner point.
        assert front[0] in kept and front[4] in kept

    def test_whole_levels_fit_untouched(self):
        a, b = _Stub(['a'], [0, 1]), _Stub(['b'], [1, 0])
        c = _Stub(['c'], [2, 2])
        optimizer = _bare([a, b, c], mu=3)
        optimizer._truncate()
        assert [len(level) for level in optimizer.pareto_levels.levels] == [2, 1]

    def test_the_archive_keeps_exactly_what_nothing_dominates(self):
        optimizer = _bare([], mu=4)
        a, b, c = _Stub(['a'], [1, 1]), _Stub(['b'], [0, 2]), _Stub(['c'], [2, 2])
        for system in (c, a, b):
            optimizer._archive_insert(system)
        assert sorted(id(s) for s in optimizer.reported_front()) == sorted([id(a), id(b)])
        assert not optimizer._archive_insert(_Stub(['d'], [3, 3]))
        assert not optimizer._archive_insert(_Stub(['a'], [1, 1]))     # the same candidate again

    def test_a_twin_replaces_the_archived_arrangement_only_when_it_dominates(self):
        optimizer = _bare([], mu=4)
        held, other = _Stub(['a'], [1, 1], 'held'), _Stub(['b'], [0, 3], 'other')
        optimizer._archive_insert(held)
        optimizer._archive_insert(other)
        assert not optimizer._archive_insert(_Stub(['a'], [0.5, 1.5], 'mixed'))     # not dominating
        better = _Stub(['a'], [0.5, 0.5], 'better')
        assert optimizer._archive_insert(better)
        assert sorted(m.name for m in optimizer.reported_front()) == ['better', 'other']

    def test_a_dropped_incumbent_stays_referenced_until_the_newcomers_are_archived(self, monkeypatch):
        """A generation that drops an incumbent nothing archived must not free it before its new
        members are archived: a newcomer could take over the freed id() and pass for an old member."""
        import weakref
        # 'weak' first: optimize()'s loop variable still holds the last member it archived.
        optimizer = _bare([_Stub(['weak'], [5, 5], 'weak')]
                          + [_Stub([f'f{k}'], [k, 3 - k], f'f{k}') for k in range(4)], mu=5)
        optimizer.best_obj = 0
        optimizer._hist, optimizer._pareto_history = [], []
        optimizer.pareto_levels.get_stats = lambda: None
        optimizer.form_processer_args = lambda *args: {}
        optimizer._snapshot = lambda: []
        monkeypatch.setattr(global_var, 'verbose', SimpleNamespace(show_iter_idx=False), raising=False)
        alive, newcomers = [], []

        def run(population_subset, EA_kwargs):
            population = population_subset.population
            archived = {id(member) for member in optimizer._archive}
            dropped = [member for member in population if id(member) not in archived]
            if dropped:
                population.remove(dropped[0])
                ref = weakref.ref(dropped[0])
                del dropped
                alive.append(ref() is not None)
            k = len(newcomers)
            newcomers.append(_Stub([f'n{k}'], [0.5 + k, 2.5 - k], f'n{k}'))     # on the front
            population.append(newcomers[-1])

        optimizer.sector_processer = SimpleNamespace(run=run)
        optimizer.optimize(1)
        assert alive == [True]
        assert {m.name for m in newcomers} <= {m.name for m in optimizer.reported_front()}


class TestTheArchiveRows:
    """The archive compares a newcomer with every member at once
    (``DominanceRows``); it must decide exactly as ``check_dominance`` one by
    one: the same verdict and the same members in the same order after every
    insert -- through ties, NaN, inf, -0.0, shared structure labels and twins."""

    @pytest.mark.parametrize('n_eq, per_eq, seed', [(n, m, s) for n in (1, 2, 3) for m in (1, 2, 3)
                                                    for s in range(3)])
    def test_the_rows_decide_as_check_dominance(self, n_eq, per_eq, seed):
        rng = np.random.default_rng(seed)
        values = np.array([0.0, -0.0, 0.5, 1.0, 1.0, 2.0, np.inf, np.nan])
        rows, scalar = _bare([], mu=4), _bare([], mu=4)
        for step in range(150):
            labels = [f'{"uvw"[e]}{rng.integers(3)}' for e in range(n_eq)]
            size = per_eq * n_eq
            objectives = np.where(rng.random(size) < 0.6, rng.choice(values, size), rng.random(size))
            system = _Stub(labels, objectives, f's{step}')
            if rng.random() < 0.3:          # an arrangement label wider than the structures
                system.equations_labels = tuple(labels) + (int(rng.integers(2)),)
            entered = rows._archive_insert(system)
            assert entered == scalar._archive_insert_scalar(system, system.equations_labels)
            assert [id(m) for m in rows._archive] == [id(m) for m in scalar._archive]

    def test_rows_are_rebuilt_for_a_new_archive_list(self):
        optimizer = _bare([], mu=4)
        a, b = _Stub(['a'], [1, 1], 'a'), _Stub(['b'], [0, 2], 'b')
        optimizer._archive_insert(a)
        optimizer._archive = [b]                       # set from outside
        assert not optimizer._archive_insert(_Stub(['c'], [0, 3], 'c'))     # b dominates it
        assert optimizer._archive_insert(_Stub(['d'], [2, 0], 'd'))
        assert [m.name for m in optimizer._archive] == ['b', 'd']

    def test_a_view_that_does_not_fit_takes_the_scalar_path(self):
        a, b = _Stub(['a'], [1, 1], 'a'), _Stub(['b'], [0], 'b')      # a shorter objective vector
        optimizer, scalar = _bare([], mu=4), _bare([], mu=4)
        for system in (a, b):
            assert optimizer._archive_insert(system) == scalar._archive_insert_scalar(system, system.equations_labels)
        assert optimizer._rows is None
        assert [m.name for m in optimizer._archive] == [m.name for m in scalar._archive]

    def test_a_member_changed_in_place_is_reported(self, monkeypatch):
        from epde import _loop_stats
        monkeypatch.setattr(_loop_stats, '_ENABLED', True)
        _loop_stats.reset()
        optimizer = _bare([], mu=4)
        members = [_Stub([f'f{k}'], [k, 3 - k], f'f{k}') for k in range(3)]
        for member in members:
            optimizer._archive_insert(member)
        assert 'NSGA.archive_rows.stale' not in _loop_stats._stats
        for member in members:
            member.obj_fun = member.obj_fun - 0.5
        optimizer._archive_insert(_Stub(['g'], [9, 9], 'g'))
        assert _loop_stats._stats['NSGA.archive_rows.stale']['entries'] == 1
        _loop_stats.reset()


# ---------------------------------------------------------------------------
# evolution.niching='per_equation' and complementary mating under NSGA-II.
# ---------------------------------------------------------------------------

def _holders_fixture():
    """Two equations; obj = (u_disc, v_disc, u_inst, v_inst).
    X: the best u (tied with V's) but the worst v -- on u's own front, yet on
       system level 1, since V matches its u and beats its v;
    V: u as good as X's, a fair v; Q: the best v;
    N: on system level 0 but on neither equation's own front."""
    n = _Stub(['uN', 'vN'], [0.5, 0.5, 0.5, 0.5], 'N')
    x = _Stub(['uX', 'vX'], [0, 9, 0, 9], 'X')
    v = _Stub(['uV', 'vV'], [0, 1, 0, 1], 'V')
    q = _Stub(['uQ', 'vQ'], [1, 0, 1, 0], 'Q')
    return n, x, v, q


class TestPerEquationNiching:

    def test_the_fixture(self):
        n, x, v, q = _holders_fixture()
        levels = cycle_safe_nds([n, x, v, q])
        assert {m.name for m in levels[0]} == {'N', 'V', 'Q'} and [m.name for m in levels[1]] == ['X']

    def test_a_front_holder_survives_where_levels_alone_drop_it(self):
        n, x, v, q = _holders_fixture()
        plain = _bare([n, x, v, q], mu=3)
        plain._truncate()
        assert {m.name for m in plain.pareto_levels.population} == {'N', 'V', 'Q'}
        with using_config(niching='per_equation'):
            niched = _bare([n, x, v, q], mu=3)
            niched._truncate()
        assert {m.name for m in niched.pareto_levels.population} == {'X', 'V', 'Q'}
        assert sum(len(level) for level in niched.pareto_levels.levels) == 3

    def test_holders_come_first_then_levels_fill_the_rest(self):
        n, x, v, q = _holders_fixture()
        with using_config(niching='per_equation'):
            niched = _bare([n, x, v, q], mu=4)
            niched._truncate()
        assert {m.name for m in niched.pareto_levels.population} == {'N', 'X', 'V', 'Q'}

    def test_an_overflow_keeps_the_systems_on_more_fronts(self):
        x = _Stub(['uX', 'vX'], [0, 9, 0, 9], 'X')
        v = _Stub(['uV', 'vV'], [0, 0, 0, 0], 'V')       # on both fronts
        q = _Stub(['uQ', 'vQ'], [1, 0, 1, 0], 'Q')
        with using_config(niching='per_equation'):
            niched = _bare([x, q, v], mu=1)
            niched._truncate()
        assert [m.name for m in niched.pareto_levels.population] == ['V']

    def test_the_tournament_reads_the_plane_ranks(self, monkeypatch):
        import epde.optimizers.nsga.operators as ops
        shallow = _Stub(['a', 'b'], [0, 0, 0, 0], 'shallow')
        deep = _Stub(['c', 'd'], [1, 1, 1, 1], 'deep')
        levels = SimpleNamespace(population=[shallow, deep], levels=[[shallow], [deep]], forced_pair=None)
        monkeypatch.setattr(ops, 'plane_ranks', lambda systems: [(1, 1), (0, 0)])
        selection = NSGASelection(['parents_fraction'])
        selection.params = {'parents_fraction': 1.0}
        np.random.seed(0)
        with using_config(niching='per_equation'):
            for _ in range(200):
                selection.apply(levels, {})
        assert deep.counter > 2 * shallow.counter

    def test_plane_ranks(self):
        n, x, v, q = _holders_fixture()
        # u ranks N 1, X 0, V 0, Q 2; v ranks N 1, X 3, V 2, Q 0 -- sorted best first.
        assert ops_plane_ranks([n, x, v, q]) == [(1, 1), (0, 3), (0, 2), (0, 2)]

    def test_complementary_mating_forces_a_pair_from_another_front(self):
        n, x, v, q = _holders_fixture()
        levels = SimpleNamespace(population=[n, x, v, q], levels=cycle_safe_nds([n, x, v, q]), forced_pair=None)
        selection = NSGASelection(['parents_fraction'])
        selection.params = {'parents_fraction': 1.0}
        np.random.seed(1)
        with using_config(complementary_mating_prob=1.0):
            selection.apply(levels, {})
        first, second = levels.forced_pair
        holders = {'u': {'X', 'V'}, 'v': {'Q'}}
        assert any(second.name in held and first.name not in held for held in holders.values())


# ---------------------------------------------------------------------------
# End to end.
# ---------------------------------------------------------------------------

def _fit(**kwargs):
    np.random.seed(0)
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2',
                             discrepancy_metric='wape', instability_metric='dyad',
                             equation_terms_max_number=5, equation_factors_max_number=1,
                             data_fun_pow=1, **kwargs)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0, 4 * np.pi, 300)
    domain = search.createDomain(t, ID=0)[1]
    trajectory = search.createTrajectory({'u': np.sin(2 * t)}, domain, cache_id=0)[1]
    try:
        search.fit(data=[trajectory], max_deriv_order=(2,))
    finally:
        global_var.samples_manager.reset()
    return search


def _holds_the_truth(search):
    for system in search.optimizer.pareto_levels.levels[0]:
        eq = system.vals['u']
        names = {tuple(sorted(f.label for f in term.structure)) for term in eq.structure}
        if names == {('d^2u/dx0^2',), ('u',)}:
            return True
    return False


class TestEndToEnd:

    @pytest.mark.parametrize('mode', ['plain', 'repair'])
    def test_a_short_run_reports_the_truth_through_the_usual_accessors(self, mode):
        search = _fit(system_rps_mode=mode, training_epochs=2)
        optimizer = search.optimizer
        assert isinstance(optimizer, NSGAOptimizer)
        levels = optimizer.pareto_levels.levels
        assert levels and levels[0]
        # The first level is the archive.
        assert {id(s) for s in levels[0]} == {id(s) for s in optimizer.reported_front()}
        assert len(search.pareto_history) == 2
        assert search.equations(only_print=False, only_str=True)
        assert _holds_the_truth(search)
