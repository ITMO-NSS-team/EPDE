"""The own front of every variable: ``per_variable.variable_fronts`` and ``EpdeSearch.equations_by_variable``.

Pinned here:

* ``variable_fronts`` on a fitted two-equation population -- the keys, the equations being the systems' own
  objects, one equation per structure, no listed equation dominated in its plane, every first-level structure
  listed, the order of the list;
* ``variable_fronts`` on hand-built systems -- a structure shared by two systems listed once (the copy of the
  better level, the newest on a tie), a dominated structure absent, the empty input;
* ``EpdeSearch.equations_by_variable`` -- the equations, their text forms, the printed report, the systems it
  reads under NSGA-II (the archive) and under MOEA/DD (the Pareto levels), the refusals before a fit and in the
  single-objective mode;
* the report changes nothing: neither the global random state nor the archive.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import active_config, set_active_config
from epde.operators.common import plain_fit as PF
from epde.optimizers.nsga import per_variable as PV


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _search(engine='nsga2', **extra):
    """A small two-variable search: four terms, population 8, two epochs."""
    if engine == 'nsga2':
        extra.update(engine='nsga2', system_rps_mode='plain')
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, discrepancy_metric='wape',
                             instability_metric='dyad', equation_terms_max_number=4,
                             equation_factors_max_number=1, data_fun_pow=1, population_size=8,
                             training_epochs=2, **extra)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _fit(search, seed):
    """u = sin t, v = cos t: u' = v, v' = -u."""
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, 300)
    domain = search.createDomain(t, ID=0)[1]
    trajectory = search.createTrajectory({'u': np.sin(t), 'v': np.cos(t)}, domain, cache_id=0)[1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.fit(data=[trajectory], max_deriv_order=(1,))


def _fitted(engine, seed):
    """Generator body of a fit fixture: the finished search, the samples manager live until the end."""
    previous = active_config()
    search = _search(engine)
    try:
        _fit(search, seed)
        yield search
    finally:
        global_var.samples_manager.reset()
        PF.release()
        set_active_config(previous)


@pytest.fixture(scope='class')
def nsga_search():
    yield from _fitted('nsga2', seed=2)


@pytest.fixture(scope='class')
def moeadd_search():
    yield from _fitted('moeadd', seed=2)


def _held(equation, key, systems):
    """True when ``equation`` is the equation of ``key`` of one of ``systems``."""
    return any(system.vals[key] is equation for system in systems)


def _dominated(P, labels, i):
    """Plane point ``i`` is dominated by a point of another structure (written apart from ``plane_table``)."""
    return any(labels[j] != labels[i] and all(P[j] <= P[i]) and any(P[j] < P[i]) for j in range(len(P)))


def _flatten(levels):
    return [system for level in levels for system in level]


class _Vals(object):
    """What the plane ranking reads of ``SoEq.vals``: the keys, the length, the item of a key."""

    def __init__(self, equations):
        self.equation_keys = list(equations)
        self._equations = dict(equations)

    def __len__(self):
        return len(self._equations)

    def __getitem__(self, key):
        return self._equations[key]


def _stub_system(a, b):
    """A system of two stub equations a = (structure, discrepancy, instability), likewise b; the objectives
    are grouped by type, then by equation."""
    (la, da, ia), (lb, db, ib) = a, b
    vals = _Vals({'a': SimpleNamespace(terms_labels=la), 'b': SimpleNamespace(terms_labels=lb)})
    return SimpleNamespace(vals=vals, obj_fun=[da, db, ia, ib])


# ---------------------------------------------------------------------------
# variable_fronts on a fitted population.
# ---------------------------------------------------------------------------

class TestFittedFronts:

    def test_keys_equations_and_structures(self, nsga_search):
        systems = list(nsga_search.optimizer.pareto_levels.population)
        T = PV.plane_table(systems)
        fronts = PV.variable_fronts(systems)
        assert list(fronts) == list(systems[0].vals.equation_keys) == T.keys
        assert len(T.keys) == 2
        for e, key in enumerate(T.keys):
            assert fronts[key]
            labels = []
            for equation, objectives in fronts[key]:
                holders = [i for i, system in enumerate(systems)
                           if system.vals[key] is equation and tuple(T.planes[e][i]) == objectives]
                assert holders, 'a listed equation is not an equation of the systems'
                assert all(isinstance(v, float) for v in objectives)
                labels.append(T.labels[e][holders[0]])
            assert len(set(labels)) == len(labels), 'a structure is listed twice'

    def test_no_listed_equation_is_dominated_and_every_first_level_structure_is_listed(self, nsga_search):
        systems = list(nsga_search.optimizer.pareto_levels.population)
        T = PV.plane_table(systems)
        fronts = PV.variable_fronts(systems)
        for e, key in enumerate(T.keys):
            P, labels = np.asarray(T.planes[e], dtype=float), T.labels[e]
            listed = set()
            for equation, objectives in fronts[key]:
                i = next(i for i, system in enumerate(systems)
                         if system.vals[key] is equation and tuple(P[i]) == objectives)
                assert not _dominated(P, labels, i)
                listed.add(labels[i])
            first_level = {labels[i] for i in range(len(P)) if not _dominated(P, labels, i)}
            assert listed == first_level

    def test_the_list_is_sorted_by_objectives(self, nsga_search):
        systems = list(nsga_search.optimizer.pareto_levels.population)
        for key, front in PV.variable_fronts(systems).items():
            objectives = [objective for _, objective in front]
            assert objectives == sorted(objectives)

    def test_the_front_is_smaller_than_the_population(self, nsga_search):
        systems = list(nsga_search.optimizer.pareto_levels.population)
        for front in PV.variable_fronts(systems).values():
            assert len(front) <= len(systems)

    def test_a_single_system_lists_its_own_equations(self, nsga_search):
        system = nsga_search.optimizer.pareto_levels.population[0]
        fronts = PV.variable_fronts([system])
        assert list(fronts) == list(system.vals.equation_keys)
        for key, front in fronts.items():
            assert len(front) == 1 and front[0][0] is system.vals[key]


# ---------------------------------------------------------------------------
# variable_fronts on hand-built systems.
# ---------------------------------------------------------------------------

class TestHandBuilt:

    def test_the_empty_input_is_an_empty_dict(self):
        assert PV.variable_fronts([]) == {}
        assert PV.variable_fronts(iter(())) == {}

    def test_shared_dominated_and_sorted(self):
        systems = [
            _stub_system((('X',), .1, .5), (('P',), .2, .2)),
            _stub_system((('X',), .3, .6), (('Q',), .4, .1)),      # X again, a worse copy
            _stub_system((('Y',), .2, .6), (('P',), .2, .2)),      # Y is dominated by X(.1, .5)
            _stub_system((('Z',), .05, .9), (('R',), .3, .3)),     # R is dominated by P(.2, .2)
        ]
        fronts = PV.variable_fronts(systems)
        assert list(fronts) == ['a', 'b']
        assert [(eq.terms_labels, obj) for eq, obj in fronts['a']] == [(('Z',), (.05, .9)), (('X',), (.1, .5))]
        assert [(eq.terms_labels, obj) for eq, obj in fronts['b']] == [(('P',), (.2, .2)), (('Q',), (.4, .1))]
        # The structure shared by two systems is the equation of the copy on the better level.
        assert fronts['a'][1][0] is systems[0].vals['a']
        # P of systems 0 and 2: system 0 stands beside the front equation X, system 2 beside the dominated Y.
        assert fronts['b'][0][0] is systems[0].vals['b']
        assert all(eq is not systems[2].vals['a'] for eq, _ in fronts['a'])

    def test_the_newest_copy_represents_a_tie(self):
        systems = [_stub_system((('X',), .1, .5), (('P',), .2, .2)),
                   _stub_system((('X',), .1, .5), (('P',), .2, .2))]
        fronts = PV.variable_fronts(systems)
        assert [len(front) for front in fronts.values()] == [1, 1]
        assert fronts['a'][0][0] is systems[1].vals['a'] and fronts['b'][0][0] is systems[1].vals['b']

    def test_equal_points_of_two_structures_keep_the_systems_order(self):
        systems = [_stub_system((('X',), .1, .5), (('P',), .2, .2)),
                   _stub_system((('W',), .1, .5), (('P',), .2, .2))]
        front = PV.variable_fronts(systems)['a']
        assert [eq.terms_labels for eq, _ in front] == [('X',), ('W',)]

    def test_a_single_equation_system_has_one_key(self):
        def system(label, d, i):
            return SimpleNamespace(vals=_Vals({'u': SimpleNamespace(terms_labels=label)}), obj_fun=[d, i])
        systems = [system(('A',), .3, .3), system(('B',), .1, .6), system(('C',), .4, .4)]
        fronts = PV.variable_fronts(systems)
        assert list(fronts) == ['u']
        assert [eq.terms_labels for eq, _ in fronts['u']] == [('B',), ('A',)]


# ---------------------------------------------------------------------------
# EpdeSearch.equations_by_variable.
# ---------------------------------------------------------------------------

class TestEquationsByVariable:

    def test_the_equations_of_every_variable(self, nsga_search):
        result = nsga_search.equations_by_variable(only_print=False)
        archive = nsga_search.optimizer.reported_front()
        assert list(result) == ['u', 'v']
        for key, equations in result.items():
            assert equations
            assert all(_held(equation, key, archive) for equation in equations)
        expected = PV.variable_fronts(archive)
        for key in result:
            assert [id(eq) for eq in result[key]] == [id(eq) for eq, _ in expected[key]]

    def test_the_text_forms(self, nsga_search):
        equations = nsga_search.equations_by_variable(only_print=False)
        texts = nsga_search.equations_by_variable(only_print=False, only_str=True)
        assert list(texts) == list(equations)
        for key in equations:
            assert texts[key] == [equation.text_form for equation in equations[key]]
            assert all(isinstance(text, str) for text in texts[key])

    def test_printing_names_every_variable_and_returns_none(self, nsga_search, capsys):
        assert nsga_search.equations_by_variable() is None
        out = capsys.readouterr().out
        for key, equations in nsga_search.equations_by_variable(only_print=False, only_str=True).items():
            assert f'front of the variable {key}' in out
            for text in equations:
                assert text in out
        assert nsga_search.equations_by_variable(only_print=True, only_str=True) is None

    def test_it_reads_the_reported_front(self, nsga_search, monkeypatch):
        archive = nsga_search.optimizer.reported_front()
        assert len(archive) > 1
        only = archive[:1]
        monkeypatch.setattr(nsga_search.optimizer, 'reported_front', lambda: list(only))
        result = nsga_search.equations_by_variable(only_print=False)
        for key, equations in result.items():
            assert len(equations) == 1 and equations[0] is only[0].vals[key]

    def test_an_empty_archive_reports_nothing(self, nsga_search, monkeypatch, capsys):
        monkeypatch.setattr(nsga_search.optimizer, 'reported_front', lambda: [])
        assert nsga_search.equations_by_variable(only_print=False) == {}
        assert nsga_search.equations_by_variable() is None
        assert capsys.readouterr().out == ''

    def test_a_search_before_a_fit_raises_attribute_error(self):
        search = _search()
        with pytest.raises(AttributeError, match='has not been discovered'):
            search.equations_by_variable()
        with pytest.raises(AttributeError, match='has not been discovered'):
            search.equations_by_variable(only_print=False)

    def test_the_single_objective_mode_points_to_equations(self):
        search = epde.EpdeSearch(multiobjective_mode=False, verbose_params={'show_iter_idx': False})
        search.search_conducted = True
        search.optimizer = SimpleNamespace(population=SimpleNamespace(population=[]))
        with pytest.raises(ValueError, match=r'multi-objective mode.*equations'):
            search.equations_by_variable()
        with pytest.raises(ValueError, match='multi-objective mode'):
            search.equations_by_variable(only_print=False, only_str=True)


class TestMoeadd:

    def test_the_fronts_come_from_the_pareto_levels(self, moeadd_search):
        optimizer = moeadd_search.optimizer
        assert not hasattr(optimizer, 'reported_front')
        systems = _flatten(optimizer.pareto_levels.levels)
        assert systems
        result = moeadd_search.equations_by_variable(only_print=False)
        expected = PV.variable_fronts(systems)
        assert list(result) == list(expected) == ['u', 'v']
        for key, equations in result.items():
            assert equations
            assert all(_held(equation, key, systems) for equation in equations)
            assert [id(eq) for eq in equations] == [id(eq) for eq, _ in expected[key]]

    def test_the_text_forms_and_the_report(self, moeadd_search, capsys):
        texts = moeadd_search.equations_by_variable(only_print=False, only_str=True)
        equations = moeadd_search.equations_by_variable(only_print=False)
        for key in equations:
            assert texts[key] == [equation.text_form for equation in equations[key]]
        assert moeadd_search.equations_by_variable() is None
        out = capsys.readouterr().out
        assert all(f'front of the variable {key}' in out for key in equations)


# ---------------------------------------------------------------------------
# The report changes nothing.
# ---------------------------------------------------------------------------

class TestReadOnly:

    def test_the_random_state_and_the_archive_are_unchanged(self, nsga_search):
        archive = nsga_search.optimizer.reported_front()
        keys = list(archive[0].vals.equation_keys)

        def snapshot():
            return [(tuple(PV._eq_label(system.vals[k]) for k in keys), tuple(float(v) for v in system.obj_fun))
                    for system in nsga_search.optimizer.reported_front()]

        before_state, before = np.random.get_state(), snapshot()
        nsga_search.equations_by_variable(only_print=False)
        nsga_search.equations_by_variable(only_print=False, only_str=True)
        nsga_search.equations_by_variable()
        after_state, after = np.random.get_state(), snapshot()
        assert before_state[0] == after_state[0] and before_state[2:] == after_state[2:]
        assert np.array_equal(before_state[1], after_state[1])
        assert before == after
        now = nsga_search.optimizer.reported_front()
        assert len(now) == len(archive) and all(a is b for a, b in zip(now, archive))
