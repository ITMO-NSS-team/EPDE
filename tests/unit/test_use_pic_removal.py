"""``use_pic`` is gone, and the MOEA/D ideal point belongs to the objectives.

``use_pic`` was a bool standing in for one three-valued choice: which
objective occupies MOEA/D's second Pareto axis. Three of its four consumers
routed through ``globals.resolve_second_objective``; the fourth -- the ideal
point in ``EpdeSearch._create_optimizer`` -- read the raw bool, so
``set_second_objective('complexity')`` together with ``use_pic=True`` left the
utopia point at the instability value while the front optimized complexity.

These tests pin both halves of the repair: the flag is gone, and the ideal
point is derived from each objective's own ``ideal_value`` rather than written
as a literal keyed on a flag.
"""

import ast
import inspect
import os
import types

import numpy as np
import pytest

import epde.globals as global_var
from conftest import using_config
from epde.interface.search_config import load_search_config
from epde.eq_mo_objectives import (equation_complexity, equation_fitness,
                                   equation_terms_stability,
                                   objective_ideal_values)
from epde.operators.common.objectives import (OBJECTIVE_REGISTRY, Complexity,
                                              Discrepancy, EquationObjective,
                                              Instability,
                                              ideal_point)
from epde.structure.main_structures import SoEq

_EPDE_ROOT = os.path.dirname(os.path.dirname(inspect.getfile(global_var)))


def _python_sources():
    for root, _, files in os.walk(os.path.join(_EPDE_ROOT, 'epde')):
        for name in files:
            if name.endswith('.py'):
                yield os.path.join(root, name)


# ---------------------------------------------------------------------------
# the flag is gone
# ---------------------------------------------------------------------------

class TestFlagRemoved:

    def test_no_identifier_named_use_pic_survives(self):
        """Comments and docstrings may still explain the removal; nothing may
        still be *named* use_pic."""
        offenders = []
        for path in _python_sources():
            with open(path, encoding='utf-8') as handle:
                source = handle.read()
            if 'use_pic' not in source:
                continue
            tree = ast.parse(source, filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.arg) and node.arg == 'use_pic':
                    offenders.append('%s: parameter' % path)
                elif isinstance(node, ast.Name) and node.id == 'use_pic':
                    offenders.append('%s:%d name' % (path, node.lineno))
                elif isinstance(node, ast.Attribute) and node.attr == 'use_pic':
                    offenders.append('%s:%d attribute' % (path, node.lineno))
                elif isinstance(node, ast.keyword) and node.arg == 'use_pic':
                    offenders.append('%s:%d keyword' % (path, node.lineno))
        assert not offenders, offenders

    def test_epde_search_does_not_accept_it(self):
        from epde.interface.interface import EpdeSearch
        params = inspect.signature(EpdeSearch.__init__).parameters
        assert 'use_pic' not in params
        assert 'second_objective' in params

    def test_default_is_what_use_pic_true_meant(self):
        cfg = load_search_config(overrides={'second_objective': None})
        assert cfg.objectives.second_objective == 'instability'

    def test_config_honours_an_explicit_choice(self):
        cfg = load_search_config(overrides={'second_objective': 'complexity'})
        assert cfg.objectives.second_objective == 'complexity'

    def test_the_axis_is_not_a_mutable_global(self):
        """It used to be a module scalar with a setter; the config is now the
        only way in, so the three lockstep sites cannot be set out of step."""
        for name in ('second_objective', 'set_second_objective',
                     'resolve_second_objective'):
            assert not hasattr(global_var, name), name


# ---------------------------------------------------------------------------
# the ideal point belongs to the objectives
# ---------------------------------------------------------------------------

class TestIdealValueOwnership:

    def test_complexity_optimum_is_one_factor(self):
        assert Complexity.ideal_value == 1.0

    def test_every_other_objective_bottoms_out_at_zero(self):
        for cls in (EquationObjective, Discrepancy, Instability):
            assert cls.ideal_value == 0.0, cls.__name__

    def test_registry_covers_every_selectable_axis(self):
        assert set(OBJECTIVE_REGISTRY) == {'discrepancy', 'instability',
                                           'complexity'}
        for name, cls in OBJECTIVE_REGISTRY.items():
            assert cls.name == name

    def test_ideal_point_for_instability(self):
        assert ideal_point(('discrepancy', 'instability')) == [0.0, 0.0]

    def test_ideal_point_for_complexity(self):
        assert ideal_point(('discrepancy', 'complexity')) == [0.0, 1.0]

    def test_ideal_point_follows_the_class_attribute(self):
        """The point is derived, not written down: move the attribute and the
        point moves with it."""
        original = Complexity.ideal_value
        try:
            Complexity.ideal_value = 7.0
            assert ideal_point(('discrepancy', 'complexity')) == [0.0, 7.0]
        finally:
            Complexity.ideal_value = original

    def test_unregistered_objective_is_rejected(self):
        with pytest.raises(ValueError, match='No objective registered'):
            ideal_point(('discrepancy', 'parsimony'))

    def test_no_ideal_point_literal_remains_in_the_interface(self):
        """The regression guard: ``[0., 0.] if use_pic else [0., 1.]`` must not
        come back in any form."""
        import textwrap

        from epde.interface import interface
        source = textwrap.dedent(
            inspect.getsource(interface.EpdeSearch._create_optimizer))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and len(node.elts) == 2:
                if all(isinstance(e, ast.Constant) and
                       isinstance(e.value, (int, float)) for e in node.elts):
                    pytest.fail('hardcoded ideal point %r back in '
                                '_create_optimizer' % ([e.value for e in node.elts],))

    def test_create_optimizer_takes_the_axis_not_a_flag(self):
        from epde.interface.interface import EpdeSearch
        params = inspect.signature(EpdeSearch._create_optimizer).parameters
        assert 'second_objective' in params
        assert 'use_pic' not in params


# ---------------------------------------------------------------------------
# the three lockstep sites
# ---------------------------------------------------------------------------

class _ObjectiveRecorder:
    """Minimal stand-in for SoEq.

    ``use_default_multiobjective_function`` dispatches to one of two sibling
    methods, both of which touch nothing but ``set_objective_functions`` -- so
    borrowing the real methods exercises the real registration without needing
    a token pool.
    """

    def __init__(self):
        self.obj_funs = None
        for name in ('use_default_multiobjective_function',
                     'use_new_multiobjective_function',
                     'use_legacy_multiobjective_function'):
            setattr(self, name, types.MethodType(getattr(SoEq, name), self))

    def set_objective_functions(self, obj_funs):
        self.obj_funs = obj_funs


def _fresh_director():
    from epde.optimizers.builder import StrategyBuilder
    from epde.optimizers.moeadd.moeadd import MOEADDSectorProcesser
    from epde.optimizers.moeadd.strategy import MOEADDDirector

    director = MOEADDDirector()
    director.builder = StrategyBuilder(MOEADDSectorProcesser)
    return director


def _registered_readers(second_objective):
    recorder = _ObjectiveRecorder()
    recorder.use_default_multiobjective_function(second_objective)
    return recorder.obj_funs


class TestLockstep:
    """Filler assembly, SoEq axis registration and the ideal point must agree
    for every value of the axis -- that agreement is what ``use_pic`` broke."""

    @pytest.mark.parametrize('axis,expected_reader', [
        ('instability', equation_terms_stability),
        ('complexity', equation_complexity),
    ])
    def test_soeq_registers_the_matching_reader(self, axis, expected_reader):
        readers = _registered_readers(axis)
        assert readers[0] is equation_fitness
        assert readers[1] is expected_reader

    @pytest.mark.parametrize('axis', ['instability', 'complexity'])
    def test_registered_readers_and_ideal_point_agree(self, axis):
        readers = _registered_readers(axis)
        assert objective_ideal_values(readers) == ideal_point(
            ('discrepancy', axis))

    def test_soeq_defers_to_the_config_when_unset(self):
        with using_config(second_objective='complexity'):
            assert _registered_readers(None)[1] is equation_complexity
        with using_config(second_objective='instability'):
            assert _registered_readers(None)[1] is equation_terms_stability

    @pytest.mark.parametrize('axis,filler_present', [('instability', True),
                                                     ('complexity', False)])
    def test_director_assembles_the_matching_filler(self, axis, filler_present,
                                                    monkeypatch):
        """Capture the objective list at assembly time -- the fitness host is
        buried as a suboperator, and what matters is which fillers it was
        built with."""
        from epde.optimizers.moeadd import strategy as strategy_mod

        built = []
        original = strategy_mod.SolverFreeFitness

        def _recording(*args, **kwargs):
            built.append([o.name for o in kwargs.get('objectives', ())])
            return original(*args, **kwargs)

        monkeypatch.setattr(strategy_mod, 'SolverFreeFitness', _recording)
        director = _fresh_director()
        director.use_baseline(second_objective=axis, params={})

        assert director.second_objective == axis
        # The first host built is the main fitness; a solver run also builds a
        # lightweight RPS host, which is fixed-role and not part of the front.
        names = built[0]
        assert names[0] == 'discrepancy'
        assert ('instability' in names) is filler_present

    def test_director_records_the_resolved_axis_when_deferring(self):
        with using_config(second_objective='complexity'):
            director = _fresh_director()
            director.use_baseline(second_objective=None, params={})
        assert director.second_objective == 'complexity'


class TestConsistencyCheck:
    """The runtime backstop: if the ideal point and the population's
    objectives ever disagree, say so loudly instead of optimizing towards a
    utopia point that describes a different front."""

    def _optimizer_stub(self, best_obj, readers):
        from epde.optimizers.moeadd.moeadd import MOEADDOptimizer

        solution = types.SimpleNamespace(obj_funs=readers)
        stub = types.SimpleNamespace(
            best_obj=best_obj,
            pareto_levels=types.SimpleNamespace(population=[solution],
                                                unplaced_candidates=[]))
        return MOEADDOptimizer._check_ideal_point_matches_objectives, stub

    def test_matching_ideal_point_passes(self):
        check, stub = self._optimizer_stub(
            [0.0, 1.0], [equation_fitness, equation_complexity])
        check(stub)

    def test_mismatched_ideal_point_is_caught(self):
        # Exactly the old bug: complexity on the axis, instability's ideal.
        check, stub = self._optimizer_stub(
            [0.0, 0.0], [equation_fitness, equation_complexity])
        with pytest.raises(AssertionError, match='desynced'):
            check(stub)

    def test_unknowable_ideal_is_not_a_mismatch(self):
        """generate_partial wrappers do not forward attributes; 'cannot tell'
        must stay distinct from 'disagrees'."""
        from functools import partial
        check, stub = self._optimizer_stub(
            [0.0, 1.0], [partial(equation_fitness, equation_key='u')])
        check(stub)

    def test_absent_ideal_point_is_tolerated(self):
        check, stub = self._optimizer_stub(
            None, [equation_fitness, equation_complexity])
        check(stub)


class TestEpdeSearchWiring:

    def test_axis_reaches_the_director(self):
        from epde.interface.interface import EpdeSearch
        search = EpdeSearch(second_objective='complexity',
                            verbose_params={'show_iter_idx': False})
        assert search._second_objective == 'complexity'
        assert search.director.second_objective == 'complexity'

    def test_default_axis_is_instability(self):
        from epde.interface.interface import EpdeSearch
        search = EpdeSearch(second_objective=None,
                            verbose_params={'show_iter_idx': False})
        assert search._second_objective == 'instability'
